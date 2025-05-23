import os
import json
import argparse
from argparse import ArgumentParser
import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import register_keras_serializable, get_registered_object, get_registered_name
from tensorflow.keras.optimizers import Adam
import re
from src.DatasetBuilder import DatasetBuilder


if __name__ == "__main__":

    # Parse tuple for dim_scales and input_shape
    def parse_tuple(s):
        try:
            # Remove brackets, spaces, convert to integers
            return tuple(map(int, s.strip().strip('()').replace(' ', '').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be a string of integers separated by commas, like '1, 2, 3'.")
    
    # Set default argument
    argparser = ArgumentParser()

    # Hyper-parameters
    argparser.add_argument("--epochs", default=1000, type=int)
    argparser.add_argument("--batch-size", default=4, type=int)
    argparser.add_argument("--in-dim", default=4, type=int)
    argparser.add_argument("--learning-rate", default=0.0001, type=float)
    
    # Control parameters for testing
    argparser.add_argument("--label", default="", type=str)
    argparser.add_argument("--device", default="GPU", type=str, choices=("GPU", "CPU"))
    argparser.add_argument("--sat-files-path", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--rainfall-files-path", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--start-date", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--end-date", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--max-folders", default=None, type=int)  # Sat Dataset path
    argparser.add_argument("--history-frames", default=0, type=int)  # history frames
    argparser.add_argument("--future-frame", default=0, type=int)  # predict one future frame
    argparser.add_argument("--refresh-rate", default=10, type=int)  # interval of frames
    argparser.add_argument("--datasets", default="", type=str)  # Saved datasets path
    argparser.add_argument("--model-path", default="", type=str)  # Saved model path
    argparser.add_argument("--results", default="", type=str)  # Test dataset sampling results
    argparser.add_argument("--train-model", action='store_true')  # store_true: default false=no train; store_false: default true=train
    argparser.add_argument("--retrieve-dataset", action='store_true')  # store_true: no retrieve; store_false: retrieve
    argparser.add_argument("--load-model", default="", type=str)
    args = argparser.parse_args()
    
    os.makedirs(args.model_path, exist_ok=True)
    
  
    # Save Hyper-parameters
    with open(os.path.join(args.model_path, f"arguments_{args.label}.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    # Print Hyper-parameters
    print("Model Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: \"{value}\"")

    devices = tf.config.list_physical_devices(args.device)
    logical_devices = tf.config.list_logical_devices(args.device)
    print(len(devices), f"Physical {args.device},", len(logical_devices), f"Logical {args.device}")
    if devices:
        for device in devices:
            try:
                tf.config.experimental.get_device_details(device)
                if device.device_type == "GPU":
                    tf.config.experimental.set_memory_growth(device, True)  # Set memory growth                
                    print(f"Memory growth enabled for {device.name}")
            except RuntimeError as e:
                print(e)
    # warnings.simplefilter("ignore")

    # Prepare dataset
    datasetbuilder = DatasetBuilder(
        sat_path=args.sat_files_path,
        radar_path=args.rainfall_files_path,
        start_date=args.start_date,
        end_date=args.end_date,
        max_folders=args.max_folders,
        history_frames=args.history_frames,
        future_frame=args.future_frame,
        refresh_rate=args.refresh_rate,
        coverage_threshold=0.05,
        seed=96
    )
    dataset_pkl_name = "dataset_filelist.pkl"
    dataset_pkl_path = os.path.join(args.model_path, dataset_pkl_name)
    if args.retrieve_dataset:
        train_files, val_files, test_files = datasetbuilder.load_filelist(dataset_pkl_path)
        print(f"Loaded existing dataset from {dataset_pkl_path}")
    else:
        train_files, val_files, test_files = datasetbuilder.build_filelist(
            save_dir=args.model_path,
            file_name=dataset_pkl_name,
            split_ratio=(0.7, 0.1, 0.2)
        )
        print(f"Built new dataset to {dataset_pkl_path}")

    # Model construction
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code
    img_rows = 128
    img_cols = 128
    smooth = 1.

    @register_keras_serializable(package='Custom', name='dice_coef')
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @register_keras_serializable(package='Custom', name='dice_coef_loss')
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    @register_keras_serializable(package='Custom', name='Hilburn_Loss')
    class Hilburn_Loss(tf.keras.losses.Loss):
        def __init__(self, b=5, c=3):
            super().__init__()
            self.b = b
            self.c = c
            
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_pred-y_true))
            weight = tf.math.exp( self.b * tf.math.pow(y_true, self.c) )
            #tf.print(tf.math.reduce_max(y_true))
            return tf.math.multiply_no_nan(weight, mse)
        
        def get_config(self):
            return {"b": self.b, "c": self.c}
    
    @register_keras_serializable(package='Custom', name='Custom_mse_loss')
    class Custom_mse_loss(tf.keras.losses.Loss):
        def __init__(self, lmbda=0.01):
            super().__init__()
            self.lmbda = lmbda
            
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.math.square(y_pred - y_true))
            zero = tf.fill(tf.shape(y_true), 0.0)
            bools = tf.cast(tf.math.greater(y_pred, y_true), tf.float32)
            reg = tf.math.reduce_sum(bools)
            #reg = tf.norm(y_pred, ord='euclidean')
            return mse + self.lmbda*reg
        
        def get_config(self):
            return {"lmbda": self.lmbda}
        
    assert get_registered_object('Custom>Hilburn_Loss') == Hilburn_Loss
    assert get_registered_name(Hilburn_Loss) == 'Custom>Hilburn_Loss'
    assert get_registered_object('Custom>Custom_mse_loss') == Custom_mse_loss
    assert get_registered_name(Custom_mse_loss) == 'Custom>Custom_mse_loss'
    
    # Model architecture
    def get_unet():
        inputs = Input((img_rows, img_cols, args.in_dim*(args.history_frames+1)), dtype=np.float32)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)    

        up5 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
        conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)    

        conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
        model = Model(inputs=[inputs], outputs=[conv8])
        
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss=Hilburn_Loss(),
            metrics=["mse"]
        )
        # model.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['mse']) 
        return model

    # Load Dataset        
    def scale_sat_img(img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        if img.shape[-1] == 6:
            img[..., 5] = img[..., 5] / 2.0 + 0.5
        np.clip(img, 0.0, 1.0, out=img)  # 0~1 for U-Net
        return img

    def scale_rainfall_img(mask):
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        np.clip(mask, 0.0, 200.0, out=mask)
        return mask / 200.0  # Normed to 0~1 for U-Net
    
    def read_data(sat_files, radar_files):
        # Read satellite data
        sats = []
        sat_times = []
        for sat_file in sat_files:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    satcomp = sat_dset['satcomp'].values  # shape: (H, W, C1)
                    normed = sat_dset['normed'].values    # shape: (H, W, C=8)
                    normed_ltng = normed[:, :, 3:4]  # shape: (H, W, 1)
                    if args.in_dim == 4:
                        sat = np.concatenate([satcomp, normed_ltng], axis=-1)  # shape: (H, W, C1+1)
                    elif args.in_dim == 6:
                        sun = normed[:, :, 6:8]
                        sat = np.concatenate([satcomp, normed_ltng, sun], axis=-1)  # shape: (H, W, C1+1+2)
                    sat = scale_sat_img(sat)
                    sats.append(sat)
                    sat_times.append(sat_dset['time'].values)
                except KeyError as e:
                    raise ValueError(f"Missing expected satellite variable: {e}")
        concated_sat = np.concatenate(sats, axis=-1)  # shape: (H, W, (C1+x)*(history_frames+1))
        sat_time = sat_times[-1]  # T0
        # Read radar data
        radar_file = radar_files[0]
        with xr.open_dataset(radar_file, engine='netcdf4') as radar_dset:
            try:
                rainfall = radar_dset['RAIN'].values  # shape: (H, W)
                radar_time = radar_dset['time'].values
                if rainfall.ndim == 2:
                    rainfall = np.expand_dims(rainfall, axis=-1)  # make it (H, W, 1)
                rainfall = scale_rainfall_img(rainfall)
            except KeyError as e:
                raise ValueError(f"Missing expected radar variable: {e}")
        return concated_sat, rainfall, sat_time, radar_time
    
    def load_data(files):
        bag = db.from_sequence(files).map(lambda f: read_data(f[0], f[1]))
        with ProgressBar():
            rslt = bag.compute()
        imgs = np.array([r[0] for r in rslt])
        masks = np.array([r[1] for r in rslt])
        img_times = np.array([r[2] for r in rslt])
        mask_times = np.array([r[3] for r in rslt])
        return imgs, masks, img_times, mask_times
    
    # Load existing model / Initiate model
    if args.load_model:
        custom_objects = {
            "dice_coef": dice_coef,
            "dice_coef_loss": dice_coef_loss,
            "Hilburn_Loss": Hilburn_Loss,
            "Custom_mse_loss": Custom_mse_loss
        }
        model = load_model(os.path.join(args.model_path, args.load_model), custom_objects=custom_objects)
        model.summary()
        loaded_epochs = int(re.search(r'epoch(\d+)', args.load_model).group(1))
    else:
        model = get_unet()
        model.summary()
        loaded_epochs = 0

    # Train
    if args.train_model:
        imgs_train, img_masks_train, *_ = load_data(train_files)
        imgs_val, img_masks_val, *_ = load_data(val_files)

        model_checkpoint = ModelCheckpoint(os.path.join(args.model_path, f"baseline_unet_best_{args.label}.keras"), monitor='val_loss', save_best_only=True)
        history = model.fit(
            imgs_train,
            img_masks_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            shuffle=True,
            validation_data=(imgs_val, img_masks_val),
            callbacks=[model_checkpoint]
        )
        model.save(os.path.join(args.model_path, f'baseline_unet_epoch{loaded_epochs+args.epochs}_{args.label}.keras'))
    
    # Test
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.datasets, exist_ok=True)
    os.makedirs(os.path.join(args.results, args.label), exist_ok=True)
    imgs_test, masks_test, img_times_test, mask_times_test = load_data(test_files)
    uoutputs = model.predict(imgs_test, verbose=1)
    
    np.save(os.path.join(args.datasets, f'sats_{args.in_dim}.npy'), imgs_test)
    np.save(os.path.join(args.datasets, f'sat_times.npy'), img_times_test)
    np.save(os.path.join(args.datasets, f'reals.npy'), masks_test)
    np.save(os.path.join(args.datasets, f'real_times.npy'), mask_times_test)
    np.save(os.path.join(args.results, args.label, f'uoutputs.npy'), uoutputs)