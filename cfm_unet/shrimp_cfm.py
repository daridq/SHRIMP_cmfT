# --- START OF FILE shrimp_cfm.py ---

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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import register_keras_serializable, get_registered_object, get_registered_name, Sequence
from tensorflow.keras.optimizers import Adam
import re

try:
    from src.DatasetBuilder import DatasetBuilder
except ImportError:
    from DatasetBuilder import DatasetBuilder


if __name__ == "__main__":

    argparser = ArgumentParser()

    # Hyper-parameters
    argparser.add_argument("--epochs", default=1000, type=int)
    argparser.add_argument("--batch-size", default=4, type=int)
    argparser.add_argument("--in-dim", default=4, type=int)
    argparser.add_argument("--learning-rate", default=0.0001, type=float)
    
    # Control parameters
    argparser.add_argument("--label", default="default_run", type=str)
    argparser.add_argument("--device", default="GPU", type=str, choices=("GPU", "CPU"))
    argparser.add_argument("--sat-files-path", default="", type=str)
    argparser.add_argument("--rainfall-files-path", default="", type=str)
    argparser.add_argument("--start-date", default="", type=str)
    argparser.add_argument("--end-date", default="", type=str)
    argparser.add_argument("--max-folders", default=None, type=int)
    argparser.add_argument("--history-frames", default=0, type=int)
    argparser.add_argument("--future-frame", default=0, type=int)
    argparser.add_argument("--refresh-rate", default=10, type=int)
    
    argparser.add_argument("--datasets", default="./output_data/datasets", type=str)
    argparser.add_argument("--model-path", default="./output_data/models", type=str)
    argparser.add_argument("--results", default="./output_data/results", type=str)

    argparser.add_argument("--train-model", action='store_true')
    argparser.add_argument("--retrieve-dataset", action='store_true')
    argparser.add_argument("--load-model", default="", type=str)

    # === CFM Specific Arguments ===
    argparser.add_argument("--cfm-mode", action='store_true', help="Enable Conditional Flow Matching mode.")
    argparser.add_argument("--cfm-sampling-steps", default=100, type=int, help="Number of steps for CFM ODE solver during inference.")
    
    args = argparser.parse_args()
    
    print(f"Ensuring output directory for models exists: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    print(f"Ensuring output directory for results exists: {args.results}")
    os.makedirs(args.results, exist_ok=True)
    print(f"Ensuring output directory for datasets (test arrays) exists: {args.datasets}")
    os.makedirs(args.datasets, exist_ok=True)
  
    hyperparams_save_path = os.path.join(args.model_path, f"arguments_{args.label}.json")
    try:
        with open(hyperparams_save_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Hyperparameters saved to: {hyperparams_save_path}")
    except Exception as e:
        print(f"Error saving hyperparameters to {hyperparams_save_path}: {e}")

    print("Effective Model Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: \"{value}\"")

    devices = tf.config.list_physical_devices(args.device)
    if devices:
        for device_item in devices: # Renamed 'device' to 'device_item' to avoid conflict with args.device
            try:
                if device_item.device_type == "GPU":
                    tf.config.experimental.set_memory_growth(device_item, True)                
                    print(f"Memory growth enabled for {device_item.name}")
            except RuntimeError as e:
                print(e)

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
    dataset_pkl_name = f"dataset_h{args.history_frames}_f{args.future_frame}_label{args.label}.pkl"
    dataset_pkl_path = os.path.join(args.model_path, dataset_pkl_name)

    if args.retrieve_dataset and os.path.exists(dataset_pkl_path):
        train_files, val_files, test_files = datasetbuilder.load_filelist(dataset_pkl_path)
        print(f"Loaded existing dataset from {dataset_pkl_path}")
    else:
        if args.retrieve_dataset:
            print(f"Warning: --retrieve-dataset was True, but {dataset_pkl_path} not found. Building new dataset.")
        train_files, val_files, test_files = datasetbuilder.build_filelist(
            save_dir=args.model_path, 
            file_name=dataset_pkl_name,
            split_ratio=(0.7, 0.1, 0.2)
        )
        print(f"Built new dataset to {dataset_pkl_path}")

    K.set_image_data_format('channels_last')
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
        def __init__(self, b=5, c=3, name="hilburn_loss"): 
            super().__init__(name=name)
            self.b = b
            self.c = c
            
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_pred-y_true))
            weight = tf.math.exp( self.b * tf.math.pow(tf.maximum(y_true, 0.0), self.c) ) 
            return tf.math.multiply_no_nan(weight, mse)
        
        def get_config(self):
            config = super().get_config()
            config.update({"b": self.b, "c": self.c})
            return config
    
    @register_keras_serializable(package='Custom', name='Custom_mse_loss')
    class Custom_mse_loss(tf.keras.losses.Loss):
        def __init__(self, lmbda=0.01, name="custom_mse_loss"): 
            super().__init__(name=name)
            self.lmbda = lmbda # Note: lmbda is not used in the simplified call below
            
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.math.square(y_pred - y_true))
            return mse
        
        def get_config(self):
            config = super().get_config()
            config.update({"lmbda": self.lmbda})
            return config
        
    sat_img_channels = args.in_dim * (args.history_frames + 1)

    def get_unet(is_cfm_mode=False):
        if is_cfm_mode:
            input_channels = 1 + sat_img_channels + 1 
            print(f"CFM Mode: U-Net input channels: {input_channels} (1 for x_t, {sat_img_channels} for sat_cond, 1 for time)")
        else: 
            input_channels = sat_img_channels
            print(f"Baseline Mode: U-Net input channels: {input_channels} (for sat_cond directly)")

        inputs = Input((img_rows, img_cols, input_channels), dtype=tf.float32)
        
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

        output_layer = Conv2D(1, (1, 1), activation='linear')(conv7) 
        model = Model(inputs=[inputs], outputs=[output_layer])
        return model
      
    def scale_sat_img(img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        if img.shape[-1] == 6: 
            img[..., 5] = img[..., 5] / 2.0 + 0.5
        np.clip(img, 0.0, 1.0, out=img)
        return img

    def scale_rainfall_img(mask): 
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        np.clip(mask, 0.0, 200.0, out=mask) 
        return mask / 200.0
    
    def read_data(sat_files_seq, radar_files_target):
        sats_c = []
        sat_times = []
        for sat_file in sat_files_seq:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    satcomp = sat_dset['satcomp'].values
                    normed = sat_dset['normed'].values
                    normed_ltng = normed[:, :, 3:4]
                    if args.in_dim == 4:
                        sat = np.concatenate([satcomp, normed_ltng], axis=-1)
                    elif args.in_dim == 6:
                        sun = normed[:, :, 6:8]
                        sat = np.concatenate([satcomp, normed_ltng, sun], axis=-1)
                    else: 
                        raise ValueError(f"Unsupported in_dim: {args.in_dim}. Expected 4 or 6.")
                    
                    sat = scale_sat_img(sat)
                    sats_c.append(sat)
                    sat_times.append(sat_dset['time'].values)
                except KeyError as e:
                    raise ValueError(f"Missing expected satellite variable in {sat_file}: {e}")
        
        concated_sat_c = np.concatenate(sats_c, axis=-1)
        if concated_sat_c.shape[-1] != sat_img_channels:
            raise ValueError(f"Satellite channel mismatch. Expected {sat_img_channels}, got {concated_sat_c.shape[-1]}")

        current_sat_time = sat_times[-1]
        
        radar_file_x1 = radar_files_target[0]
        with xr.open_dataset(radar_file_x1, engine='netcdf4') as radar_dset:
            try:
                rainfall_x1 = radar_dset['RAIN'].values
                radar_time = radar_dset['time'].values
                if rainfall_x1.ndim == 2:
                    rainfall_x1 = np.expand_dims(rainfall_x1, axis=-1)
                rainfall_x1 = scale_rainfall_img(rainfall_x1)
            except KeyError as e:
                raise ValueError(f"Missing expected radar variable 'RAIN' in {radar_file_x1}: {e}")
        return concated_sat_c, rainfall_x1, current_sat_time, radar_time
    
    def load_data_from_filelist(file_list_tuples):
        if not file_list_tuples:
            return np.array([]), np.array([]), np.array([]), np.array([])

        bag = db.from_sequence(file_list_tuples).map(lambda f_tuple: read_data(f_tuple[0], f_tuple[1]))
        with ProgressBar():
            rslt = bag.compute()
        
        sats_c_arr = np.array([r[0] for r in rslt if r is not None and r[0] is not None])
        masks_x1_arr = np.array([r[1] for r in rslt if r is not None and r[1] is not None])
        img_times_arr = np.array([r[2] for r in rslt if r is not None and r[2] is not None])
        mask_times_arr = np.array([r[3] for r in rslt if r is not None and r[3] is not None])
        
        return sats_c_arr, masks_x1_arr, img_times_arr, mask_times_arr

    class CFMDataGenerator(Sequence):
        def __init__(self, sat_images_c, rain_images_x1, batch_size, img_rows_local, img_cols_local, sat_img_channels_count, shuffle=True, epsilon=1e-5): # Renamed img_rows, img_cols to avoid conflict
            self.sat_images_c = sat_images_c
            self.rain_images_x1 = rain_images_x1
            self.batch_size = batch_size
            self.img_rows_local = img_rows_local
            self.img_cols_local = img_cols_local
            self.sat_img_channels_count = sat_img_channels_count
            self.shuffle = shuffle
            self.epsilon = epsilon
            self.indexes = np.arange(len(self.rain_images_x1))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __len__(self):
            return int(np.floor(len(self.rain_images_x1) / self.batch_size))

        def __getitem__(self, index):
            batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            c_batch = self.sat_images_c[batch_indexes]
            x1_batch = self.rain_images_x1[batch_indexes]
            x0_batch = np.random.normal(size=x1_batch.shape).astype(np.float32)
            t_scalar_batch = np.random.uniform(
                low=self.epsilon, 
                high=1.0 - self.epsilon, 
                size=(len(batch_indexes), 1, 1, 1)
            ).astype(np.float32)
            xt_batch = (1.0 - t_scalar_batch) * x0_batch + t_scalar_batch * x1_batch
            ut_batch_target = x1_batch - x0_batch
            t_channel_batch = np.ones_like(x1_batch) * t_scalar_batch 
            model_input_batch = np.concatenate([xt_batch, c_batch, t_channel_batch], axis=-1)
            return model_input_batch, ut_batch_target

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indexes)

    def sample_cfm_euler(model_cfm, c_condition_single, num_steps, img_shape_target, sat_img_channels_count, current_img_rows, current_img_cols):
        x_t_current = np.random.normal(size=img_shape_target).astype(np.float32) 
        dt = 1.0 / num_steps
        c_condition_batch = np.expand_dims(c_condition_single, axis=0)

        for i in range(num_steps):
            t_current_scalar = i * dt
            t_channel_for_input = np.full(
                (1, current_img_rows, current_img_cols, 1), 
                t_current_scalar, 
                dtype=np.float32
            )
            x_t_current_batch = np.expand_dims(x_t_current, axis=0)
            model_input_this_step = np.concatenate([
                x_t_current_batch, 
                c_condition_batch, 
                t_channel_for_input
            ], axis=-1)
            v_predicted_batch = model_cfm.predict_on_batch(model_input_this_step)
            v_predicted_single = v_predicted_batch[0]
            x_t_current = x_t_current + v_predicted_single * dt
        return x_t_current

    custom_objects_dict = {
        "dice_coef": dice_coef,
        "dice_coef_loss": dice_coef_loss,
        "Hilburn_Loss": Hilburn_Loss,
        "Custom_mse_loss": Custom_mse_loss
    }
    loaded_epochs = 0
    model = None

    if args.load_model:
        model_load_path = args.load_model
        if not os.path.exists(model_load_path):
             model_load_path = os.path.join(args.model_path, args.load_model)

        print(f"Attempting to load model from: {model_load_path}")
        if os.path.exists(model_load_path):
            model = load_model(model_load_path, custom_objects=custom_objects_dict)
            model.summary()
            try:
                match = re.search(r'epoch(\d+)', args.load_model)
                if match:
                    loaded_epochs = int(match.group(1))
            except:
                print("Could not parse epoch number from loaded model filename.")
        else:
            print(f"Error: Model file not found at {model_load_path}. Initializing new model.")
            args.load_model = "" 

    if model is None:
        print("Initializing new U-Net model.")
        model = get_unet(is_cfm_mode=args.cfm_mode)
        model.summary()
        loaded_epochs = 0

    if args.cfm_mode:
        print("CFM Mode: Compiling model with MSE loss.")
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss='mse', 
            metrics=["mae"] 
        )
    else: 
        print("Baseline Mode: Compiling model with Hilburn_Loss.")
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss=Hilburn_Loss(),
            metrics=["mse", dice_coef] 
        )

    if args.train_model:
        print("Loading training and validation data...")
        imgs_train_c, imgs_masks_train_x1, _, _ = load_data_from_filelist(train_files) # Assign to _ if not used
        imgs_val_c, imgs_masks_val_x1, _, _ = load_data_from_filelist(val_files)     # Assign to _ if not used

        if imgs_train_c.size == 0 or imgs_val_c.size == 0:
             print("ERROR: Training or validation data is empty. Halting training.")
        else:
            print(f"Training data shape (conditions_c): {imgs_train_c.shape}")
            print(f"Training data shape (targets_x1): {imgs_masks_train_x1.shape}")
            print(f"Validation data shape (conditions_c): {imgs_val_c.shape}")
            print(f"Validation data shape (targets_x1): {imgs_masks_val_x1.shape}")

            model_checkpoint_filename = f"{'cfm' if args.cfm_mode else 'baseline'}_unet_best_val_loss_{args.label}.keras"
            model_checkpoint_path = os.path.join(args.model_path, model_checkpoint_filename)
            model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
            
            print(f"Starting training for {args.epochs} epochs...")
            if args.cfm_mode:
                print("Using CFMDataGenerator for training.")
                train_generator = CFMDataGenerator(
                    imgs_train_c, imgs_masks_train_x1, args.batch_size, 
                    img_rows, img_cols, sat_img_channels, shuffle=True
                )
                val_generator = CFMDataGenerator(
                    imgs_val_c, imgs_masks_val_x1, args.batch_size, 
                    img_rows, img_cols, sat_img_channels, shuffle=False
                )
                history = model.fit( # Storing history if needed for plotting later
                    train_generator,
                    epochs=args.epochs,
                    validation_data=val_generator,
                    callbacks=[model_checkpoint],
                    verbose=1
                )
            else: 
                print("Using direct numpy arrays for training (baseline mode).")
                history = model.fit( # Storing history
                    imgs_train_c, 
                    imgs_masks_train_x1, 
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    verbose=1,
                    shuffle=True,
                    validation_data=(imgs_val_c, imgs_masks_val_x1),
                    callbacks=[model_checkpoint]
                )
            
            final_model_filename = f"{'cfm' if args.cfm_mode else 'baseline'}_unet_epoch{loaded_epochs+args.epochs}_{args.label}.keras"
            final_model_save_path = os.path.join(args.model_path, final_model_filename)
            model.save(final_model_save_path)
            print(f"Training complete. Final model saved to: {final_model_save_path}")
    
    print("Loading test data for evaluation...")
    imgs_test_c, masks_test_x1, img_times_test, mask_times_test = load_data_from_filelist(test_files)
    
    if imgs_test_c.size == 0: 
        print("Warning: Test data is empty. Skipping evaluation.")
    else:
        print(f"Test data loaded: {imgs_test_c.shape[0]} samples.")
        
        uoutputs = None 

        if args.cfm_mode:
            if not model: 
                print("ERROR: CFM mode inference requires a loaded model, but no model is available.")
            else:
                print(f"CFM Mode: Generating predictions using Euler sampler with {args.cfm_sampling_steps} steps.")
                cfm_predictions_list = []
                target_rainfall_shape = (img_rows, img_cols, 1)

                for i in range(len(imgs_test_c)):
                    print(f"  Sampling CFM for test image {i+1}/{len(imgs_test_c)}...")
                    c_single_condition = imgs_test_c[i]
                    
                    predicted_map = sample_cfm_euler(
                        model, 
                        c_single_condition, 
                        args.cfm_sampling_steps, 
                        target_rainfall_shape, 
                        sat_img_channels,
                        img_rows,
                        img_cols
                    )
                    cfm_predictions_list.append(predicted_map)
                uoutputs = np.array(cfm_predictions_list)
        else: 
            if not model:
                print("ERROR: Baseline mode inference requires a loaded model, but no model is available.")
            else:
                if args.cfm_sampling_steps != argparser.get_default("cfm_sampling_steps"):
                     print(f"Note: Baseline mode is active. --cfm-sampling-steps ({args.cfm_sampling_steps}) will be ignored.")
                print("Baseline Mode: Generating predictions with model.predict().")
                uoutputs = model.predict(imgs_test_c, batch_size=args.batch_size, verbose=1)
        
        if uoutputs is not None: # Proceed only if predictions were generated
            prefix = "cfm_" if args.cfm_mode else "baseline_"

            predictions_savename = f'{prefix}uoutputs_label{args.label}.npy'
            predictions_savepath = os.path.join(args.results, predictions_savename)
            np.save(predictions_savepath, uoutputs)
            print(f"Predictions saved to: {predictions_savepath}")

            inputs_savename = f'{prefix}sats_inputs_test_label{args.label}_indim{args.in_dim}.npy'
            inputs_savepath = os.path.join(args.datasets, inputs_savename)
            np.save(inputs_savepath, imgs_test_c)
            print(f"Test inputs saved to: {inputs_savepath}")

            reals_savename = f'{prefix}reals_targets_test_label{args.label}.npy'
            reals_savepath = os.path.join(args.datasets, reals_savename)
            np.save(reals_savepath, masks_test_x1)
            print(f"Test ground truth saved to: {reals_savepath}")
            
            print(f"Test data arrays (inputs, reals) saved in: {args.datasets}")
        else:
            print("No predictions were generated (uoutputs is None). Skipping saving of results.")


    print(f"Script for label '{args.label}' finished.")

# --- END OF FILE shrimp_cfm.py ---