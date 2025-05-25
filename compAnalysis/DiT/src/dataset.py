import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

class SatelliteDataset(Dataset):
    def __init__(self, files, in_dim=5, transform=None):
        self.files = files
        self.in_dim = in_dim
        self.transform = transform

    def scale_sat_img(self, img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        if img.shape[-1] == 6:
            img[..., 5] = img[..., 5] / 2.0 + 0.5
        np.clip(img, 0.0, 1.0, out=img)
        scaled_img = 2 * img - 1
        return scaled_img  # -1 ~ 1

    def scale_rainfall_img(self, mask):
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        np.clip(mask, 0.0, 100.0, out=mask)
        normalized_mask = mask / 50.0 - 1
        return normalized_mask  # -1 ~ 1

    def read_data(self, sat_files, radar_files):
        # Read satellite data
        sats = []
        sat_times = []
        for sat_file in sat_files:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    satcomp = sat_dset['satcomp'].values  # shape: (H, W, C1)
                    normed = sat_dset['normed'].values    # shape: (H, W, C=8)
                    normed_ltng = normed[:, :, 3:4]  # shape: (H, W, 1)
                    if self.in_dim == 5:  # Sat + radar
                        sat = np.concatenate([satcomp, normed_ltng], axis=-1)  # shape: (H, W, C1+1)
                    elif self.in_dim == 7:  # Sat (include sun) + radar
                        sun = normed[:, :, 6:8]
                        sat = np.concatenate([satcomp, normed_ltng, sun], axis=-1)  # shape: (H, W, C1+1+2)
                    sat = self.scale_sat_img(sat)
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
                rainfall = self.scale_rainfall_img(rainfall)
            except KeyError as e:
                raise ValueError(f"Missing expected radar variable: {e}")
        return concated_sat, rainfall, sat_time, radar_time
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sat_files, radar_files = self.files[idx]
        img, mask, sat_time, radar_time = self.read_data(sat_files, radar_files)
        
        #img = torch.from_numpy(img).float()
        #mask = torch.from_numpy(mask).float()
        sat_time = sat_time.astype('datetime64[s]').astype('int64')
        radar_time = radar_time.astype('datetime64[s]').astype('int64')
        
        return img, mask, sat_time, radar_time
    
# For old dataset
class SatelliteDataset_o(Dataset):
    def __init__(self, files, in_dim=5, transform=None):
        self.files = files
        self.in_dim = in_dim
        self.transform = transform

    def scale_sat_img(self, img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        np.clip(img, 0.0, 1.0, out=img)
        scaled_img = 2 * img - 1
        return scaled_img  # -1 ~ 1

    def scale_radar_img(self, mask):
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        np.clip(mask, 0.0, 60.0, out=mask)
        normalized_mask = mask / 30.0 - 1
        return normalized_mask  # -1 ~ 1

    def read_data(self, sat_files, radar_files):
        # Read satellite data
        sats = []
        sat_times = []
        for sat_file in sat_files:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    satcomp = sat_dset['satellite'].values  # shape: (H, W, C=4)
                    sat = self.scale_sat_img(satcomp)
                    sats.append(sat)
                    sat_times.append(sat_dset['date'].values)
                except KeyError as e:
                    raise ValueError(f"Missing expected satellite variable: {e}")
        concated_sat = np.concatenate(sats, axis=-1)  # shape: (H, W, C*(history_frames+1))
        sat_time = sat_times[-1]  # T0
        # Read radar data
        radar_file = radar_files[0]
        with xr.open_dataset(radar_file, engine='netcdf4') as radar_dset:
            try:
                radar = radar_dset['reflectivity'].values  # shape: (H, W)
                radar_time = radar_dset['date'].values
                if radar.ndim == 2:
                    radar = np.expand_dims(radar, axis=-1)  # make it (H, W, 1)
                radar = self.scale_radar_img(radar)
            except KeyError as e:
                raise ValueError(f"Missing expected radar variable: {e}")
        return concated_sat, radar, sat_time, radar_time
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sat_files, radar_files = self.files[idx]
        img, mask, sat_time, radar_time = self.read_data(sat_files, radar_files)
        
        #img = torch.from_numpy(img).float()
        #mask = torch.from_numpy(mask).float()
        sat_time = sat_time.astype('datetime64[s]').astype('int64')
        radar_time = radar_time.astype('datetime64[s]').astype('int64')
        
        return img, mask, sat_time, radar_time