import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration: Set these paths according to your output ---

# Replace with the specific label of the experiment you want to inspect
experiment_label = "c322ba9918" # e.g., "abcdef1234" from cfm_cmd_generate.py
history_frames_val = 0 # The h value for the experiment, e.g., 1
future_frame_val = 0   # The f value for the experiment, e.g., 1
input_dim_val = 4      # The in_dim value for the experiment, e.g., 4
is_cfm_experiment = True # Set to True if it was a CFM run, False for baseline

# Base directory where experiment_outputs are stored
base_output_dir = "./experiment_outputs"

# Construct paths based on the configuration
experiment_run_folder = f"h{history_frames_val}_f{future_frame_val}_{experiment_label}"
results_data_path = os.path.join(base_output_dir, experiment_run_folder, "results_data")
datasets_used_path = os.path.join(base_output_dir, experiment_run_folder, "datasets_used")

prefix = "cfm_" if is_cfm_experiment else "baseline_"

# File names
predictions_filename = f"{prefix}uoutputs_label{experiment_label}.npy"
ground_truth_filename = f"{prefix}reals_targets_test_label{experiment_label}.npy"
satellite_input_filename = f"{prefix}sats_inputs_test_label{experiment_label}_indim{input_dim_val}.npy"

predictions_filepath = os.path.join(results_data_path, predictions_filename)
ground_truth_filepath = os.path.join(datasets_used_path, ground_truth_filename)
satellite_input_filepath = os.path.join(datasets_used_path, satellite_input_filename)

# --- Function to load a .npy file ---
def load_npy_data(filepath, description="data"):
    """Loads an .npy file and prints basic info."""
    print(f"Attempting to load {description} from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        data_array = np.load(filepath)
        print(f"Successfully loaded {description}.")
        print(f"  Shape: {data_array.shape}")
        print(f"  Data type: {data_array.dtype}")
        print(f"  Min value: {np.min(data_array):.4f}, Max value: {np.max(data_array):.4f}, Mean value: {np.mean(data_array):.4f}")
        return data_array
    except Exception as e:
        print(f"Error loading {description} from {filepath}: {e}")
        return None

# --- Load the data ---
predicted_rainfall = load_npy_data(predictions_filepath, "Predicted Rainfall (uoutputs)")
actual_rainfall = load_npy_data(ground_truth_filepath, "Actual Rainfall (reals_targets)")
satellite_images = load_npy_data(satellite_input_filepath, "Satellite Inputs (sats_inputs)")

# --- Visualization ---
if predicted_rainfall is not None:
    num_samples = predicted_rainfall.shape[0]
    print(f"\nFound {num_samples} samples in the predictions.")

    # Choose which sample to visualize
    sample_index_to_show = 0  # You can change this to see other samples (e.g., 0, 1, 2, ...)

    if num_samples > sample_index_to_show:
        pred_sample = predicted_rainfall[sample_index_to_show]

        # Assuming predictions are (Height, Width, 1) or (Height, Width)
        if pred_sample.ndim == 3 and pred_sample.shape[-1] == 1:
            pred_sample_squeezed = np.squeeze(pred_sample, axis=-1)
        elif pred_sample.ndim == 2:
            pred_sample_squeezed = pred_sample
        else:
            print(f"Unexpected shape for predicted sample: {pred_sample.shape}")
            pred_sample_squeezed = None

        if pred_sample_squeezed is not None:
            # Determine number of plots needed
            num_plots = 1
            plot_titles = ["Model Prediction"]
            images_to_plot = [pred_sample_squeezed]
            cmaps = ['viridis'] # Colormap for rainfall

            # Add ground truth if available
            if actual_rainfall is not None and actual_rainfall.shape[0] > sample_index_to_show:
                actual_sample = actual_rainfall[sample_index_to_show]
                if actual_sample.ndim == 3 and actual_sample.shape[-1] == 1:
                    actual_sample_squeezed = np.squeeze(actual_sample, axis=-1)
                elif actual_sample.ndim == 2:
                    actual_sample_squeezed = actual_sample
                else:
                    actual_sample_squeezed = None
                
                if actual_sample_squeezed is not None:
                    num_plots += 1
                    plot_titles.append("Ground Truth")
                    images_to_plot.append(actual_sample_squeezed)
                    cmaps.append('viridis')


            # Add one channel of satellite input if available
            # Satellite input shape: (num_samples, history_frames+1, H, W, in_dim_per_frame)
            # OR (num_samples, H, W, total_sat_channels) if already concatenated by read_data
            # The saved satellite_images array is (num_samples, H, W, total_sat_channels)
            if satellite_images is not None and satellite_images.shape[0] > sample_index_to_show:
                sat_sample = satellite_images[sample_index_to_show] # Shape (H, W, total_sat_channels)
                
                # Display the first channel of the *current* satellite frame as an example
                # Assuming history_frames = 0 means sat_sample has in_dim_val channels.
                # If history_frames > 0, total_sat_channels = in_dim_val * (history_frames_val + 1)
                # We can pick one channel from this combined stack. Let's pick the very first one.
                if sat_sample.shape[-1] > 0:
                    num_plots += 1
                    plot_titles.append(f"Satellite Input (Channel 0 of T0)")
                    # The channels are already concatenated in sat_sample
                    # sat_channel_0_T0 = sat_sample[..., 0] # Display the first channel of the whole stack
                    
                    # More specific: if history_frames > 0, the last 'in_dim_val' channels are from T0
                    # If history_frames = 0, all channels are T0
                    if history_frames_val == 0:
                        channel_to_show_from_sat = sat_sample[..., 0] # First channel of T0
                    else:
                        # Last block of channels corresponds to T0 satellite image
                        t0_channels_start_index = input_dim_val * history_frames_val
                        channel_to_show_from_sat = sat_sample[..., t0_channels_start_index] # First channel of T0 block
                    
                    images_to_plot.append(channel_to_show_from_sat)
                    cmaps.append('gray')


            # Create the plot
            fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
            if num_plots == 1: # Handle case for a single subplot
                axes = [axes]

            for i in range(num_plots):
                im = axes[i].imshow(images_to_plot[i], cmap=cmaps[i])
                axes[i].set_title(f"{plot_titles[i]}\nSample Index: {sample_index_to_show}")
                axes[i].axis('off')
                fig.colorbar(im, ax=axes[i], shrink=0.8)
            
            plt.tight_layout()
            plt.suptitle(f"Experiment: {experiment_label}, h={history_frames_val}, f={future_frame_val}", fontsize=14)
            fig.subplots_adjust(top=0.85) # Adjust top to make space for suptitle
            plt.show()

    else:
        print(f"Sample index {sample_index_to_show} is out of bounds for {num_samples} available samples.")
else:
    print("\nCannot visualize predictions as they were not loaded.")

print("\nVisualization script finished.")