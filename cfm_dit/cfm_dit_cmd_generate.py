#!/usr/bin/env python3
# --- START OF CFM_DiT Command Generation Script ---

import os
import hashlib
import itertools

# === Configuration === #
prefix = "CFMDiT" # Flow Matching with DiT experiments
per_group = 1 # Number of commands per PBS job script

# === Base Params for Flow Matching + DiT === #
base_params = {
    # Basic training parameters
    "epochs": [50],
    "batch_size": [4],
    "in_dim": [5],  # satellite + radar channels
    "out_dim": [1],  # radar output
    
    # DiT model configurations
    "dit_model": ["DiT-L/4"],
    
    # Flow Matching specific parameters
    "sigma_min": [1e-4],
    "sigma_max": [1.0],
    "rho": [7.0],  # Time distribution parameter
    "target_type": ["velocity"],  # Focus on velocity field learning
    "solver_type": ["heun"],  # ODE solver types
    "sampling_timesteps": [50],  # Different sampling speeds
    
    # Training configuration
    "loss_type": ["Hilburn_Loss"],
    "learning_rate": [0.0001],
    "gf_sigmat": [0],  # Gaussian filter for training noise
    "gf_sigma1": [0],  # Gaussian filter for initial sampling noise
    "gf_sigma2": [0],  # Gaussian filter for intermediate sampling
    "cfg_scale": [1.0],  # Classifier-free guidance scale
    
    # Data parameters
    "device": ["cuda"],
    "sat_files_path": ["/g/data/kl02/vhl548/SHRIMP/noradar"],
    "rainfall_files_path": ["/g/data/kl02/vhl548/SHRIMP/radar/71"],
    "start_date": ["20210101"],
    "end_date": ["20210110"],  # Longer period for better training
    "max_folders": [20],
    "history_frames": [0],  # Experiment with temporal history
    "future_frame": [1],    # Different prediction horizons
    "refresh_rate": [10],
    
    # Execution control
    "train_model": [True],
    "retrieve_dataset": [False],
    "load_model": [""],
}
# === Experimental Configurations === #
# Define specific experimental groups for targeted comparisons



# Solver comparison experiments
solver_comparison_params = {
    "epochs": [50],
    "batch_size": [4],
    "in_dim": [5],
    "out_dim": [1],
    "input_shape": ["(5,128,128)"],
    "dit_model": ["DiT-B/2"],
    "sigma_min": [1e-4],
    "sigma_max": [1.0],
    "rho": [7.0],
    "target_type": ["velocity"],
    "solver_type": ["euler", "heun"],  # Compare solvers
    "sampling_timesteps": [10, 20, 50],  # Different sampling speeds
    "loss_type": ["Hilburn_Loss"],
    "learning_rate": [0.0001],
    "gf_sigmat": [0.5],
    "cfg_scale": [1.0],
    "device": ["cuda"],
    "sat_files_path": ["/g/data/kl02/vhl548/SHRIMP/noradar"],
    "rainfall_files_path": ["/g/data/kl02/vhl548/SHRIMP/radar/71"],
    "start_date": ["20210101"],
    "end_date": ["20210107"],
    "max_folders": [10],
    "history_frames": [0],
    "future_frame": [1],
    "refresh_rate": [10],
    "train_model": [True],
    "retrieve_dataset": [False],
    "load_model": [""],
}

quick_params = solver_comparison_params

# === Command Generation Functions === #

def generate_label(params):
    """Generate unique label for parameter combination."""
    excluded_keys_from_label = {
        "sat_files_path", "rainfall_files_path", "model_path", 
        "results", "datasets", "device"
    }
    label_params = {k: v for k, v in params.items() if k not in excluded_keys_from_label}
    label_str = "_".join(f"{k}={v}" for k, v in sorted(label_params.items()))
    return hashlib.md5(label_str.encode()).hexdigest()[:12]  # Longer hash for uniqueness

def generate_cmds(base_params_dict, experiment_name="standard"):
    """Generate commands for all parameter combinations."""
    keys, values = zip(*base_params_dict.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    cmds_with_meta = []

    for params_combo in combinations:
        h = params_combo["history_frames"]
        f = params_combo["future_frame"]
        solver = params_combo["solver_type"]
        model = params_combo["dit_model"].replace("/", "_")
        
        label = generate_label(params_combo)
        params_combo["label"] = label

        # Create experiment directory structure
        experiment_run_base_dir = f"./cfm_dit_experiments/{experiment_name}/h{h}_f{f}_{solver}_{model}_{label}"
        model_path = os.path.join(experiment_run_base_dir, "models")
        results_path = os.path.join(experiment_run_base_dir, "results_data")
        datasets_path = os.path.join(experiment_run_base_dir, "datasets_used")

        params_combo["model_path"] = model_path
        params_combo["results"] = results_path
        params_combo["datasets"] = datasets_path

        # Create directories
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(datasets_path, exist_ok=True)
        os.makedirs("./cfm_dit_job_logs", exist_ok=True)

        # Generate command using shrimp_cfm_dit.py
        cmd_lines = [f'python3 -u "./shrimp_cfm_dit.py" \\']

        sorted_params = sorted(params_combo.items())
        for k, v_val in sorted_params:
            arg_name = k.replace("_", "-")
            if isinstance(v_val, bool):
                if v_val:
                    cmd_lines.append(f'    --{arg_name} \\')
            elif isinstance(v_val, (int, float, str)):
                if v_val == "" and k == "load_model":
                     cmd_lines.append(f'    --{arg_name} "" \\')
                elif v_val != "":
                    cmd_lines.append(f'    --{arg_name} "{v_val}" \\')
                elif k != "load_model":
                    pass
            elif isinstance(v_val, (tuple, list)):
                cmd_lines.append(f'    --{arg_name} "{tuple(v_val)}" \\')

        # Log file naming
        log_file_name = f"${{PBS_JOBID}}_{experiment_name}_{label}" if "PBS_JOBID" in os.environ else f"local_{experiment_name}_{label}"
        cmd_lines.append(f'    > ./cfm_dit_job_logs/{log_file_name}.log 2>&1')

        cmd_str = "\n".join(cmd_lines)
        cmds_with_meta.append((cmd_str, h, f, solver, model, label))
    
    return cmds_with_meta

def group_cmds(cmds_with_meta_list, per_group=per_group):
    """Group commands for batch execution."""
    # Sort by (history_frames, future_frame, solver, model, label)
    cmds_with_meta_list.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5]))
    sorted_cmds = [item[0] for item in cmds_with_meta_list]
    groups = [sorted_cmds[i:i + per_group] for i in range(0, len(sorted_cmds), per_group)]
    return groups

def generate_pbs_script(group_of_cmds, group_id, experiment_name, walltime="24:00:00", mem="90GB"):
    """Generate PBS script for a group of commands."""
    script_file_name = f'{prefix}_{experiment_name}_Group{group_id}.sh'
    script_content = [
        "#!/bin/bash",
        "#PBS -P jp09",
        "#PBS -q gpuvolta",
        f"#PBS -l walltime={walltime}",
        "#PBS -l storage=gdata/kl02+scratch/kl02",
        "#PBS -l ncpus=12",
        "#PBS -l ngpus=1",
        f"#PBS -l mem={mem}",
        "#PBS -l jobfs=90GB",
        "#PBS -l wd",
        "#PBS -M junyanbai0918@gmail.com",
        "#PBS -m abe",
        f"#PBS -N {prefix}_{experiment_name}_G{group_id}",
        "",
        "module purge",
        "module load use.own",
        "module load python3/3.9.2",
        "",
        "echo \"Flow Matching + DiT Job $PBS_JOBID started at $(date)\"",
        "echo \"Working directory: $(pwd)\"",
        "echo \"Node: $(hostname)\"",
        "echo \"GPUs: $CUDA_VISIBLE_DEVICES\"",
        "echo \"Experiment: " + experiment_name + "\"",
        "",
        "mkdir -p ./cfm_dit_experiments",
        "mkdir -p ./cfm_dit_job_logs",
        ""
    ]
    
    for i, single_cmd in enumerate(group_of_cmds):
        script_content.append(f"echo \"Starting Flow Matching command {i+1} in group {group_id} at $(date)\"")
        script_content.append(single_cmd)
        if i != len(group_of_cmds) - 1:
            script_content.append("wait")
        script_content.append(f"echo \"Finished Flow Matching command {i+1} in group {group_id} at $(date)\"")
        script_content.append("")
    
    script_content.append('echo "All Flow Matching experiments in this group done at $(date)."')
    
    return script_file_name, script_content

# === Main Execution === #

def run_experiment(experiment_name, params_dict, walltime="24:00:00", mem="90GB", submit_jobs=True):
    """Run a specific experiment configuration."""
    print(f"\n🌊 Generating Flow Matching + DiT experiment: {experiment_name}")
    print("=" * 60)
    
    all_cmds_with_meta = generate_cmds(params_dict, experiment_name)
    print(f"Generated {len(all_cmds_with_meta)} unique Flow Matching commands.")

    if not all_cmds_with_meta:
        print("No commands were generated. Check params and logic.")
        return

    cmds_groups = group_cmds(all_cmds_with_meta, per_group=per_group)
    print(f"Grouped commands into {len(cmds_groups)} scripts, with up to {per_group} commands per script.")

    for g_id, group_of_cmds in enumerate(cmds_groups):
        script_file_name, script_content = generate_pbs_script(
            group_of_cmds, g_id, experiment_name, walltime, mem
        )
        
        with open(script_file_name, "w") as f:
            for line in script_content:
                f.write(line + "\n")
        
        print(f"Generated PBS script: {script_file_name}")
        os.chmod(script_file_name, 0o755)
        
        if submit_jobs:
            print(f"Submitting {script_file_name} to PBS queue...")
            os.system(f"qsub {script_file_name}")
        else:
            print(f"Script ready for manual submission: qsub {script_file_name}")

if __name__ == "__main__":
    import sys
    
    print("🚀 Flow Matching + DiT Experiment Generator")
    print("=" * 50)
    
    # Command line options
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        experiment_type = "standard"
    
    # Determine which experiment to run
    if experiment_type == "quick":
        print("🏃 Running quick test experiments...")
        run_experiment("quick", quick_params, walltime="2:00:00", mem="32GB", submit_jobs=False)
        
    elif experiment_type == "solver":
        print("🔬 Running solver comparison experiments...")
        run_experiment("solver_comparison", solver_comparison_params, walltime="12:00:00", mem="64GB")
        
    elif experiment_type == "full":
        print("🎯 Running full parameter sweep...")
        run_experiment("full_sweep", base_params, walltime="24:00:00", mem="128GB")
        
    elif experiment_type == "test":
        print("🧪 Running test mode (no submission)...")
        run_experiment("test", quick_params, walltime="1:00:00", mem="16GB", submit_jobs=False)
        
    else:
        print("📊 Running standard experiments...")
        # Default: moderate experiment set
        standard_params = {
            "epochs": [100],
            "batch_size": [4],
            "in_dim": [5],
            "out_dim": [1],
            "input_shape": ["(5,128,128)"],
            "dit_model": ["DiT-B/2"],
            "sigma_min": [1e-4],
            "sigma_max": [1.0],
            "rho": [7.0],
            "target_type": ["velocity"],
            "solver_type": ["heun"],
            "sampling_timesteps": [50],
            "loss_type": ["Hilburn_Loss"],
            "learning_rate": [0.0001],
            "gf_sigmat": [0.5],
            "cfg_scale": [1.0],
            "device": ["cuda"],
            "sat_files_path": ["/g/data/kl02/vhl548/SHRIMP/noradar"],
            "rainfall_files_path": ["/g/data/kl02/vhl548/SHRIMP/radar/71"],
            "start_date": ["20210101"],
            "end_date": ["20210107"],
            "max_folders": [15],
            "history_frames": [0],
            "future_frame": [1],
            "refresh_rate": [10],
            "train_model": [True],
            "retrieve_dataset": [False],
            "load_model": [""],
        }
        run_experiment("standard", standard_params, walltime="24:00:00", mem="90GB")

    print("\n✅ Script generation complete!")
    print("\n📚 Usage Examples:")
    print("  python cfm_dit_cmd_generate.py quick     # Quick test (no submission)")
    print("  python cfm_dit_cmd_generate.py solver    # Solver comparison")
    print("  python cfm_dit_cmd_generate.py full      # Full parameter sweep")
    print("  python cfm_dit_cmd_generate.py test      # Test mode (no submission)")
    print("  python cfm_dit_cmd_generate.py standard  # Default experiments")
    print("\n📁 Output directories:")
    print("  ./cfm_dit_experiments/     # Experiment results")
    print("  ./cfm_dit_job_logs/        # Job logs")
    print("  ./*CFMDiT*.sh              # PBS scripts")

# --- END OF CFM_DiT Command Generation Script --- 