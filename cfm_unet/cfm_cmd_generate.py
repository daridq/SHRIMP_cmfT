# --- START OF MODIFIED cfm_cmd_generate.py ---

import os
import hashlib
import itertools

# === Configuration === #
prefix = "CFMUExp" # Changed prefix to reflect CFM experiments
per_group = 1 # Number of commands per PBS job script

# === Base Params === #
# Using the same base_params as before, adjust as needed for your CFM runs
base_params = {
    "epochs": [100],
    "batch_size": [4],
    "in_dim": [4],
    "learning_rate": [0.0001],
    "device": ["GPU"],
    "sat_files_path": ["/g/data/kl02/vhl548/SHRIMP/noradar"],
    "rainfall_files_path": ["/g/data/kl02/vhl548/SHRIMP/radar/71"],
    "start_date": ["20210101"],
    "end_date": ["20210105"],
    "max_folders": [5],
    "history_frames": [0],
    "future_frame": [0],
    "refresh_rate": [10],
    "train_model": [True],
    "retrieve_dataset": [False],
    "load_model": [""],
    "cfm_mode": [True], # Keep experimenting with CFM on/off
    "cfm_sampling_steps": [100],
}

# === Command Generation Functions === #

def generate_label(params):
    excluded_keys_from_label = {"sat_files_path", "rainfall_files_path", "model_path", "results", "datasets"}
    label_params = {k: v for k, v in params.items() if k not in excluded_keys_from_label}
    label_str = "_".join(f"{k}={v}" for k, v in sorted(label_params.items()))
    return hashlib.md5(label_str.encode()).hexdigest()[:10]

def generate_cmds(base_params_dict):
    keys, values = zip(*base_params_dict.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    cmds_with_meta = []

    for params_combo in combinations:
        h = params_combo["history_frames"]
        f = params_combo["future_frame"]
        label = generate_label(params_combo)
        params_combo["label"] = label

        experiment_run_base_dir = f"./experiment_outputs/h{h}_f{f}_{label}"
        model_path = os.path.join(experiment_run_base_dir, "models")
        results_path = os.path.join(experiment_run_base_dir, "results_data")
        datasets_path = os.path.join(experiment_run_base_dir, "datasets_used")

        params_combo["model_path"] = model_path
        params_combo["results"] = results_path
        params_combo["datasets"] = datasets_path

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(datasets_path, exist_ok=True)
        os.makedirs("./job_logs", exist_ok=True)

        # MODIFICATION: Changed "shrimp_baseline.py" to "shrimp_cfm.py"
        cmd_lines = [f'python3 -u "./shrimp_cfm.py" \\'] # <-- MODIFIED SCRIPT NAME

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

        log_file_name = f"${{PBS_JOBID}}_{label}" if "PBS_JOBID" in os.environ else f"local_{label}"
        cmd_lines.append(f'    > ./job_logs/{log_file_name}.log 2>&1')

        cmd_str = "\n".join(cmd_lines)
        cmds_with_meta.append((cmd_str, h, f, label))
    return cmds_with_meta

def group_cmds(cmds_with_meta_list, per_group=per_group): # Corrected parameter name
    cmds_with_meta_list.sort(key=lambda x: (x[1], x[2], x[3]))
    sorted_cmds = [item[0] for item in cmds_with_meta_list]
    groups = [sorted_cmds[i:i + per_group] for i in range(0, len(sorted_cmds), per_group)]
    return groups

# === Script Generation === #

print("Generating experiment commands...")
all_cmds_with_meta = generate_cmds(base_params)
print(f"Generated {len(all_cmds_with_meta)} unique commands.")

if not all_cmds_with_meta:
    print("No commands were generated. Check base_params and logic.")
else:
    cmds_groups = group_cmds(all_cmds_with_meta, per_group=per_group)
    print(f"Grouped commands into {len(cmds_groups)} scripts, with up to {per_group} commands per script.")

    for g_id, group_of_cmds in enumerate(cmds_groups):
        script_file_name = f'{prefix}_Group{g_id}.sh' # Uses the new prefix
        script_content = [
            "#!/bin/bash",
            "#PBS -P jp09",
            "#PBS -q gpuvolta",
            "#PBS -l walltime=12:00:00",
            "#PBS -l storage=gdata/kl02+scratch/kl02",
            "#PBS -l ncpus=12",
            "#PBS -l ngpus=1",
            "#PBS -l mem=90GB",
            "#PBS -l jobfs=90GB",
            "#PBS -l wd",
            "#PBS -M hsun3103@uni.sydney.edu.au",
            "#PBS -m abe",
            f"#PBS -N {prefix}_G{g_id}", # Uses the new prefix
            "",
            "module purge",
            "module load use.own",
            "module load tensorflow/2.15.0",
            "",
            "echo \"Job $PBS_JOBID started at $(date)\"",
            "echo \"Working directory: $(pwd)\"",
            "echo \"Node: $(hostname)\"",
            "echo \"GPUs: $CUDA_VISIBLE_DEVICES\"",
            "",
            "mkdir -p ./experiment_outputs",
            "mkdir -p ./job_logs",
            ""
        ]
        for i, single_cmd in enumerate(group_of_cmds):
            script_content.append(f"echo \"Starting command {i+1} in group {g_id} at $(date)\"")
            script_content.append(single_cmd)
            if i != len(group_of_cmds) - 1:
                script_content.append("wait")
            script_content.append(f"echo \"Finished command {i+1} in group {g_id} at $(date)\"")
            script_content.append("")
        script_content.append('echo "All experiments in this group done at $(date)."')

        with open(script_file_name, "w") as f:
            for line in script_content:
                f.write(line + "\n")
        print(f"Generated PBS script: {script_file_name}")
        os.chmod(script_file_name, 0o755)
        print(f"Submitting {script_file_name} to PBS queue...")
        os.system(f"qsub {script_file_name}")

    print("\nScript generation complete. Review generated .sh files and submit them using 'qsub <script_name>.sh'")
    print("Output and log files will be generated in './experiment_outputs/' and './job_logs/' respectively.")

# --- END OF MODIFIED cfm_cmd_generate.py ---