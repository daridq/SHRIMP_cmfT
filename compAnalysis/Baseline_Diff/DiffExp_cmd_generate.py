import os
import hashlib
import itertools

# === Configuration === #
prefix = "DiffExp"
per_group = 1

# === Base Params === #
base_params = {
    "epochs": [400],
    "batch_size": [4],
    "timesteps": [1000],
    "sampling_timesteps": [500],
    "in_dim": [5],  # input_shape will be auto calculated
    "out_dim": [1],
    "embed_dim": [64],
    "dim_scales": [(1,2,4,8)],
    "target_type": ["pred_x_0"],
    "loss_type": ["l2"],
    "gamma_type": ["ddim"],
    "noise_schedule_type": ["linear"],
    "learning_rate": [0.0001],
    "gf_sigmat": [0.0],
    "gf_sigma1": [0.0],
    "gf_sigma2": [0.0],
    "no_clp": [False],
    "device": ["cuda"],
    "num_workers": [2],
    "sat_files_path": ["/g/data/kl02/vhl548/SHRIMP/noradar"],
    "rainfall_files_path": ["/g/data/kl02/vhl548/SHRIMP/radar/71"],
    "start_date": ["20210101"],
    "end_date": ["20210401"],
    "max_folders": [180],
    "history_frames": [1, 6],
    "future_frame": [1],
    "refresh_rate": [10],
    "train_model": [True],
    "retrieve_dataset": [False],
    "load_model": [""],
}

# === Command Generation Functions === #

def compute_input_shape(in_dim, history_frames):
    first_dim = (in_dim - 1) * (history_frames + 1) + 1
    return (first_dim, 128, 128)

def generate_label(params):
    label_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.md5(label_str.encode()).hexdigest()

def generate_cmds(base_params):
    keys, values = zip(*base_params.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    cmds = []
    for params in combinations:
        h = params["history_frames"]
        f = params["future_frame"]
        v = params["in_dim"] - 1

        input_shape = compute_input_shape(params["in_dim"], h)
        params["input_shape"] = input_shape

        label = generate_label(params)

        model_path = f"./h{h}-f{f}/models"
        results_path = f"./h{h}-f{f}/results"
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        params["label"] = label
        params["model_path"] = model_path
        params["results"] = results_path

        cmd_lines = [f'python3 -u "./shrimp.py" \\']
        cmd_lines.append(f'    --label "{label}" \\')

        for k, v in params.items():
            if k == "label":
                continue
            if isinstance(v, bool):
                if v:
                    cmd_lines.append(f'    --{k.replace("_", "-")} \\')
            elif isinstance(v, (int, float, str)):
                cmd_lines.append(f'    --{k.replace("_", "-")} "{v}" \\')
            elif isinstance(v, tuple) or isinstance(v, list):
                cmd_lines.append(f'    --{k.replace("_", "-")} "{tuple(v)}" \\')

        cmd_lines.append(f'    > ./job_logs/${{PBS_JOBID}}_{label}.log 2>&1')

        cmd = "\n".join(cmd_lines)
        cmds.append((cmd, h, f))

    return cmds

def group_cmds(cmds, per_group=per_group):
    cmds.sort()
    groups = [cmds[i:i + per_group] for i in range(0, len(cmds), per_group)]
    return groups

# === Script Generation === #

cmds = generate_cmds(base_params)
cmds_groups = group_cmds(cmds)

for g_id, group in enumerate(cmds_groups):
    script_file = f'{prefix}_Group{g_id}.sh'
    script = [
        "#!/bin/bash",
        "#PBS -P jp09",
        "#PBS -q gpuvolta",
        "#PBS -l walltime=48:00:00",
        "#PBS -l storage=gdata/kl02+scratch/kl02",
        "#PBS -l ncpus=12",
        "#PBS -l ngpus=1",
        "#PBS -l mem=90GB",
        "#PBS -l jobfs=90GB",
        "#PBS -l wd",
        "#PBS -M hsun3103@uni.sydney.edu.au",
        "#PBS -m abe",
        f"#PBS -N {prefix}_Group{g_id}",
        "module load use.own",
        "module load python3/3.9.2"
    ]
    for i, (cmd, _, _) in enumerate(group):
        if i != len(group) - 1:
            script.append(cmd + " &")
        else:
            script.append(cmd)
    script.append("wait")
    script.append('echo "All experiments done."')

    with open(script_file, "w") as f:
        for line in script:
            f.write(line + "\n")
    os.makedirs("job_logs", exist_ok=True)
    os.system(f"qsub {script_file}")
