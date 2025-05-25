#!/bin/bash
#PBS -P jp09
#PBS -q gpuvolta
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M junyanbai0918@gmail.com
#PBS -m abe
#PBS -N CFMDiT_full_sweep_G0

module purge
module load use.own
module load python3/3.9.2

echo "Flow Matching + DiT Job $PBS_JOBID started at $(date)"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment: full_sweep"

mkdir -p ./cfm_dit_experiments
mkdir -p ./cfm_dit_job_logs

echo "Starting Flow Matching command 1 in group 0 at $(date)"
python3 -u "./shrimp_cfm_dit.py" \
    --batch-size "4" \
    --cfg-scale "1.0" \
    --datasets "./cfm_dit_experiments/full_sweep/h0_f1_heun_DiT-L_4_18082c05c6c5/datasets_used" \
    --device "cuda" \
    --dit-model "DiT-L/4" \
    --end-date "20210110" \
    --epochs "50" \
    --future-frame "1" \
    --gf-sigma1 "0" \
    --gf-sigma2 "0" \
    --gf-sigmat "0" \
    --history-frames "0" \
    --in-dim "5" \
    --label "18082c05c6c5" \
    --learning-rate "0.0001" \
    --load-model "" \
    --loss-type "Hilburn_Loss" \
    --max-folders "20" \
    --model-path "./cfm_dit_experiments/full_sweep/h0_f1_heun_DiT-L_4_18082c05c6c5/models" \
    --out-dim "1" \
    --rainfall-files-path "/g/data/kl02/vhl548/SHRIMP/radar/71" \
    --refresh-rate "10" \
    --results "./cfm_dit_experiments/full_sweep/h0_f1_heun_DiT-L_4_18082c05c6c5/results_data" \
    --rho "7.0" \
    --sampling-timesteps "50" \
    --sat-files-path "/g/data/kl02/vhl548/SHRIMP/noradar" \
    --sigma-max "1.0" \
    --sigma-min "0.0001" \
    --solver-type "heun" \
    --start-date "20210101" \
    --target-type "velocity" \
    --train-model \
    > ./cfm_dit_job_logs/local_full_sweep_18082c05c6c5.log 2>&1
echo "Finished Flow Matching command 1 in group 0 at $(date)"

echo "All Flow Matching experiments in this group done at $(date)."
