#!/bin/bash
#PBS -P jp09
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M didq0417@uni.sydney.edu.au
#PBS -m abe
#PBS -N DiTExp_Group0
module load use.own
module load python3/3.9.2
python3 -u "./shrimp_dit.py" \
    --label "2d6558b339863a9312db0f12f026d2cb" \
    --epochs "200" \
    --batch-size "4" \
    --timesteps "1000" \
    --sampling-timesteps "500" \
    --in-dim "5" \
    --out-dim "1" \
    --dit-model "DiT-S/4" \
    --target-type "pred_x_0" \
    --loss-type "l2" \
    --gamma-type "ddim" \
    --noise-schedule-type "linear" \
    --learning-rate "0.0001" \
    --gf-sigmat "0.0" \
    --gf-sigma1 "0.0" \
    --gf-sigma2 "0.0" \
    --device "cuda" \
    --num-workers "2" \
    --sat-files-path "/g/data/kl02/vhl548/SHRIMP/noradar" \
    --rainfall-files-path "/g/data/kl02/vhl548/SHRIMP/radar/71" \
    --start-date "20210101" \
    --end-date "20210401" \
    --max-folders "180" \
    --history-frames "0" \
    --future-frame "0" \
    --refresh-rate "10" \
    --train-model \
    --load-model "" \
    --input-shape "(5, 128, 128)" \
    --model-path "./h0-f0/models" \
    --results "./h0-f0/results" \
    > ./job_logs/${PBS_JOBID}_2d6558b339863a9312db0f12f026d2cb.log 2>&1
wait
echo "All experiments done."
