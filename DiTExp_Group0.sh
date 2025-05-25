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
#PBS -M junyanbai0918@gmail.com
#PBS -m abe
#PBS -N DiTExp_Group0
module load use.own
module load python3/3.9.2
python3 -u "./shrimp_dit.py" \
    --label "423b25d1fc5f3437939f9ad25cc30ee6" \
    --epochs "3" \
    --batch-size "4" \
    --timesteps "1000" \
    --sampling-timesteps "500" \
    --in-dim "5" \
    --out-dim "1" \
    --dit-model "DiT-L/4" \
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
    --end-date "20210630" \
    --max-folders "180" \
    --history-frames "0" \
    --future-frame "0" \
    --refresh-rate "10" \
    --train-model \
    --load-model "" \
    --input-shape "(5, 128, 128)" \
    --model-path "./h0-f0/models" \
    --results "./h0-f0/results" \
    > ./job_logs/${PBS_JOBID}_423b25d1fc5f3437939f9ad25cc30ee6.log 2>&1
wait
echo "All experiments done."
