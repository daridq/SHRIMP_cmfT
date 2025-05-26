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
#PBS -M hsun3103@uni.sydney.edu.au
#PBS -m abe
#PBS -N DiffExp_Group0
module load use.own
module load python3/3.9.2
python3 -u "./shrimp.py" \
    --label "f88581c7f536c689631f7745d270b0c7" \
    --epochs "200" \
    --batch-size "4" \
    --timesteps "1000" \
    --sampling-timesteps "500" \
    --in-dim "5" \
    --out-dim "1" \
    --embed-dim "64" \
    --dim-scales "(1, 2, 4, 8)" \
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
    --history-frames "1" \
    --future-frame "1" \
    --refresh-rate "10" \
    --train-model \
    --load-model "" \
    --input-shape "(9, 128, 128)" \
    --model-path "./h1-f1/models" \
    --results "./h1-f1/results" \
    > ./job_logs/${PBS_JOBID}_f88581c7f536c689631f7745d270b0c7.log 2>&1
wait
echo "All experiments done."
