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
#PBS -N CfmU_Group1
module load use.own
module load python3/3.9.2
python3 -u "./shrimp_cfmU.py" \
    --label "2ca96a0f255e36a71f01117ddc74b009" \
    --epochs "200" \
    --batch-size "4" \
    --timesteps "1000" \
    --sampling-timesteps "500" \
    --path-type "optimal_transport" \
    --sigma-min "0.005" \
    --in-dim "5" \
    --out-dim "1" \
    --embed-dim "64" \
    --dim-scales "(1, 2, 4, 8)" \
    --loss-type "l2" \
    --learning-rate "0.001" \
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
    > ./job_logs/${PBS_JOBID}_2ca96a0f255e36a71f01117ddc74b009.log 2>&1 &
python3 -u "./shrimp_cfmU.py" \
    --label "d2b9428eb75f8d199702eac9366a5b40" \
    --epochs "200" \
    --batch-size "4" \
    --timesteps "1000" \
    --sampling-timesteps "500" \
    --path-type "optimal_transport" \
    --sigma-min "0.01" \
    --in-dim "5" \
    --out-dim "1" \
    --embed-dim "64" \
    --dim-scales "(1, 2, 4, 8)" \
    --loss-type "l2" \
    --learning-rate "0.001" \
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
    > ./job_logs/${PBS_JOBID}_d2b9428eb75f8d199702eac9366a5b40.log 2>&1
wait
echo "All experiments done."
