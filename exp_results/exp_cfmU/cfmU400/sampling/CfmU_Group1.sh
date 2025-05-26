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
    --label "af1ec5fcc4366c503a82890cc7aa287f" \
    --epochs "400" \
    --batch-size "8" \
    --timesteps "500" \
    --sampling-timesteps "250" \
    --path-type "optimal_transport" \
    --sigma-min "0.001" \
    --in-dim "5" \
    --out-dim "1" \
    --embed-dim "64" \
    --dim-scales "(1, 2, 4, 8)" \
    --loss-type "l2" \
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
    > ./job_logs/${PBS_JOBID}_af1ec5fcc4366c503a82890cc7aa287f.log 2>&1 &
python3 -u "./shrimp_cfmU.py" \
    --label "c27e465b18985d4e52c7a74a6b7d32d0" \
    --epochs "400" \
    --batch-size "8" \
    --timesteps "250" \
    --sampling-timesteps "100" \
    --path-type "optimal_transport" \
    --sigma-min "0.001" \
    --in-dim "5" \
    --out-dim "1" \
    --embed-dim "64" \
    --dim-scales "(1, 2, 4, 8)" \
    --loss-type "l2" \
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
    > ./job_logs/${PBS_JOBID}_c27e465b18985d4e52c7a74a6b7d32d0.log 2>&1
wait
echo "All experiments done."
