#!/usr/bin/env python3
"""
SHRIMP CFM 本地演示运行脚本
自动化数据生成、模型训练和推理的完整流程
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并显示输出"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    else:
        print(f"✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True

def main():
    print("🚀 SHRIMP Flow Matching 本地演示")
    print("=" * 50)
    
    # 1. 生成演示数据
    print("\n📊 Step 1: Generating demo data...")
    if not run_command(
        "python generate_demo_data.py --num-days 3 --files-per-day 12", 
        "Generating synthetic satellite and radar data"
    ):
        return
    
    # 2. CFM模式训练（短时间演示）
    print("\n🧠 Step 2: Training CFM model (demo)...")
    cfm_cmd = """
    python cfm_unet/shrimp_cfm.py \
        --cfm-mode \
        --train-model \
        --epochs 5 \
        --batch-size 2 \
        --learning-rate 0.001 \
        --sat-files-path "./demo_data/satellite" \
        --rainfall-files-path "./demo_data/radar" \
        --start-date "20240101" \
        --end-date "20240103" \
        --max-folders 3 \
        --history-frames 0 \
        --future-frame 1 \
        --label "demo_cfm" \
        --device "CPU"
    """.replace('\n', ' ').strip()
    
    if not run_command(cfm_cmd, "Training CFM model"):
        print("⚠️  Training failed, but continuing with baseline...")
    
    # 3. 基线模式训练（对比）
    print("\n📈 Step 3: Training baseline model (demo)...")
    baseline_cmd = """
    python cfm_unet/shrimp_cfm.py \
        --train-model \
        --epochs 5 \
        --batch-size 2 \
        --learning-rate 0.001 \
        --sat-files-path "./demo_data/satellite" \
        --rainfall-files-path "./demo_data/radar" \
        --start-date "20240101" \
        --end-date "20240103" \
        --max-folders 3 \
        --history-frames 0 \
        --future-frame 1 \
        --label "demo_baseline" \
        --device "CPU"
    """.replace('\n', ' ').strip()
    
    if not run_command(baseline_cmd, "Training baseline model"):
        print("⚠️  Baseline training failed")
    
    # 4. 显示结果
    print("\n📁 Step 4: Checking outputs...")
    
    output_dirs = [
        "./output_data/models",
        "./output_data/results", 
        "./output_data/datasets"
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            files = list(Path(dir_path).glob("*"))
            print(f"\n📂 {dir_path}:")
            for file in files[:5]:  # 显示前5个文件
                print(f"  - {file.name}")
            if len(files) > 5:
                print(f"  ... and {len(files)-5} more files")
        else:
            print(f"\n📂 {dir_path}: (not created)")
    
    print("\n🎉 Demo complete!")
    print("\n💡 Next steps:")
    print("1. Check output_data/ for trained models and results")
    print("2. Modify parameters in the commands above for your use case")
    print("3. Use real satellite/radar data by changing file paths")

if __name__ == "__main__":
    main() 