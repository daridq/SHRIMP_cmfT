#!/usr/bin/env python
"""
为Flow Matching快速测试创建最小测试数据
"""
import os
import numpy as np
from datetime import datetime, timedelta

def create_minimal_test_data():
    """创建最小的测试数据集"""
    print("🚀 创建快速测试数据...")
    
    # 创建目录
    sat_dir = "quick_test_data/satellite"
    radar_dir = "quick_test_data/radar"
    os.makedirs(sat_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)
    
    # 最小配置
    num_days = 3
    hours_per_day = 2  # 每天只有2个时间点
    image_size = (64, 64)  # 小图像
    sat_channels = 4
    radar_channels = 1
    
    base_date = datetime(2023, 1, 1)
    
    total_files = 0
    
    for day in range(num_days):
        date = base_date + timedelta(days=day)
        date_str = date.strftime("%Y%m%d")
        
        # 创建日期目录
        sat_date_dir = os.path.join(sat_dir, date_str)
        radar_date_dir = os.path.join(radar_dir, date_str)
        os.makedirs(sat_date_dir, exist_ok=True)
        os.makedirs(radar_date_dir, exist_ok=True)
        
        # 每天只创建2个时间点的数据
        for hour in [6, 18]:  # 早上6点和晚上6点
            time_str = f"{hour:02d}00"
            
            # 创建卫星数据 (64x64x4)
            sat_data = np.random.rand(*image_size, sat_channels).astype(np.float32)
            sat_file = os.path.join(sat_date_dir, f"sat_{date_str}_{time_str}.npy")
            np.save(sat_file, sat_data)
            
            # 创建雷达数据 (64x64x1)
            radar_data = np.random.rand(*image_size, radar_channels).astype(np.float32)
            radar_file = os.path.join(radar_date_dir, f"radar_{date_str}_{time_str}.npy")
            np.save(radar_file, radar_data)
            
            total_files += 2
    
    print(f"✅ 创建完成!")
    print(f"   - 总文件数: {total_files}")
    print(f"   - 卫星数据目录: {sat_dir}")
    print(f"   - 雷达数据目录: {radar_dir}")
    print(f"   - 图像尺寸: {image_size}")
    print(f"   - 时间跨度: {num_days} 天")
    
    return sat_dir, radar_dir

def verify_test_data():
    """验证测试数据"""
    print("\n🔍 验证测试数据...")
    
    sat_dir = "quick_test_data/satellite"
    radar_dir = "quick_test_data/radar"
    
    if not os.path.exists(sat_dir) or not os.path.exists(radar_dir):
        print("❌ 测试数据不存在，请先运行创建数据")
        return False
    
    # 统计文件
    sat_files = []
    radar_files = []
    
    for root, dirs, files in os.walk(sat_dir):
        sat_files.extend([f for f in files if f.endswith('.npy')])
    
    for root, dirs, files in os.walk(radar_dir):
        radar_files.extend([f for f in files if f.endswith('.npy')])
    
    print(f"   - 卫星文件数: {len(sat_files)}")
    print(f"   - 雷达文件数: {len(radar_files)}")
    
    # 检查第一个文件的形状
    if sat_files:
        first_sat = None
        for root, dirs, files in os.walk(sat_dir):
            for file in files:
                if file.endswith('.npy'):
                    first_sat = os.path.join(root, file)
                    break
            if first_sat:
                break
        
        if first_sat:
            data = np.load(first_sat)
            print(f"   - 卫星数据形状: {data.shape}")
    
    if radar_files:
        first_radar = None
        for root, dirs, files in os.walk(radar_dir):
            for file in files:
                if file.endswith('.npy'):
                    first_radar = os.path.join(root, file)
                    break
            if first_radar:
                break
        
        if first_radar:
            data = np.load(first_radar)
            print(f"   - 雷达数据形状: {data.shape}")
    
    success = len(sat_files) > 0 and len(radar_files) > 0
    print(f"   - 验证结果: {'✅ 通过' if success else '❌ 失败'}")
    
    return success

def get_quick_test_command():
    """生成快速测试命令"""
    return """python cfm_dit/shrimp_cfm_dit.py \\
    --epochs 2 \\
    --batch-size 1 \\
    --sampling-timesteps 10 \\
    --dit-model "DiT-S/2" \\
    --input-shape "(5,64,64)" \\
    --sigma-min 0.01 \\
    --sigma-max 0.5 \\
    --rho 1.0 \\
    --target-type "velocity" \\
    --solver-type "euler" \\
    --loss-type "l2" \\
    --history-frames 0 \\
    --future-frame 1 \\
    --refresh-rate 50 \\
    --max-folders 2 \\
    --learning-rate 0.001 \\
    --num-workers 0 \\
    --device "cpu" \\
    --label "quick_test" \\
    --model-path "./quick_test_models" \\
    --sat-files-path "./quick_test_data/satellite" \\
    --rainfall-files-path "./quick_test_data/radar" \\
    --train-model"""

if __name__ == "__main__":
    print("=== Flow Matching 快速测试数据生成 ===")
    
    # 创建测试数据
    sat_dir, radar_dir = create_minimal_test_data()
    
    # 验证数据
    if verify_test_data():
        print("\n🎯 快速测试命令:")
        print(get_quick_test_command())
        
        print("\n📝 使用说明:")
        print("1. 激活虚拟环境: source flow/bin/activate")
        print("2. 运行上面的命令")
        print("3. 观察是否有错误，验证模型能否正常运行")
        print("4. 如果成功，再使用完整数据和参数进行正式训练")
    else:
        print("❌ 数据验证失败") 