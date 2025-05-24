#!/bin/bash

# SHRIMP Flow Matching 环境设置脚本

echo "🚀 Setting up SHRIMP Flow Matching environment..."

# 1. 创建虚拟环境
echo "📦 Creating virtual environment..."
python3 -m venv shrimp_env
source shrimp_env/bin/activate

# 2. 升级pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# 3. 安装依赖
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# 4. 创建必要的目录
echo "📁 Creating output directories..."
mkdir -p output_data/models
mkdir -p output_data/results
mkdir -p output_data/datasets
mkdir -p demo_data/satellite
mkdir -p demo_data/radar

# 5. 检查TensorFlow和GPU
echo "🔍 Checking TensorFlow installation..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
print('✅ TensorFlow setup complete!')
"

echo "✅ Environment setup complete!"
echo "💡 To activate the environment, run: source shrimp_env/bin/activate" 