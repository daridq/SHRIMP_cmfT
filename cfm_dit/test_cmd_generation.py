#!/usr/bin/env python3
"""
Test script for CFM DiT command generation.
Validates the generated commands and demonstrates usage.
"""

import os
import sys
import tempfile
import shutil
from cfm_dit_cmd_generate import (
    generate_cmds, quick_params, solver_comparison_params,
    generate_label, generate_pbs_script
)

def test_label_generation():
    """Test unique label generation."""
    print("🔍 Testing label generation...")
    
    params1 = {
        "epochs": 10,
        "batch_size": 2,
        "solver_type": "heun",
        "dit_model": "DiT-S/2"
    }
    
    params2 = {
        "epochs": 20,
        "batch_size": 2,
        "solver_type": "heun", 
        "dit_model": "DiT-S/2"
    }
    
    label1 = generate_label(params1)
    label2 = generate_label(params2)
    
    print(f"  Label 1: {label1}")
    print(f"  Label 2: {label2}")
    
    assert label1 != label2, "Labels should be different for different parameters"
    assert len(label1) == 12, "Label should be 12 characters"
    print("✅ Label generation test passed")

def test_command_generation():
    """Test command generation."""
    print("\n🔍 Testing command generation...")
    
    # Test with minimal parameters
    test_params = {
        "epochs": [5],
        "batch_size": [2],
        "in_dim": [5],
        "out_dim": [1],
        "input_shape": ["(5,64,64)"],
        "dit_model": ["DiT-S/2"],
        "sigma_min": [1e-4],
        "sigma_max": [1.0],
        "rho": [7.0],
        "target_type": ["velocity"],
        "solver_type": ["heun"],
        "sampling_timesteps": [10],
        "loss_type": ["l2"],
        "learning_rate": [0.001],
        "gf_sigmat": [0],
        "cfg_scale": [1.0],
        "device": ["cpu"],
        "sat_files_path": ["/tmp/sat"],
        "rainfall_files_path": ["/tmp/radar"],
        "start_date": ["20210101"],
        "end_date": ["20210102"],
        "max_folders": [1],
        "history_frames": [0],
        "future_frame": [0],
        "refresh_rate": [10],
        "train_model": [True],
        "retrieve_dataset": [False],
        "load_model": [""],
    }
    
    cmds = generate_cmds(test_params, "test")
    
    assert len(cmds) == 1, "Should generate exactly one command"
    
    cmd_str, h, f, solver, model, label = cmds[0]
    
    # Check command structure
    assert "python3 -u \"./shrimp_cfm_dit.py\"" in cmd_str
    assert "--epochs \"5\"" in cmd_str
    assert "--solver-type \"heun\"" in cmd_str
    assert "--dit-model \"DiT-S/2\"" in cmd_str
    assert "--train-model" in cmd_str
    
    print(f"  Generated command preview:")
    print(f"  {cmd_str[:100]}...")
    print(f"  Metadata: h={h}, f={f}, solver={solver}, model={model}")
    print("✅ Command generation test passed")

def test_pbs_script_generation():
    """Test PBS script generation."""
    print("\n🔍 Testing PBS script generation...")
    
    test_cmd = 'python3 -u "./shrimp_cfm_dit.py" --epochs "10" > ./logs/test.log 2>&1'
    
    script_name, script_content = generate_pbs_script(
        [test_cmd], 0, "test", walltime="1:00:00", mem="16GB"
    )
    
    # Check PBS script structure
    script_str = "\n".join(script_content)
    
    assert "#PBS -P jp09" in script_str
    assert "#PBS -q gpuvolta" in script_str
    assert "#PBS -l walltime=1:00:00" in script_str
    assert "#PBS -l mem=16GB" in script_str
    assert "module load pytorch" in script_str
    assert test_cmd in script_str
    
    print(f"  Script name: {script_name}")
    print(f"  Script length: {len(script_content)} lines")
    print("✅ PBS script generation test passed")

def test_experiment_configs():
    """Test different experiment configurations."""
    print("\n🔍 Testing experiment configurations...")
    
    configs = {
        "quick": quick_params,
        "solver": solver_comparison_params
    }
    
    for name, params in configs.items():
        cmds = generate_cmds(params, name)
        print(f"  {name.capitalize()} config: {len(cmds)} commands generated")
        
        if cmds:
            cmd_str = cmds[0][0]
            # Check Flow Matching specific parameters are present
            assert "--sigma-min" in cmd_str
            assert "--solver-type" in cmd_str
            assert "--target-type" in cmd_str
    
    print("✅ Experiment configuration test passed")

def test_parameter_combinations():
    """Test parameter combination generation."""
    print("\n🔍 Testing parameter combinations...")
    
    multi_params = {
        "epochs": [5, 10],
        "solver_type": ["euler", "heun"],
        "dit_model": ["DiT-S/2"],
        "batch_size": [2],
        "in_dim": [5],
        "out_dim": [1],
        "input_shape": ["(5,64,64)"],
        "sigma_min": [1e-4],
        "sigma_max": [1.0],
        "rho": [7.0],
        "target_type": ["velocity"],
        "sampling_timesteps": [10],
        "loss_type": ["l2"],
        "learning_rate": [0.001],
        "gf_sigmat": [0],
        "cfg_scale": [1.0],
        "device": ["cpu"],
        "sat_files_path": ["/tmp"],
        "rainfall_files_path": ["/tmp"],
        "start_date": ["20210101"],
        "end_date": ["20210102"],
        "max_folders": [1],
        "history_frames": [0],
        "future_frame": [0],
        "refresh_rate": [10],
        "train_model": [True],
        "retrieve_dataset": [False],
        "load_model": [""],
    }
    
    cmds = generate_cmds(multi_params, "multi_test")
    
    # Should generate 2 epochs × 2 solvers = 4 combinations
    expected_combinations = 2 * 2
    assert len(cmds) == expected_combinations, f"Expected {expected_combinations} combinations, got {len(cmds)}"
    
    # Check that different combinations have different labels
    labels = [cmd[5] for cmd in cmds]
    assert len(set(labels)) == len(labels), "All combinations should have unique labels"
    
    print(f"  Generated {len(cmds)} parameter combinations")
    print(f"  Unique labels: {len(set(labels))}")
    print("✅ Parameter combination test passed")

def demo_generated_command():
    """Demonstrate a generated command."""
    print("\n📝 Demo: Generated Command Example")
    print("=" * 50)
    
    cmds = generate_cmds(quick_params, "demo")
    if cmds:
        cmd_str, h, f, solver, model, label = cmds[0]
        
        print("Generated Flow Matching + DiT command:")
        print("-" * 40)
        
        # Format command for better readability
        lines = cmd_str.split('\n')
        for line in lines:
            print(line)
        
        print("-" * 40)
        print(f"Experiment metadata:")
        print(f"  • History frames: {h}")
        print(f"  • Future frames: {f}")
        print(f"  • ODE solver: {solver}")
        print(f"  • DiT model: {model}")
        print(f"  • Unique label: {label}")

def main():
    """Run all tests."""
    print("🧪 CFM DiT Command Generation Test Suite")
    print("=" * 50)
    
    try:
        test_label_generation()
        test_command_generation()
        test_pbs_script_generation()
        test_experiment_configs()
        test_parameter_combinations()
        demo_generated_command()
        
        print("\n🎉 All tests passed!")
        print("\n📋 Summary:")
        print("  ✅ Label generation works correctly")
        print("  ✅ Commands include all Flow Matching parameters")
        print("  ✅ PBS scripts have proper structure")
        print("  ✅ Multiple experiment configs supported")
        print("  ✅ Parameter combinations generate unique labels")
        
        print("\n🚀 Ready to use:")
        print("  python cfm_dit_cmd_generate.py quick")
        print("  python cfm_dit_cmd_generate.py solver")
        print("  python cfm_dit_cmd_generate.py standard")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 