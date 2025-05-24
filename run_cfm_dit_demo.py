#!/usr/bin/env python3
"""
Demo script to run Flow Matching with DiT backbone training.
This script demonstrates how to use the shrimp_cfm_dit.py for training and testing.
"""

import os
import subprocess
import sys
from datetime import datetime

def run_cfm_dit_demo():
    """Run Flow Matching + DiT demo with synthetic data."""
    
    print("🌊 SHRIMP Flow Matching + DiT Demo 🌊")
    print("=" * 50)
    
    # Create demo directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    demo_dir = f"demo_cfm_dit_{timestamp}"
    model_path = os.path.join(demo_dir, "models")
    results_path = os.path.join(demo_dir, "results")
    data_path = os.path.join(demo_dir, "data")
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    
    print(f"📁 Demo directory: {demo_dir}")
    
    # First, generate synthetic data
    print("\n🔧 Generating synthetic weather data...")
    try:
        result = subprocess.run([
            sys.executable, "generate_demo_data.py",
            "--output-dir", data_path,
            "--num-samples", "100",
            "--image-size", "64"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print(f"❌ Error generating data: {result.stderr}")
            return
        print("✅ Synthetic data generated successfully!")
    except FileNotFoundError:
        print("⚠️  generate_demo_data.py not found, assuming data exists...")
    
    # Training parameters for Flow Matching
    base_args = [
        sys.executable, "cfm_dit/shrimp_cfm_dit.py",
        
        # Model and training parameters
        "--epochs", "10",  # Short demo training
        "--batch-size", "2",
        "--sampling-timesteps", "20",  # Fast sampling for demo
        "--in-dim", "5",
        "--out-dim", "1", 
        "--input-shape", "(5,64,64)",
        "--dit-model", "DiT-S/2",  # Smaller model for demo
        
        # Flow Matching specific parameters
        "--sigma-min", "0.0001",
        "--sigma-max", "1.0",
        "--rho", "7.0",
        "--target-type", "velocity",
        "--solver-type", "heun",
        
        # Training settings
        "--loss-type", "l2",
        "--learning-rate", "0.0002",
        "--gf-sigmat", "0.5",
        
        # Demo data paths (adjust these if you have real data)
        "--sat-files-path", data_path,
        "--rainfall-files-path", data_path,
        "--start-date", "2020-01-01",
        "--end-date", "2020-01-10",
        "--max-folders", "10",
        
        # Temporal settings
        "--history-frames", "0",
        "--future-frame", "1",
        "--refresh-rate", "1",
        
        # Output paths
        "--model-path", model_path,
        "--results", results_path,
        "--label", f"cfm_demo_{timestamp}",
        
        # Execution control
        "--device", "cpu",  # Use CPU for demo (change to cuda if available)
        "--num-workers", "1",
        "--train-model",  # Enable training
    ]
    
    print("\n🚀 Starting Flow Matching training...")
    print("Training Parameters:")
    print(f"  • Model: DiT-S/2 with Flow Matching")
    print(f"  • Epochs: 10 (demo)")
    print(f"  • Batch Size: 2")
    print(f"  • Solver: Heun (2nd order ODE)")
    print(f"  • Target: Velocity field")
    print(f"  • Sampling Steps: 20")
    
    try:
        # Run training
        result = subprocess.run(base_args, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Flow Matching training completed successfully!")
            
            # Show generated files
            print(f"\n📊 Generated Files:")
            for root, dirs, files in os.walk(demo_dir):
                for file in files:
                    if file.endswith(('.pt', '.npy', '.pdf', '.json', '.txt')):
                        rel_path = os.path.relpath(os.path.join(root, file), demo_dir)
                        print(f"  • {rel_path}")
                        
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
    
    # Optional: Run comparison with different solvers
    print(f"\n🔬 Running solver comparison...")
    
    solvers = ["euler", "heun"]
    solver_results = {}
    
    for solver in solvers:
        print(f"  Testing {solver} solver...")
        solver_args = base_args.copy()
        
        # Update solver and label
        solver_idx = solver_args.index("--solver-type") + 1
        solver_args[solver_idx] = solver
        
        label_idx = solver_args.index("--label") + 1
        solver_args[label_idx] = f"cfm_{solver}_{timestamp}"
        
        # Remove training flag for testing
        if "--train-model" in solver_args:
            solver_args.remove("--train-model")
        
        # Add model loading (use best model from previous training)
        best_model = f"model_best_cfm_demo_{timestamp}.pt"
        solver_args.extend(["--load-model", best_model])
        
        try:
            result = subprocess.run(solver_args, capture_output=True, text=True, cwd=".")
            solver_results[solver] = "✅ Success" if result.returncode == 0 else "❌ Failed"
        except:
            solver_results[solver] = "❌ Error"
    
    print(f"\n📈 Solver Comparison Results:")
    for solver, status in solver_results.items():
        print(f"  • {solver.capitalize()}: {status}")
    
    print(f"\n🎯 Demo Summary:")
    print(f"  • Demo directory: {demo_dir}")
    print(f"  • Model files: {model_path}")
    print(f"  • Results: {results_path}")
    print(f"  • View logs: {model_path}/logs_cfm_demo_{timestamp}.txt")
    
    print(f"\n📚 Next Steps:")
    print(f"  1. Check training curves: {model_path}/FlowMatching_Loss_Curve_*.pdf")
    print(f"  2. Analyze sampling results: {results_path}/*/cfm_outputs.npy")
    print(f"  3. Compare with diffusion results using shrimp_dit_o.py")
    print(f"  4. Experiment with different solvers and sampling steps")

def show_help():
    """Show usage help."""
    print("Flow Matching + DiT Demo Script")
    print("=" * 30)
    print()
    print("Usage:")
    print("  python run_cfm_dit_demo.py")
    print()
    print("This script will:")
    print("  1. Generate synthetic weather data")
    print("  2. Train a Flow Matching model with DiT backbone")
    print("  3. Test different ODE solvers")
    print("  4. Save results and visualizations")
    print()
    print("Requirements:")
    print("  • PyTorch")
    print("  • NumPy") 
    print("  • Matplotlib")
    print("  • tqdm")
    print()
    print("For real data, modify the data paths in the script.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        show_help()
    else:
        run_cfm_dit_demo() 