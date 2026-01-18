#!/usr/bin/env python3
"""
🚀 ADVANCED WORKFLOW RUNNER
===========================
Jalanin semua step advanced workflow dalam satu command!

Usage:
    python run_advanced.py

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import subprocess
import sys
import os
from datetime import datetime

def print_header(text):
    """Print fancy header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"▶️  {description}...")
    print(f"   Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} - SELESAI!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - GAGAL!")
        print(f"   Error: {e}\n")
        return False

def main():
    start_time = datetime.now()
    
    print_header("🚀 ADVANCED WORKFLOW RUNNER")
    print("PKM-RE: H3N2 Antigenic Prediction - Advanced System")
    print("\nWorkflow ini akan menjalankan:")
    print("  1. Advanced data collection (phylogenetic clades, glycosylation)")
    print("  2. Advanced feature extraction (200+ features)")
    print("  3. Run comprehensive tests")
    print("  4. Train advanced models (ensemble, deep learning)")
    print("  5. Analyze features")
    print("  6. Update dashboard")
    print("\n⚠️  WARNING: Ini butuh waktu LAMA (2-4 jam)!")
    print("   Pastikan:")
    print("   • Koneksi internet stabil")
    print("   • Laptop/PC tidak sleep")
    print("   • RAM minimal 8GB")
    print("\nTekan Ctrl+C untuk cancel...")
    
    try:
        response = input("\nLanjut? (yes/no): ").lower()
        if response not in ['yes', 'y']:
            print("\n❌ Dibatalkan oleh user")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n❌ Dibatalkan oleh user")
        sys.exit(0)
    
    # Step 1: Advanced Data Collection
    print_header("STEP 1/6: Advanced Data Collection")
    if not run_command(
        "python scripts/advanced_data_collection.py",
        "Collecting H3N2 data with phylogenetic clades & glycosylation"
    ):
        print("❌ Workflow gagal di step 1")
        sys.exit(1)
    
    # Step 2: Advanced Feature Extraction
    print_header("STEP 2/6: Advanced Feature Extraction")
    print("⚠️  Ini akan extract 200+ features per sequence")
    print("   Kalau mau pakai deep learning embeddings (ESM-2),")
    print("   pastikan sudah install: pip install transformers torch\n")
    
    if not run_command(
        "python scripts/advanced_feature_extraction.py",
        "Extracting 200+ advanced features"
    ):
        print("❌ Workflow gagal di step 2")
        sys.exit(1)
    
    # Step 3: Run Tests
    print_header("STEP 3/6: Run Comprehensive Tests")
    print("⚠️  Kalau pytest belum install: pip install pytest pytest-cov\n")
    
    if not run_command(
        "pytest tests/ -v",
        "Running unit tests and integration tests"
    ):
        print("⚠️  Tests gagal, tapi lanjut...")
    
    # Step 4: Train Advanced Models
    print_header("STEP 4/6: Train Advanced Models")
    print("⚠️  INI STEP PALING LAMA! (1-2 jam)")
    print("   Models yang akan di-train:")
    print("   • Stacking Ensemble (6 base models)")
    print("   • Voting Ensemble")
    print("   • Multi-Layer Perceptron (MLP)")
    print("   • 1D CNN (kalau TensorFlow installed)")
    print("   • CatBoost (kalau installed)")
    print("   • LightGBM (kalau installed)")
    print("   • SHAP analysis")
    print("   • Model calibration\n")
    
    try:
        response = input("Lanjut train models? (yes/no): ").lower()
        if response not in ['yes', 'y']:
            print("\n⚠️  Skipping model training")
        else:
            if not run_command(
                "python scripts/advanced_model_training.py",
                "Training advanced models (ensemble, deep learning)"
            ):
                print("❌ Workflow gagal di step 4")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Skipping model training")
    
    # Step 5: Analyze Features
    print_header("STEP 5/6: Analyze Features")
    if not run_command(
        "python scripts/analyze_features.py",
        "Analyzing feature importance and correlations"
    ):
        print("⚠️  Feature analysis gagal, tapi lanjut...")
    
    # Step 6: Update Dashboard
    print_header("STEP 6/6: Update Dashboard")
    if not run_command(
        "python scripts/update_dashboard.py",
        "Updating interactive dashboard"
    ):
        print("⚠️  Dashboard update gagal, tapi lanjut...")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("✅ ADVANCED WORKFLOW SELESAI!")
    print(f"Total waktu: {duration}")
    print("\n📊 Output Files:")
    print("\n  Data:")
    print("  • data/advanced/h3n2_ha_advanced.csv")
    print("  • data/advanced/h3n2_ha_ultra_high_quality.csv")
    print("  • data/advanced/h3n2_advanced_features.csv")
    
    print("\n  Models:")
    print("  • models/advanced/stacking_binary_model.pkl")
    print("  • models/advanced/voting_soft_binary_model.pkl")
    print("  • models/advanced/mlp_binary_model.pkl")
    print("  • models/advanced/catboost_binary_model.pkl")
    print("  • ... dan banyak lagi")
    
    print("\n  Results:")
    print("  • results/advanced/advanced_training_results.json")
    print("  • results/advanced/model_comparison.csv")
    print("  • results/advanced/model_comparison.png")
    print("  • results/advanced/shap_summary_*.png")
    print("  • results/advanced/calibration_*.png")
    
    print("\n  Dashboard:")
    print("  • dashboard/index.html")
    
    print("\n🎯 Next Steps:")
    print("  1. Buka dashboard: start dashboard/index.html")
    print("  2. Lihat model comparison:")
    print("     python -c \"import pandas as pd; print(pd.read_csv('results/advanced/model_comparison.csv').head())\"")
    print("  3. Batch prediction:")
    print("     python scripts/batch_prediction.py --fasta input.fasta --output results.csv --model-type advanced --ensemble --analyze")
    
    print("\n" + "="*60)
    print("Happy Analyzing! 🎉")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
