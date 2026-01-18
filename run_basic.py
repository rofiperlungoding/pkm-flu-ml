#!/usr/bin/env python3
"""
🚀 BASIC WORKFLOW RUNNER
========================
Jalanin semua step basic workflow dalam satu command!

Usage:
    python run_basic.py

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
    
    print_header("🚀 BASIC WORKFLOW RUNNER")
    print("PKM-RE: H3N2 Antigenic Prediction")
    print("Workflow ini akan menjalankan:")
    print("  1. Download data")
    print("  2. Extract features")
    print("  3. Train model")
    print("  4. Evaluate model")
    print("  5. Update dashboard")
    print("\nEstimasi waktu: 30-60 menit")
    print("\nTekan Ctrl+C untuk cancel...")
    
    try:
        input("\nTekan ENTER untuk mulai... ")
    except KeyboardInterrupt:
        print("\n\n❌ Dibatalkan oleh user")
        sys.exit(0)
    
    # Step 1: Download Data
    print_header("STEP 1/5: Download Data")
    if not run_command(
        "python scripts/download_comprehensive_h3n2.py",
        "Downloading H3N2 sequences from NCBI"
    ):
        print("❌ Workflow gagal di step 1")
        sys.exit(1)
    
    # Step 2: Extract Features
    print_header("STEP 2/5: Extract Features")
    if not run_command(
        "python scripts/extract_features.py",
        "Extracting 74 physicochemical features"
    ):
        print("❌ Workflow gagal di step 2")
        sys.exit(1)
    
    # Step 3: Train Model
    print_header("STEP 3/5: Train Model")
    if not run_command(
        "python scripts/train_model.py",
        "Training XGBoost models (binary & multiclass)"
    ):
        print("❌ Workflow gagal di step 3")
        sys.exit(1)
    
    # Step 4: Evaluate Model
    print_header("STEP 4/5: Evaluate Model")
    if not run_command(
        "python scripts/evaluate_model.py",
        "Evaluating model performance"
    ):
        print("⚠️  Evaluation gagal, tapi lanjut...")
    
    # Step 5: Update Dashboard
    print_header("STEP 5/5: Update Dashboard")
    if not run_command(
        "python scripts/update_dashboard.py",
        "Updating interactive dashboard"
    ):
        print("⚠️  Dashboard update gagal, tapi lanjut...")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("✅ WORKFLOW SELESAI!")
    print(f"Total waktu: {duration}")
    print("\n📊 Output Files:")
    print("  • Data: data/processed/h3n2_ha_comprehensive.csv")
    print("  • Features: data/processed/h3n2_features.csv")
    print("  • Models: models/h3n2_binary_model.pkl")
    print("  •         models/h3n2_multiclass_model.pkl")
    print("  • Results: results/training_results.json")
    print("  • Dashboard: dashboard/index.html")
    
    print("\n🎯 Next Steps:")
    print("  1. Buka dashboard: start dashboard/index.html")
    print("  2. Cek results: cat results/training_results.json")
    print("  3. Prediksi baru: python scripts/predict_sequence.py --help")
    
    print("\n" + "="*60)
    print("Happy Analyzing! 🎉")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
