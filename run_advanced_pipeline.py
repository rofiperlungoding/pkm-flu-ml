"""
Advanced Pipeline Runner
========================
Automated execution of complete advanced ML pipeline for H3N2 antigenic prediction

This script runs the entire advanced pipeline in sequence:
1. Advanced data collection
2. Advanced feature extraction
3. Advanced model training
4. Model evaluation and analysis
5. Dashboard update

Author: PKM-RE Team (Syifa & Rofi)
Date: 2026-01-18
"""
import os
import sys
import subprocess
import time
from datetime import datetime
import json

class PipelineRunner:
    """Automated pipeline execution with logging and error handling"""
    
    def __init__(self):
        self.start_time = None
        self.log_file = f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.results = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'steps': []
        }
    
    def log(self, message):
        """Log message to console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def run_step(self, step_name, script_path, description):
        """Run a pipeline step and log results"""
        self.log(f"\n{'='*60}")
        self.log(f"STEP: {step_name}")
        self.log(f"Description: {description}")
        self.log('='*60)
        
        step_start = time.time()
        
        try:
            # Run script
            self.log(f"Executing: python {script_path}")
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            step_duration = time.time() - step_start
            
            # Log output
            if result.stdout:
                self.log("Output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log(f"  {line}")
            
            # Check for errors
            if result.returncode != 0:
                self.log(f"❌ ERROR: Step failed with return code {result.returncode}")
                if result.stderr:
                    self.log("Error details:")
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            self.log(f"  {line}")
                
                step_result = {
                    'name': step_name,
                    'status': 'failed',
                    'duration': step_duration,
                    'error': result.stderr
                }
                self.results['steps'].append(step_result)
                return False
            
            self.log(f"✅ Step completed successfully in {step_duration:.2f} seconds")
            
            step_result = {
                'name': step_name,
                'status': 'success',
                'duration': step_duration
            }
            self.results['steps'].append(step_result)
            return True
            
        except Exception as e:
            step_duration = time.time() - step_start
            self.log(f"❌ ERROR: Exception occurred: {str(e)}")
            
            step_result = {
                'name': step_name,
                'status': 'failed',
                'duration': step_duration,
                'error': str(e)
            }
            self.results['steps'].append(step_result)
            return False
    
    def run_pipeline(self, skip_data_collection=False, skip_feature_extraction=False):
        """Run complete advanced pipeline"""
        self.start_time = time.time()
        self.results['start_time'] = datetime.now().isoformat()
        
        self.log("="*60)
        self.log("ADVANCED H3N2 ANTIGENIC PREDICTION PIPELINE")
        self.log("PKM-RE Team: Syifa & Rofi")
        self.log("="*60)
        
        # Pipeline steps
        steps = []
        
        if not skip_data_collection:
            steps.append({
                'name': 'Advanced Data Collection',
                'script': 'scripts/advanced_data_collection.py',
                'description': 'Collect H3N2 sequences with phylogenetic clade assignment and quality scoring'
            })
        
        if not skip_feature_extraction:
            steps.append({
                'name': 'Advanced Feature Extraction',
                'script': 'scripts/advanced_feature_extraction.py',
                'description': 'Extract 200+ features including structural, evolutionary, and deep learning embeddings'
            })
        
        steps.extend([
            {
                'name': 'Advanced Model Training',
                'script': 'scripts/advanced_model_training.py',
                'description': 'Train ensemble and deep learning models with SHAP analysis'
            },
            {
                'name': 'Dashboard Update',
                'script': 'scripts/update_dashboard.py',
                'description': 'Update interactive HTML dashboard with latest results'
            }
        ])
        
        # Run each step
        for i, step in enumerate(steps, 1):
            self.log(f"\n\nPipeline Progress: Step {i}/{len(steps)}")
            
            success = self.run_step(
                step['name'],
                step['script'],
                step['description']
            )
            
            if not success:
                self.log(f"\n❌ Pipeline failed at step: {step['name']}")
                self.log("Please check the error messages above and fix the issue.")
                self.finalize_results(success=False)
                return False
        
        # Pipeline completed successfully
        self.finalize_results(success=True)
        return True
    
    def finalize_results(self, success):
        """Finalize and save pipeline results"""
        total_duration = time.time() - self.start_time
        self.results['end_time'] = datetime.now().isoformat()
        self.results['duration'] = total_duration
        self.results['success'] = success
        
        # Save results
        results_file = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self.log("\n" + "="*60)
        self.log("PIPELINE SUMMARY")
        self.log("="*60)
        self.log(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        self.log(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        self.log(f"\nStep Results:")
        
        for step in self.results['steps']:
            status_icon = '✅' if step['status'] == 'success' else '❌'
            self.log(f"  {status_icon} {step['name']}: {step['duration']:.2f}s")
        
        self.log(f"\nLog file: {self.log_file}")
        self.log(f"Results file: {results_file}")
        
        if success:
            self.log("\n🎉 Advanced pipeline completed successfully!")
            self.log("\nNext steps:")
            self.log("  1. View dashboard: open dashboard/index.html")
            self.log("  2. Check model results: results/advanced/")
            self.log("  3. Run batch predictions: python scripts/batch_prediction.py --help")
        else:
            self.log("\n⚠️  Pipeline failed. Please check the error messages above.")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run advanced H3N2 antigenic prediction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_advanced_pipeline.py
  
  # Skip data collection (use existing data)
  python run_advanced_pipeline.py --skip-data-collection
  
  # Skip data collection and feature extraction
  python run_advanced_pipeline.py --skip-data-collection --skip-feature-extraction
        """
    )
    
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='Skip data collection step (use existing data)')
    parser.add_argument('--skip-feature-extraction', action='store_true',
                       help='Skip feature extraction step (use existing features)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    runner = PipelineRunner()
    success = runner.run_pipeline(
        skip_data_collection=args.skip_data_collection,
        skip_feature_extraction=args.skip_feature_extraction
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
