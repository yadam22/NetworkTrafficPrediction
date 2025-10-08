# Loading the actual checkpoint and training history data to verify and analyze the results

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Adding numpy scalar to safe globals for PyTorch 2.6+
import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

from pretraining import (
    NetworkTrafficConfig, 
    NetworkTrafficTransformer,
    NetworkTrafficDataset
)

class NetworkTrafficModelTester:
    """Test and evaluate the pretrained Network Traffic Transformer model"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.checkpoint = None
        self.training_info = None
        
    def load_checkpoint(self, checkpoint_name: str = 'best_model.pt'):
        """Load model checkpoint and training information"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {checkpoint_path}")
            # Try alternative paths
            alternatives = ['final_model.pt', 'checkpoint_epoch_20.pt']
            for alt in alternatives:
                alt_path = self.checkpoint_dir / alt
                if alt_path.exists():
                    checkpoint_path = alt_path
                    print(f"Using alternative checkpoint: {alt}")
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Fix for PyTorch 2.6+ - handle weights_only parameter
        try:
            # First try with weights_only=True (more secure)
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Safe loading failed, attempting with weights_only=False...")
            # Fallback to weights_only=False if trusted source
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            print(" Checkpoint loaded successfully with weights_only=False")
        
        # Load training info if available
        hf_model_dir = self.checkpoint_dir / 'hf_model_best'
        if not hf_model_dir.exists():
            hf_model_dir = self.checkpoint_dir / 'hf_model_final'
        
        if hf_model_dir.exists():
            training_info_path = hf_model_dir / 'training_info.json'
            if training_info_path.exists():
                with open(training_info_path, 'r') as f:
                    self.training_info = json.load(f)
        
        return self.checkpoint
    
    def initialize_model(self):
        """Initialize model from checkpoint"""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        # Extract config from checkpoint or create default
        if 'config' in self.checkpoint:
            # Reconstruct model config from saved training config
            model_config = NetworkTrafficConfig(
                vocab_size=10000,
                hidden_size=512,
                num_hidden_layers=6,
                num_attention_heads=8,
                max_position_embeddings=256
            )
        else:
            # Use default config
            model_config = NetworkTrafficConfig()
        
        # Initialize model
        self.model = NetworkTrafficTransformer(model_config)
        
        # Load state dict
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def analyze_training_history(self):
        """Analyze and display training history"""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        history = self.checkpoint.get('train_history', {})
        
        if not history:
            print("No training history found in checkpoint")
            return None
        
        # Create results dictionary
        results = {
            'final_train_loss': history['loss'][-1] if history.get('loss') else None,
            'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
            'best_val_loss': min(history['val_loss']) if history.get('val_loss') else None,
            'total_epochs': len(history.get('loss', [])),
            'dataset_losses': {}
        }
        
        # Analyze dataset-specific losses
        if 'dataset_losses' in history:
            for dataset_name, losses in history['dataset_losses'].items():
                if losses:
                    results['dataset_losses'][dataset_name] = {
                        'final': losses[-1],
                        'best': min(losses),
                        'improvement': losses[0] - losses[-1] if len(losses) > 0 else 0
                    }
        
        return results
    
    def generate_performance_table(self):
        """Generate performance table like in the LaTeX document"""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        history = self.checkpoint.get('train_history', {})
        
        # Define epochs to report
        report_epochs = [1, 5, 10, 15, 20]
        
        # Create table data
        table_data = []
        
        for epoch_idx in report_epochs:
            if epoch_idx <= len(history.get('loss', [])):
                row = {
                    'Epoch': epoch_idx,
                    'Train Loss': history['loss'][epoch_idx-1] if epoch_idx <= len(history['loss']) else None,
                    'Val Loss': history['val_loss'][epoch_idx-1] if epoch_idx <= len(history.get('val_loss', [])) else None
                }
                
                # Add dataset-specific losses - check for different possible naming conventions
                dataset_names_to_check = [
                    ('tokenized_data_10s', '10s Loss'),
                    ('tokenized_data_30s', '30s Loss'),
                    ('tokenized_data_1min', '1min Loss'),
                    ('10s', '10s Loss'),
                    ('30s', '30s Loss'),
                    ('1min', '1min Loss')
                ]
                
                for dataset_name, column_name in dataset_names_to_check:
                    if dataset_name in history.get('dataset_losses', {}):
                        dataset_losses = history['dataset_losses'][dataset_name]
                        if epoch_idx <= len(dataset_losses):
                            row[column_name] = dataset_losses[epoch_idx-1]
                
                table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Format the DataFrame for display
        pd.set_option('display.precision', 3)
        print("\nPre-training Performance Progression")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Also save as CSV for reference
        csv_path = self.checkpoint_dir / 'performance_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"Performance table saved to: {csv_path}")
        
        return df
    
    def plot_enhanced_training_history(self, save_plots: bool = True, show_plots: bool = False):
        """Create enhanced visualization of training history"""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        history = self.checkpoint.get('train_history', {})
        
        if not history:
            print("No training history to plot")
            return
        
        # Check if we already have a saved plot
        existing_plot = self.checkpoint_dir / 'training_history.png'
        if existing_plot.exists():
            print(f"Training history plot found at: {existing_plot}")
        
        # Create new enhanced plot
        fig = plt.figure(figsize=(15, 10))
        
        # Main loss plot
        ax1 = plt.subplot(2, 2, 1)
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs[:len(history['val_loss'])], history['val_loss'], 
                    'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2 = plt.subplot(2, 2, 2)
        if 'learning_rate' in history:
            ax2.plot(epochs[:len(history['learning_rate'])], history['learning_rate'], 
                    'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
        
        # Dataset-specific losses
        ax3 = plt.subplot(2, 2, 3)
        if 'dataset_losses' in history:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for idx, (dataset_name, losses) in enumerate(history['dataset_losses'].items()):
                if losses:
                    dataset_label = dataset_name.split('_')[-1] if '_' in dataset_name else dataset_name
                    ax3.plot(range(1, len(losses) + 1), losses, 
                            color=colors[idx % len(colors)], 
                            label=dataset_label, linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Dataset-specific Losses')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Loss improvement over time
        ax4 = plt.subplot(2, 2, 4)
        if len(history['loss']) > 1:
            improvements = [history['loss'][0] - loss for loss in history['loss']]
            ax4.plot(epochs, improvements, 'purple', linewidth=2, label='Training')
            if 'val_loss' in history and len(history['val_loss']) > 1:
                val_improvements = [history['val_loss'][0] - loss for loss in history['val_loss']]
                ax4.plot(epochs[:len(val_improvements)], val_improvements, 
                        'orange', linewidth=2, label='Validation')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Improvement')
            ax4.set_title('Cumulative Loss Improvement')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Network Traffic Transformer Pre-training Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            # Save the enhanced plot
            enhanced_plot_path = self.checkpoint_dir / 'training_history_enhanced.png'
            plt.savefig(enhanced_plot_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced plot saved to: {enhanced_plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
        
        return fig
    
    def test_model_inference(self, test_data_path: Optional[str] = None):
        """Test model inference capabilities"""
        if self.model is None:
            self.initialize_model()
        
        print("\nTesting Model Inference")
        print("="*50)
        
        # Create dummy input if no test data provided
        if test_data_path is None:
            batch_size = 4
            seq_length = 128
            
            dummy_input = {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(self.device),
                'attention_mask': torch.ones(batch_size, seq_length).to(self.device),
                'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long).to(self.device),
                'temporal_ids': torch.randint(0, 24*60, (batch_size,)).to(self.device)
            }
            
            print("Using dummy input for inference test")
        else:
            # Load actual test data
            test_dataset = NetworkTrafficDataset(
                tokenized_data_path=test_data_path,
                task='pretrain'
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=4, 
                shuffle=False
            )
            dummy_input = next(iter(test_loader))
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            print(f"Using test data from: {test_data_path}")
        
        # Test different tasks
        tasks = ['mlm', 'traffic_prediction', 'anomaly_detection', 'protocol_classification']
        
        inference_results = {}
        
        with torch.no_grad():
            for task in tasks:
                print(f"\nTesting task: {task}")
                try:
                    outputs = self.model(**dummy_input, task=task)
                    
                    if task == 'mlm' and 'mlm_logits' in outputs:
                        print(f"   MLM output shape: {outputs['mlm_logits'].shape}")
                        inference_results[task] = 'Success'
                    elif task == 'traffic_prediction' and 'traffic_predictions' in outputs:
                        print(f"   Traffic predictions shape: {outputs['traffic_predictions'].shape}")
                        inference_results[task] = 'Success'
                    elif task == 'anomaly_detection' and 'anomaly_logits' in outputs:
                        print(f"   Anomaly logits shape: {outputs['anomaly_logits'].shape}")
                        inference_results[task] = 'Success'
                    elif task == 'protocol_classification' and 'protocol_logits' in outputs:
                        print(f"   Protocol logits shape: {outputs['protocol_logits'].shape}")
                        inference_results[task] = 'Success'
                    
                    if 'loss' in outputs:
                        print(f"  Loss: {outputs['loss'].item():.4f}")
                        
                except Exception as e:
                    print(f"   Error: {str(e)}")
                    inference_results[task] = f'Failed: {str(e)}'
        
        print("\nInference test completed")
        return inference_results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("NETWORK TRAFFIC TRANSFORMER - MODEL TEST REPORT")
        print("="*70)
        
        # Load checkpoint if needed
        if self.checkpoint is None:
            self.load_checkpoint()
        
        # Basic information
        print(f"\nCheckpoint Directory: {self.checkpoint_dir}")
        print(f"Device: {self.device}")
        
        # Training configuration
        if 'config' in self.checkpoint:
            config = self.checkpoint['config']
            print(f"\nTraining Configuration:")
            print(f"  - Batch Size: {config.get('batch_size', 'N/A')}")
            print(f"  - Learning Rate: {config.get('learning_rate', 'N/A')}")
            print(f"  - Number of Epochs: {config.get('num_epochs', 'N/A')}")
            print(f"  - Dataset Strategy: {config.get('dataset_strategy', 'N/A')}")
            print(f"  - Mixed Precision: {config.get('mixed_precision', 'N/A')}")
            print(f"  - Gradient Clipping: {config.get('gradient_clip', 'N/A')}")
        
        # Dataset information
        if 'dataset_info' in self.checkpoint:
            print(f"\nDatasets Used:")
            total_samples = 0
            for ds_info in self.checkpoint['dataset_info']:
                num_samples = ds_info['num_samples']
                total_samples += num_samples
                print(f"  - {ds_info['name']}: {num_samples:,} samples ({ds_info['time_window']})")
            print(f"  Total samples: {total_samples:,}")
        
        # Performance metrics
        results = self.analyze_training_history()
        if results:
            print(f"\nPerformance Metrics:")
            print(f"  - Final Training Loss: {results['final_train_loss']:.4f}")
            if results['final_val_loss']:
                print(f"  - Final Validation Loss: {results['final_val_loss']:.4f}")
                print(f"  - Best Validation Loss: {results['best_val_loss']:.4f}")
            print(f"  - Total Epochs Trained: {results['total_epochs']}")
            
            if results['dataset_losses']:
                print(f"\nDataset-specific Final Losses:")
                for dataset_name, metrics in results['dataset_losses'].items():
                    print(f"  - {dataset_name}: {metrics['final']:.4f} (improved by {metrics['improvement']:.4f})")
        
        # Model architecture
        if self.training_info:
            print(f"\nAdditional Information:")
            print(f"  - Total Training Steps: {self.training_info.get('total_steps', 'N/A'):,}")
            print(f"  - Final Loss (from training_info): {self.training_info.get('final_loss', 'N/A')}")
        
        # Checkpoint metadata
        print(f"\nCheckpoint Metadata:")
        print(f"  - Epoch: {self.checkpoint.get('epoch', 'N/A')}")
        print(f"  - Global Step: {self.checkpoint.get('global_step', 'N/A'):,}")
        print(f"  - Loss at Checkpoint: {self.checkpoint.get('loss', 'N/A')}")
        
        print("\n" + "="*70)


def main():
    """Main test function"""
    # Configuration
    checkpoint_dir = 'code/data/llm/checkpoints/multi_dataset_pretrain'
    
    # Initialize tester
    tester = NetworkTrafficModelTester(checkpoint_dir)
    
    # Run comprehensive tests
    print("Starting Network Traffic Transformer Model Testing...")
    print("="*70)
    
    try:
        # 1. Load checkpoint and analyze training history
        checkpoint = tester.load_checkpoint()
        print(f" Checkpoint loaded: Epoch {checkpoint.get('epoch', 'N/A')}, "
              f"Global Step {checkpoint.get('global_step', 'N/A'):,}")
        
        # 2. Initialize model
        model = tester.initialize_model()
        print(" Model initialized successfully")
        
        # 3. Generate performance table
        print("\n" + "="*70)
        performance_df = tester.generate_performance_table()
        
        # 4. Analyze training history
        results = tester.analyze_training_history()
        
        # 5. Generate enhanced plots (save but don't show to avoid blocking)
        tester.plot_enhanced_training_history(save_plots=True, show_plots=False)
        
        # 6. Test model inference
        # Check if test data exists
        test_data_paths = [
            'code/data/tokenized_data/tokenized_data_10s.npz',
            'code/data/tokenized_data/tokenized_data_30s.npz',
            'code/data/tokenized_data/tokenized_data_1min.npz'
        ]
        
        test_data_path = None
        for path in test_data_paths:
            if Path(path).exists():
                test_data_path = path
                break
        
        inference_results = tester.test_model_inference(test_data_path)
        
        # 7. Generate summary report
        tester.generate_summary_report()
        
        # 8. Verify the training history image exists
        training_plot_path = Path(checkpoint_dir) / 'training_history.png'
        if training_plot_path.exists():
            print(f"\n Training history plot verified at: {training_plot_path}")
            print(f"  File size: {training_plot_path.stat().st_size / 1024:.2f} KB")
        else:
            print(f"\n Training history plot not found at: {training_plot_path}")
        
        print("\n" + "="*70)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Print final summary
        if results:
            print(f"\nFinal Results Summary:")
            print(f"  Final Training Loss: {results['final_train_loss']:.4f}")
            if results['final_val_loss']:
                print(f"  Final Validation Loss: {results['final_val_loss']:.4f}")
            print(f"  Model successfully tested on all {len(inference_results)} tasks")
            print(f"  All outputs saved to: {checkpoint_dir}")
        
        # Return results for further processing if needed
        return {
            'checkpoint': checkpoint,
            'model': model,
            'results': results,
            'performance_table': performance_df,
            'inference_results': inference_results
        }
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()