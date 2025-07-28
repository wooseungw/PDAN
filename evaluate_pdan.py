#!/usr/bin/env python3
"""
PDAN Model Evaluation and Visualization
Load trained checkpoint and perform validation with visualization
"""



import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_recall_curve
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PDAN import PDAN
from apmeter import APMeter
from charades_i3d_per_video import MultiThumos as Dataset
from charades_i3d_per_video import mt_collate_fn as collate_fn
from train_pdan_lightning import PDANLightningModule, PDANDataModule


class PDANEvaluator:
    """PDAN Model Evaluator with Visualization"""
    
    def __init__(self, checkpoint_path, data_root, split_file, device='cuda'):
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.split_file = split_file
        self.device = device
        
        # Load model from checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        self.model = PDANLightningModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(device)
        
        # Load action classes (Charades dataset)
        self.action_classes = self.load_action_classes()
        
        # Setup dataset
        self.setup_dataset()
        
    def load_action_classes(self):
        """Load Charades action class names"""
        # Charades 157 action classes - simplified version
        # In practice, you would load from a proper class mapping file
        return [f"action_{i:03d}" for i in range(157)]
    
    def setup_dataset(self):
        """Setup evaluation dataset"""
        self.dataset = Dataset(
            self.split_file, 
            'testing',  # Use testing split for evaluation
            self.data_root, 
            1,  # batch_size = 1 for evaluation
            157  # num_classes
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        print(f"Evaluation dataset loaded: {len(self.dataset)} samples")
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_video_ids = []
        results_per_video = {}
        
        apm = APMeter()
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                inputs, mask, labels, other = batch
                video_id = other[0][0]
                duration = other[1][0]
                
                # Move to device
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                
                # Prepare data and masks (same as training)
                inputs, mask, mask_new, labels, other = self.model.prepare_data_and_masks(batch)
                
                # Forward pass
                activation = self.model.forward(inputs, mask_new)
                outputs_final = activation[-1]  # Take last stage output
                outputs_final = outputs_final.permute(0, 2, 1)  # [B, T, C]
                
                # Calculate probabilities
                probs = F.sigmoid(outputs_final) * mask.unsqueeze(2)
                
                # Store results
                probs_np = probs.detach().cpu().numpy()[0]  # [T, C]
                labels_np = labels.detach().cpu().numpy()[0]  # [T, C]
                
                # Add to AP meter
                apm.add(probs_np, labels_np)
                
                # Store per-video results
                results_per_video[video_id] = {
                    'predictions': probs_np,
                    'labels': labels_np,
                    'duration': duration,
                    'fps': probs_np.shape[0] / duration
                }
                
                all_predictions.append(probs_np)
                all_labels.append(labels_np)
                all_video_ids.append(video_id)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(self.dataloader)} videos")
        
        # Calculate overall mAP
        ap_scores = apm.value()
        mean_ap = np.mean(ap_scores) * 100
        
        print(f"\nEvaluation Results:")
        print(f"Mean Average Precision (mAP): {mean_ap:.2f}%")
        
        return {
            'mean_ap': mean_ap,
            'ap_scores': ap_scores,
            'results_per_video': results_per_video,
            'all_predictions': all_predictions,
            'all_labels': all_labels,
            'video_ids': all_video_ids
        }
    
    def visualize_results(self, results, output_dir='./evaluation_results'):
        """Create various visualizations of the results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. mAP per action class
        self.plot_map_per_class(results['ap_scores'], output_dir)
        
        # 2. Precision-Recall curves for top classes
        self.plot_precision_recall_curves(results, output_dir)
        
        # 3. Temporal action predictions for sample videos
        self.plot_temporal_predictions(results['results_per_video'], output_dir)
        
        # 4. Confusion matrix visualization
        self.plot_prediction_statistics(results, output_dir)
        
        print(f"\nVisualization results saved to: {output_dir}")
    
    def plot_map_per_class(self, ap_scores, output_dir):
        """Plot mAP scores for each action class"""
        plt.figure(figsize=(20, 8))
        
        # Convert to percentage
        ap_scores_pct = ap_scores * 100
        
        # Sort for better visualization
        sorted_indices = np.argsort(ap_scores_pct)[::-1]
        sorted_scores = ap_scores_pct[sorted_indices]
        sorted_classes = [self.action_classes[i] for i in sorted_indices]
        
        # Show top 30 classes
        top_n = min(30, len(sorted_scores))
        
        plt.bar(range(top_n), sorted_scores[:top_n])
        plt.title(f'Average Precision (AP) per Action Class (Top {top_n})', fontsize=16)
        plt.xlabel('Action Classes', fontsize=12)
        plt.ylabel('Average Precision (%)', fontsize=12)
        plt.xticks(range(top_n), [f"Class {sorted_indices[i]}" for i in range(top_n)], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/map_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save statistics
        stats_df = pd.DataFrame({
            'Class_ID': sorted_indices[:top_n],
            'AP_Score': sorted_scores[:top_n]
        })
        stats_df.to_csv(f'{output_dir}/class_ap_scores.csv', index=False)
    
    def plot_precision_recall_curves(self, results, output_dir):
        """Plot Precision-Recall curves for top performing classes"""
        # Combine all predictions and labels
        all_preds = np.concatenate(results['all_predictions'], axis=0)  # [N_frames, 157]
        all_labels = np.concatenate(results['all_labels'], axis=0)  # [N_frames, 157]
        
        # Get top 10 classes by AP score
        top_classes = np.argsort(results['ap_scores'])[-10:]
        
        plt.figure(figsize=(12, 8))
        
        for class_idx in top_classes:
            y_true = all_labels[:, class_idx]
            y_scores = all_preds[:, class_idx]
            
            # Skip if no positive samples
            if np.sum(y_true) == 0:
                continue
                
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap_score = results['ap_scores'][class_idx] * 100
            
            plt.plot(recall, precision, 
                    label=f'Class {class_idx} (AP={ap_score:.1f}%)', 
                    linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves (Top 10 Classes)', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_predictions(self, results_per_video, output_dir, max_videos=5):
        """Plot temporal action predictions for sample videos"""
        video_ids = list(results_per_video.keys())[:max_videos]
        
        for i, video_id in enumerate(video_ids):
            result = results_per_video[video_id]
            predictions = result['predictions']  # [T, 157]
            labels = result['labels']  # [T, 157]
            duration = result['duration']
            
            # Get top predicted classes
            max_preds_per_frame = np.max(predictions, axis=1)
            top_classes = np.argsort(np.sum(predictions, axis=0))[-5:]  # Top 5 classes
            
            # Create time axis
            time_axis = np.linspace(0, duration, predictions.shape[0])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot predictions
            for class_idx in top_classes:
                ax1.plot(time_axis, predictions[:, class_idx], 
                        label=f'Class {class_idx}', linewidth=2, alpha=0.8)
            
            ax1.set_title(f'Action Predictions - Video: {video_id}', fontsize=14)
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Prediction Score', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot ground truth (if available)
            gt_actions = np.where(labels > 0.5)
            if len(gt_actions[0]) > 0:
                for frame, class_idx in zip(gt_actions[0], gt_actions[1]):
                    ax2.scatter(time_axis[frame], class_idx, 
                              c='red', s=20, alpha=0.6)
            
            ax2.set_title(f'Ground Truth Actions - Video: {video_id}', fontsize=14)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Action Class', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/temporal_predictions_{video_id}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_prediction_statistics(self, results, output_dir):
        """Plot various prediction statistics"""
        all_preds = np.concatenate(results['all_predictions'], axis=0)
        all_labels = np.concatenate(results['all_labels'], axis=0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction score distribution
        ax1.hist(all_preds.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Distribution of Prediction Scores', fontsize=14)
        ax1.set_xlabel('Prediction Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Ground truth distribution
        gt_counts = np.sum(all_labels, axis=0)
        ax2.hist(gt_counts, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Distribution of Ground Truth Actions per Class', fontsize=14)
        ax2.set_xlabel('Number of Positive Frames', fontsize=12)
        ax2.set_ylabel('Number of Classes', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. AP score distribution
        ap_scores_pct = results['ap_scores'] * 100
        ax3.hist(ap_scores_pct, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Distribution of AP Scores', fontsize=14)
        ax3.set_xlabel('Average Precision (%)', fontsize=12)
        ax3.set_ylabel('Number of Classes', fontsize=12)
        ax3.axvline(x=np.mean(ap_scores_pct), color='red', linestyle='--', 
                   label=f'Mean AP: {np.mean(ap_scores_pct):.1f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction vs Ground Truth scatter
        frame_means_pred = np.mean(all_preds, axis=1)
        frame_means_gt = np.mean(all_labels, axis=1)
        ax4.scatter(frame_means_gt, frame_means_pred, alpha=0.5, s=10)
        ax4.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        ax4.set_title('Frame-level Prediction vs Ground Truth', fontsize=14)
        ax4.set_xlabel('Mean Ground Truth', fontsize=12)
        ax4.set_ylabel('Mean Prediction', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, results, output_dir):
        """Save detailed results to files"""
        # Save overall results
        summary = {
            'mean_ap': float(results['mean_ap']),
            'ap_per_class': results['ap_scores'].tolist(),
            'num_videos': len(results['video_ids']),
            'num_classes': len(results['ap_scores'])
        }
        
        with open(f'{output_dir}/evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save per-video results summary
        video_summaries = []
        for video_id, result in results['results_per_video'].items():
            preds = result['predictions']
            labels = result['labels']
            
            video_summary = {
                'video_id': video_id,
                'duration': float(result['duration']),
                'fps': float(result['fps']),
                'num_frames': preds.shape[0],
                'mean_prediction': float(np.mean(preds)),
                'max_prediction': float(np.max(preds)),
                'num_gt_actions': int(np.sum(labels > 0.5)),
                'active_classes': int(np.sum(np.max(labels, axis=0) > 0.5))
            }
            video_summaries.append(video_summary)
        
        video_df = pd.DataFrame(video_summaries)
        video_df.to_csv(f'{output_dir}/video_results_summary.csv', index=False)
        
        print(f"Detailed results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='PDAN Model Evaluation and Visualization')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_root', type=str, 
                       default='/data/1_personal/4_SWWOO/actiondetect/PDAN/data',
                       help='Root directory for data')
    parser.add_argument('--split_file', type=str,
                       default='/data/1_personal/4_SWWOO/actiondetect/PDAN/data/charades.json',
                       help='Data split JSON file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results and visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    print("=" * 60)
    print("PDAN Model Evaluation and Visualization")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Create evaluator
    evaluator = PDANEvaluator(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split_file=args.split_file,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_model()
    
    # Create visualizations
    evaluator.visualize_results(results, args.output_dir)
    
    # Save detailed results
    evaluator.save_detailed_results(results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

# %%
