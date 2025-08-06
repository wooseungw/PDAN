#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_pdan_lightning import PDANLightningModule
from PDAN import PDAN
from charades_i3d_per_video import make_dataset

class ModelVisualizer:
    def __init__(self, checkpoint_path, data_root, json_path, device='cuda'):
        """
        체크포인트를 사용한 PDAN 모델 시각화
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            data_root: 비디오 데이터 루트 경로
            json_path: charades.json 파일 경로
            device: 사용할 디바이스
        """
        self.device = device
        self.data_root = data_root
        
        # JSON 데이터 로드
        print("Loading dataset metadata...")
        with open(json_path, 'r') as f:
            self.dataset_metadata = json.load(f)
        
        # 모델 로드
        print(f"Loading model from checkpoint: {checkpoint_path}")
        self.model = PDANLightningModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(device)
        
        # Action classes (Charades 데이터셋의 157개 클래스)
        self.action_classes = self._load_action_classes()
        
        # 클래스별 고정 색상 매핑 생성
        self.class_colors = self._generate_class_colors()
        
        print(f"Model loaded successfully. Device: {device}")
        print(f"Number of action classes: {len(self.action_classes)}")
    
    def _load_action_classes(self):
        """액션 클래스 이름들을 로드 (번호 기반)"""
        # Charades 데이터셋의 액션 클래스들 (0-156)
        # 실제 액션 이름이 있다면 여기에 추가하세요
        return [f"Action_{i:03d}" for i in range(157)]
    
    def _generate_class_colors(self):
        """각 액션 클래스에 대해 고정된 색상 매핑 생성"""
        # 시각적으로 구분하기 쉬운 색상 팔레트 사용
        base_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D2B4DE',
            '#A9DFBF', '#F9E79F', '#D5A6BD', '#A3E4D7', '#FCF3CF'
        ]
        
        # 157개 클래스에 대해 색상 생성 (기본 색상을 반복하면서 약간씩 변형)
        colors = {}
        for i in range(157):
            base_idx = i % len(base_colors)
            base_color = base_colors[base_idx]
            
            # HSV로 변환하여 밝기나 채도를 약간 조정
            import matplotlib.colors as mcolors
            rgb = mcolors.hex2color(base_color)
            hsv = mcolors.rgb_to_hsv(rgb)
            
            # 같은 기본 색상을 사용하는 경우 밝기를 조정
            cycle = i // len(base_colors)
            if cycle > 0:
                # 밝기를 순환적으로 조정 (0.7 ~ 1.0)
                hsv = (hsv[0], hsv[1], 0.7 + (cycle % 4) * 0.1)
            
            rgb_adjusted = mcolors.hsv_to_rgb(hsv)
            colors[i] = rgb_adjusted
            
        return colors
    
    def get_test_videos(self, max_videos=20):
        """테스트 비디오 리스트 반환"""
        test_videos = []
        for video_id, metadata in self.dataset_metadata.items():
            if metadata.get('subset') == 'testing':
                test_videos.append(video_id)
                if len(test_videos) >= max_videos:
                    break
        return test_videos
    
    def load_video_data(self, video_id):
        """비디오 데이터와 메타데이터 로드"""
        # I3D 특징 파일 로드
        feature_path = os.path.join(self.data_root, f"{video_id}.npy")
        if not os.path.exists(feature_path):
            print(f"Warning: Feature file not found: {feature_path}")
            return None, None, None
        
        features = np.load(feature_path)  # Shape: (T, 1024)
        
        # 메타데이터 가져오기
        metadata = self.dataset_metadata[video_id]
        duration = metadata['duration']
        actions = metadata['actions']
        
        return features, duration, actions
    
    def predict_video(self, features):
        """비디오에 대한 예측 수행"""
        # 데이터 형태 확인 및 정리: (T, 1, 1, 1024) -> (T, 1024)
        if len(features.shape) == 4:
            features = features.squeeze()  # (T, 1024)
        elif len(features.shape) == 3 and features.shape[1] == 1:
            features = features.squeeze(1)  # (T, 1024)
        
        # 텐서로 변환하고 배치 차원 추가
        features_tensor = torch.FloatTensor(features).transpose(0, 1).unsqueeze(0)  # (1, 1024, T)
        features_tensor = features_tensor.to(self.device)
        
        # 마스크 생성 (모든 프레임이 유효하다고 가정)
        T = features_tensor.shape[2]
        mask = torch.ones(1, 1, T).to(self.device)
        
        with torch.no_grad():
            # 모델 예측
            outputs = self.model.model(features_tensor, mask)  # Shape: (num_stages, 1, num_classes, T)
            
            # 마지막 스테이지의 출력 사용
            predictions = outputs[-1, 0]  # Shape: (num_classes, T)
            
            # 시그모이드 적용 (다중 라벨 분류)
            predictions = torch.sigmoid(predictions)
            
        return predictions.cpu().numpy()  # Shape: (num_classes, T)
    
    def visualize_video_predictions(self, video_id, save_path=None, threshold=0.5):
        """단일 비디오의 예측 결과 시각화"""
        # 데이터 로드
        features, duration, gt_actions = self.load_video_data(video_id)
        if features is None:
            return
        
        # 예측 수행
        predictions = self.predict_video(features)  # (num_classes, T)
        
        T = predictions.shape[1]
        time_steps = np.linspace(0, duration, T)
        
        # 임계값 이상의 예측만 선택
        confident_predictions = predictions > threshold
        active_classes = np.where(np.any(confident_predictions, axis=1))[0]
        
        if len(active_classes) == 0:
            print(f"No confident predictions for video {video_id}")
            return
        
        # 그래프 설정
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 상단: Ground Truth 액션들
        ax1.set_title(f"Video {video_id} - Ground Truth Actions", fontsize=14, fontweight='bold')
        ax1.set_xlim(0, duration)
        ax1.set_ylabel("Actions")
        
        gt_y_pos = 0
        
        for i, (action_id, start_time, end_time) in enumerate(gt_actions):
            # 액션 ID에 따른 고정 색상 사용
            color = self.class_colors[action_id]
            rect = patches.Rectangle((start_time, gt_y_pos), end_time - start_time, 0.8,
                                   linewidth=1, edgecolor='black', 
                                   facecolor=color, alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(start_time + (end_time - start_time)/2, gt_y_pos + 0.4,
                    f"GT_{action_id}", ha='center', va='center', fontsize=8)
            gt_y_pos += 1
        
        ax1.set_ylim(-0.5, gt_y_pos + 0.5)
        ax1.grid(True, alpha=0.3)
        
        # 하단: 예측된 액션들
        ax2.set_title(f"Predicted Actions (threshold={threshold})", fontsize=14, fontweight='bold')
        ax2.set_xlim(0, duration)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Action Classes")
        
        # 예측 결과 시각화 - 클래스 ID에 따른 고정 색상 사용
        for i, class_idx in enumerate(active_classes):
            class_predictions = predictions[class_idx]
            # 클래스 ID에 해당하는 고정 색상 사용
            class_color = self.class_colors[class_idx]
            
            # 연속된 활성 구간 찾기
            active_frames = confident_predictions[class_idx]
            starts_ends = []
            start = None
            
            for t in range(T):
                if active_frames[t] and start is None:
                    start = t
                elif not active_frames[t] and start is not None:
                    starts_ends.append((start, t-1))
                    start = None
            if start is not None:
                starts_ends.append((start, T-1))
            
            # 각 구간을 그리기
            for start_frame, end_frame in starts_ends:
                start_time = time_steps[start_frame]
                end_time = time_steps[end_frame]
                
                # 구간의 평균 확신도
                avg_confidence = np.mean(class_predictions[start_frame:end_frame+1])
                
                rect = patches.Rectangle((start_time, i), end_time - start_time, 0.8,
                                       linewidth=1, edgecolor='black',
                                       facecolor=class_color, 
                                       alpha=min(avg_confidence * 2, 1.0))
                ax2.add_patch(rect)
                
                # 확신도 표시
                ax2.text(start_time + (end_time - start_time)/2, i + 0.4,
                        f"{self.action_classes[class_idx][:10]}\n{avg_confidence:.2f}",
                        ha='center', va='center', fontsize=7)
        
        ax2.set_ylim(-0.5, len(active_classes) + 0.5)
        ax2.set_yticks(range(len(active_classes)))
        ax2.set_yticklabels([self.action_classes[idx] for idx in active_classes])
        ax2.grid(True, alpha=0.3)
        
        # 범례 추가 - GT와 예측에서 공통으로 나타나는 액션들
        gt_action_ids = set([action[0] for action in gt_actions])
        common_actions = gt_action_ids.intersection(set(active_classes))
        
        if common_actions:
            # 범례용 패치 생성
            legend_patches = []
            for action_id in sorted(common_actions):
                color = self.class_colors[action_id]
                patch = patches.Patch(color=color, label=f'Action {action_id}')
                legend_patches.append(patch)
            
            # 범례 위치 조정 (그래프 오른쪽)
            ax2.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_confidence_heatmap(self, video_id, save_path=None, top_k=20):
        """액션별 확신도 히트맵 생성"""
        features, duration, gt_actions = self.load_video_data(video_id)
        if features is None:
            return
        
        predictions = self.predict_video(features)  # (num_classes, T)
        
        # 전체 비디오에서 가장 활성화된 상위 k개 클래스 선택
        max_activations = np.max(predictions, axis=1)
        top_classes = np.argsort(max_activations)[-top_k:][::-1]
        
        # 히트맵 데이터 준비
        heatmap_data = predictions[top_classes]
        
        # 시간 축 라벨
        T = predictions.shape[1]
        time_labels = [f"{i*duration/T:.1f}s" for i in range(0, T, max(1, T//10))]
        
        # 히트맵 생성 - 클래스별 색상 정보를 y축 라벨에 반영
        plt.figure(figsize=(15, 8))
        
        # 클래스별 색상 바 추가
        fig, (ax_color, ax_heatmap) = plt.subplots(1, 2, figsize=(16, 8), 
                                                  gridspec_kw={'width_ratios': [0.5, 10]})
        
        # 왼쪽: 클래스별 색상 바
        for i, class_idx in enumerate(top_classes):
            color = self.class_colors[class_idx]
            ax_color.barh(i, 1, color=color, alpha=0.8)
            ax_color.text(0.5, i, f"{class_idx}", ha='center', va='center', fontweight='bold')
        
        ax_color.set_ylim(-0.5, len(top_classes) - 0.5)
        ax_color.set_xlim(0, 1)
        ax_color.set_ylabel("Action Classes")
        ax_color.set_title("Class\nColors")
        ax_color.set_xticks([])
        ax_color.set_yticks(range(len(top_classes)))
        ax_color.set_yticklabels([self.action_classes[idx] for idx in top_classes])
        
        # 오른쪽: 히트맵
        sns.heatmap(heatmap_data, 
                   xticklabels=time_labels,
                   yticklabels=[f"Action_{idx}" for idx in top_classes],
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Confidence Score'},
                   ax=ax_heatmap)
        
        ax_heatmap.set_title(f"Action Confidence Heatmap - Video {video_id}", fontsize=14, fontweight='bold')
        ax_heatmap.set_xlabel("Time")
        ax_heatmap.set_ylabel("")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_multiple_videos(self, video_ids=None, output_dir="visualizations", max_videos=5):
        """여러 비디오 분석"""
        if video_ids is None:
            video_ids = self.get_test_videos(max_videos)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Analyzing {len(video_ids)} videos...")
        
        for video_id in tqdm(video_ids):
            try:
                # 예측 시각화
                pred_save_path = os.path.join(output_dir, f"{video_id}_predictions.png")
                self.visualize_video_predictions(video_id, save_path=pred_save_path)
                
                # 확신도 히트맵
                heatmap_save_path = os.path.join(output_dir, f"{video_id}_heatmap.png")
                self.create_confidence_heatmap(video_id, save_path=heatmap_save_path)
                
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue
        
        print(f"Analysis complete. Results saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize PDAN model predictions")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", default="data", help="Root directory for video features")
    parser.add_argument("--json_path", default="charades.json", help="Path to charades.json")
    parser.add_argument("--output_dir", default="visualizations", help="Output directory")
    parser.add_argument("--video_id", help="Specific video ID to analyze")
    parser.add_argument("--max_videos", type=int, default=5, help="Maximum number of videos to analyze")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 체크포인트 파일 확인
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # 비주얼라이저 초기화
    visualizer = ModelVisualizer(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        json_path=args.json_path,
        device=args.device
    )
    
    if args.video_id:
        # 특정 비디오 분석
        print(f"Analyzing video: {args.video_id}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        pred_path = os.path.join(args.output_dir, f"{args.video_id}_predictions.png")
        heatmap_path = os.path.join(args.output_dir, f"{args.video_id}_heatmap.png")
        
        visualizer.visualize_video_predictions(args.video_id, save_path=pred_path, threshold=args.threshold)
        visualizer.create_confidence_heatmap(args.video_id, save_path=heatmap_path)
    else:
        # 여러 비디오 분석
        visualizer.analyze_multiple_videos(
            output_dir=args.output_dir,
            max_videos=args.max_videos
        )

if __name__ == "__main__":
    main()
