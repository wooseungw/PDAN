#!/usr/bin/env python3
"""
PDAN 모델 시각화 사용 예시
다양한 방법으로 모델의 예측 결과를 시각화하는 방법을 보여줍니다.
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_model import ModelVisualizer

def example_1_basic_usage():
    """기본 사용법: 여러 비디오 분석"""
    print("=== Example 1: Basic Usage ===")
    
    # 모델 시각화 객체 생성
    visualizer = ModelVisualizer(
        checkpoint_path="pdan_original_25.33.ckpt",
        data_root="data",
        json_path="charades.json",
        device='cuda'
    )
    
    # 5개 비디오 분석
    visualizer.analyze_multiple_videos(
        output_dir="visualizations/basic",
        max_videos=5
    )
    
    print("Basic analysis completed!\n")

def example_2_specific_video():
    """특정 비디오 분석"""
    print("=== Example 2: Specific Video Analysis ===")
    
    visualizer = ModelVisualizer(
        checkpoint_path="pdan_original_25.33.ckpt",
        data_root="data",
        json_path="charades.json",
        device='cuda'
    )
    
    # 테스트 비디오 목록 가져오기
    test_videos = visualizer.get_test_videos(max_videos=10)
    
    if test_videos:
        # 첫 번째 비디오 분석
        video_id = test_videos[0]
        print(f"Analyzing video: {video_id}")
        
        # 예측 시각화
        visualizer.visualize_video_predictions(
            video_id=video_id,
            save_path=f"visualizations/specific/{video_id}_predictions.png",
            threshold=0.5
        )
        
        # 확신도 히트맵
        visualizer.create_confidence_heatmap(
            video_id=video_id,
            save_path=f"visualizations/specific/{video_id}_heatmap.png",
            top_k=15
        )
        
        print(f"Specific video analysis completed for {video_id}!\n")
    else:
        print("No test videos found!\n")

def example_3_threshold_comparison():
    """다양한 임계값으로 비교 분석"""
    print("=== Example 3: Threshold Comparison ===")
    
    visualizer = ModelVisualizer(
        checkpoint_path="pdan_original_25.33.ckpt",
        data_root="data",
        json_path="charades.json",
        device='cuda'
    )
    
    test_videos = visualizer.get_test_videos(max_videos=3)
    thresholds = [0.3, 0.5, 0.7]
    
    for video_id in test_videos:
        print(f"Analyzing {video_id} with different thresholds...")
        
        for threshold in thresholds:
            output_path = f"visualizations/threshold_comparison/{video_id}_th{threshold}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            visualizer.visualize_video_predictions(
                video_id=video_id,
                save_path=output_path,
                threshold=threshold
            )
    
    print("Threshold comparison completed!\n")

def example_4_batch_analysis():
    """배치 분석: 특정 비디오 리스트"""
    print("=== Example 4: Batch Analysis ===")
    
    visualizer = ModelVisualizer(
        checkpoint_path="pdan_original_25.33.ckpt",
        data_root="data", 
        json_path="charades.json",
        device='cuda'
    )
    
    # 특정 비디오 리스트로 분석
    test_videos = visualizer.get_test_videos(max_videos=20)
    selected_videos = test_videos[:3]  # 처음 3개 선택
    
    print(f"Selected videos: {selected_videos}")
    
    visualizer.analyze_multiple_videos(
        video_ids=selected_videos,
        output_dir="visualizations/batch",
        max_videos=len(selected_videos)
    )
    
    print("Batch analysis completed!\n")

def example_5_custom_analysis():
    """커스텀 분석: 프로그래밍 방식"""
    print("=== Example 5: Custom Analysis ===")
    
    visualizer = ModelVisualizer(
        checkpoint_path="pdan_original_25.33.ckpt",
        data_root="data",
        json_path="charades.json", 
        device='cuda'
    )
    
    test_videos = visualizer.get_test_videos(max_videos=5)
    
    # 각 비디오별 통계 수집
    video_stats = []
    
    for video_id in test_videos:
        try:
            # 데이터 로드
            features, duration, gt_actions = visualizer.load_video_data(video_id)
            if features is None:
                continue
                
            # 예측 수행
            predictions = visualizer.predict_video(features)
            
            # 통계 계산
            max_confidence = predictions.max()
            avg_confidence = predictions.mean()
            active_classes = (predictions > 0.5).sum(axis=0).max()  # 최대 동시 액션 수
            
            video_stats.append({
                'video_id': video_id,
                'duration': duration,
                'gt_actions': len(gt_actions),
                'max_confidence': max_confidence,
                'avg_confidence': avg_confidence,
                'max_simultaneous_actions': active_classes
            })
            
            print(f"Video {video_id}: {duration:.1f}s, {len(gt_actions)} GT actions, "
                  f"max conf: {max_confidence:.3f}, avg conf: {avg_confidence:.3f}")
                  
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue
    
    # 통계 요약
    if video_stats:
        avg_duration = sum(s['duration'] for s in video_stats) / len(video_stats)
        avg_gt_actions = sum(s['gt_actions'] for s in video_stats) / len(video_stats)
        
        print(f"\nSummary of {len(video_stats)} videos:")
        print(f"  Average duration: {avg_duration:.1f} seconds")
        print(f"  Average GT actions per video: {avg_gt_actions:.1f}")
    
    print("Custom analysis completed!\n")

def main():
    """모든 예시 실행"""
    print("PDAN Model Visualization Examples")
    print("=" * 40)
    
    # 출력 디렉토리 생성
    os.makedirs("visualizations", exist_ok=True)
    
    try:
        # 예시 1: 기본 사용법
        example_1_basic_usage()
        
        # 예시 2: 특정 비디오 분석
        example_2_specific_video()
        
        # 예시 3: 임계값 비교
        example_3_threshold_comparison()
        
        # 예시 4: 배치 분석
        example_4_batch_analysis()
        
        # 예시 5: 커스텀 분석
        example_5_custom_analysis()
        
        print("All examples completed successfully!")
        print("\nGenerated visualizations:")
        print("  - visualizations/basic/: Multiple videos analysis")
        print("  - visualizations/specific/: Single video detailed analysis")
        print("  - visualizations/threshold_comparison/: Different thresholds")
        print("  - visualizations/batch/: Selected videos batch analysis")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please check:")
        print("  1. Checkpoint file exists")
        print("  2. Data directory contains .npy feature files")
        print("  3. charades.json file exists")
        print("  4. GPU is available (or use --device cpu)")

if __name__ == "__main__":
    main()
