"""
비디오 테스트 데이터 빠른 시작 가이드
"""

import os
from pathlib import Path
from video_data_recorder import VideoDataRecorder
from video_data_loader import VideoDataLoader


def example_1_record_from_video():
    """
    예제 1: 비디오 파일에서 테스트 데이터 기록
    """
    print("\n" + "="*60)
    print("예제 1: 비디오 파일에서 테스트 데이터 기록")
    print("="*60)
    
    # 테스트 비디오 찾기
    test_videos_dir = Path('data/test_videos')
    if not test_videos_dir.exists():
        print("⚠️  data/test_videos 디렉토리가 없습니다.")
        print("먼저 테스트 비디오 파일을 data/test_videos/에 저장하세요.")
        print("예: data/test_videos/my_video.mp4")
        return
    
    video_files = list(test_videos_dir.glob('*.mp4')) + list(test_videos_dir.glob('*.avi'))
    if not video_files:
        print("⚠️  mp4 또는 avi 파일을 찾을 수 없습니다.")
        print("data/test_videos/ 디렉토리에 비디오 파일을 저장하세요.")
        return
    
    video_path = str(video_files[0])
    print(f"\n📹 사용할 비디오: {video_path}")
    
    # 레코더 생성 및 처리
    recorder = VideoDataRecorder(
        output_dir='data/recorded_sessions',
        device='cpu'  # GPU가 있으면 'cuda' 사용
    )
    
    try:
        session_path = recorder.process_video(
            video_path=video_path,
            skip_frames=0  # 모든 프레임 처리 (빠르게 하려면 1 이상)
        )
        print(f"\n✅ 저장 완료!")
        print(f"📁 경로: {session_path}")
        return session_path
    except Exception as e:
        print(f"❌ 오류: {e}")
        return None


def example_2_load_and_analyze(session_path):
    """
    예제 2: 저장된 데이터 로드 및 분석
    """
    print("\n" + "="*60)
    print("예제 2: 저장된 데이터 로드 및 분석")
    print("="*60)
    
    if not session_path:
        print("⚠️  세션 경로가 제공되지 않았습니다.")
        return
    
    try:
        loader = VideoDataLoader(session_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 통계 출력
    print("\n📊 세션 통계:")
    stats = loader.get_statistics()
    print(f"  - 총 프레임: {stats['total_frames']}")
    print(f"  - 총 감지: {stats['total_detections']}")
    print(f"  - 비디오 길이: {stats['duration_seconds']:.1f}초")
    print(f"  - FPS: {stats['fps']:.1f}")
    print(f"  - 프레임당 평균 감지: {stats['average_detections_per_frame']:.2f}")
    print(f"  - 고유 객체 수: {stats['unique_objects']}")
    
    # 객체 종류별 분포
    print(f"\n  📦 객체 종류별 분포:")
    for class_name, count in sorted(
        stats['class_distribution'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        percentage = (count / stats['total_detections']) * 100
        bar = "█" * int(percentage / 2)
        print(f"    {class_name:12s}: {count:4d} ({percentage:5.1f}%) {bar}")


def example_3_frame_inspection(session_path):
    """
    예제 3: 개별 프레임 검사
    """
    print("\n" + "="*60)
    print("예제 3: 개별 프레임 검사")
    print("="*60)
    
    if not session_path:
        print("⚠️  세션 경로가 제공되지 않았습니다.")
        return
    
    try:
        loader = VideoDataLoader(session_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 몇 개 프레임 샘플 검사
    num_samples = min(5, loader.metadata['total_frames'])
    print(f"\n🔍 처음 {num_samples}개 프레임 검사:")
    
    for frame_idx in range(num_samples):
        detections = loader.get_detections_for_frame(frame_idx)
        frame_time = frame_idx / loader.metadata['fps']
        
        print(f"\n  프레임 {frame_idx} (시간: {frame_time:.2f}초) - {len(detections)}개 객체:")
        
        for det in detections:
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            print(f"    • {det['class_name']:10s} | "
                  f"신뢰도: {det['confidence']:.2f} | "
                  f"ID: {det['object_id']:3s} | "
                  f"크기: {width:.0f}x{height:.0f} | "
                  f"깊이: {det['depth_value']:.2f}")


def example_4_object_tracking(session_path):
    """
    예제 4: 객체 궤적 추적
    """
    print("\n" + "="*60)
    print("예제 4: 객체 궤적 추적")
    print("="*60)
    
    if not session_path:
        print("⚠️  세션 경로가 제공되지 않았습니다.")
        return
    
    try:
        loader = VideoDataLoader(session_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 모든 고유 객체 ID 찾기
    object_ids = set()
    for frame_data in loader.detections:
        for det in frame_data['detections']:
            if det['object_id'] is not None:
                object_ids.add(det['object_id'])
    
    if not object_ids:
        print("⚠️  추적된 객체가 없습니다.")
        return
    
    print(f"\n📍 {len(object_ids)}개의 고유 객체 궤적:")
    
    # 각 객체의 궤적 정보
    for obj_id in sorted(list(object_ids))[:5]:  # 처음 5개만 표시
        trajectory = loader.get_object_trajectory(obj_id)
        if trajectory:
            start_frame = trajectory[0]['frame_index']
            end_frame = trajectory[-1]['frame_index']
            class_name = trajectory[0]['class_name']
            
            print(f"\n  객체 ID {obj_id}: {class_name}")
            print(f"    - 감지 프레임: {len(trajectory)}개")
            print(f"    - 시간 범위: {start_frame/loader.metadata['fps']:.2f}s ~ "
                  f"{end_frame/loader.metadata['fps']:.2f}s")
            
            # 처음과 마지막 위치
            first_bbox = trajectory[0]['bbox']
            last_bbox = trajectory[-1]['bbox']
            first_center = ((first_bbox[0]+first_bbox[2])/2, (first_bbox[1]+first_bbox[3])/2)
            last_center = ((last_bbox[0]+last_bbox[2])/2, (last_bbox[1]+last_bbox[3])/2)
            
            movement = ((last_center[0]-first_center[0])**2 + 
                       (last_center[1]-first_center[1])**2)**0.5
            print(f"    - 이동 거리: {movement:.1f} 픽셀")


def example_5_batch_loading(session_path):
    """
    예제 5: 배치 로딩 및 처리
    """
    print("\n" + "="*60)
    print("예제 5: 배치 로딩 및 처리")
    print("="*60)
    
    if not session_path:
        print("⚠️  세션 경로가 제공되지 않았습니다.")
        return
    
    try:
        loader = VideoDataLoader(session_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"\n🔄 모든 프레임 순회 처리 (처음 10개 프레임만):")
    
    for i, (frame, detections) in enumerate(loader.get_all_frames_with_detections()):
        if i >= 10:
            break
        
        print(f"  프레임 {i}: 해상도 {frame.shape[1]}x{frame.shape[0]}, "
              f"{len(detections)}개 감지")


def main():
    """
    모든 예제 실행
    """
    print("\n" + "="*60)
    print("🚀 비디오 테스트 데이터 빠른 시작 가이드")
    print("="*60)
    
    # 예제 1: 데이터 기록
    session_path = example_1_record_from_video()
    
    if session_path:
        # 예제 2-5: 데이터 분석
        example_2_load_and_analyze(session_path)
        example_3_frame_inspection(session_path)
        example_4_object_tracking(session_path)
        example_5_batch_loading(session_path)
    
    print("\n" + "="*60)
    print("✅ 모든 예제 완료!")
    print("="*60)
    print("\n📚 다음 단계:")
    print("1. VIDEO_DATA_GUIDE.md 문서 참고")
    print("2. video_data_recorder.py로 더 많은 비디오 처리")
    print("3. video_data_loader.py로 데이터 분석")
    print("4. example_usage.py와 통합하여 테스트 진행")


if __name__ == '__main__':
    main()
