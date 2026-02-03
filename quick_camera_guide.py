"""
실시간 카메라 기록 빠른 시작 예제
"""

import cv2
import numpy as np
from pathlib import Path
from video_data_loader import VideoDataLoader


def example_1_basic_recording():
    """
    예제 1: 기본 카메라 기록
    
    터미널에서:
    python realtime_camera_recorder.py --duration 10
    """
    print("\n" + "="*60)
    print("예제 1: 기본 카메라 기록")
    print("="*60)
    print("""
    카메라로부터 데이터 기록:
    
    터미널 실행:
    $ python realtime_camera_recorder.py --duration 10
    
    결과:
    - data/recorded_sessions/camera_YYYYMMDD_HHMMSS/ 생성
    - RGB 프레임, 깊이맵, 감지 정보 저장
    - 10초 후 자동 종료
    """)


def example_2_load_and_inspect(session_path=None):
    """
    예제 2: 저장된 데이터 로드 및 검사
    """
    print("\n" + "="*60)
    print("예제 2: 저장된 카메라 데이터 로드 및 검사")
    print("="*60)
    
    # 최신 카메라 세션 찾기
    if not session_path:
        import glob
        camera_sessions = sorted(
            glob.glob('data/recorded_sessions/camera_*'),
            reverse=True
        )
        if not camera_sessions:
            print("⚠️  저장된 카메라 세션이 없습니다.")
            print("먼저 realtime_camera_recorder.py를 실행하세요.")
            return
        session_path = camera_sessions[0]
    
    try:
        loader = VideoDataLoader(session_path)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print(f"\n✓ 세션 로드: {loader.metadata['session_name']}")
    print(f"  - 카메라 모델: {loader.metadata.get('camera_model', 'N/A')}")
    print(f"  - 데이터 타입: {'카메라' if loader.is_camera_data else '비디오'}")
    
    # 기본 정보 출력
    print(f"\n📊 기록 정보:")
    print(f"  - 총 프레임: {loader.metadata['total_frames']}")
    print(f"  - 기록 시간: {loader.metadata['recording_duration_seconds']:.1f}초")
    print(f"  - 평균 FPS: {loader.metadata['average_fps']:.1f}")
    print(f"  - 해상도: 640x480")
    
    # 감지 통계
    print(f"\n📦 객체 감지:")
    print(f"  - 총 감지: {loader.metadata['total_detections']}")
    for class_name, count in sorted(
        loader.metadata['class_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / loader.metadata['total_detections']) * 100
        print(f"    • {class_name}: {count} ({percentage:.1f}%)")


def example_3_depth_analysis(session_path=None):
    """
    예제 3: 깊이 정보 분석
    """
    print("\n" + "="*60)
    print("예제 3: 깊이 정보 분석")
    print("="*60)
    
    # 최신 카메라 세션 찾기
    if not session_path:
        import glob
        camera_sessions = sorted(
            glob.glob('data/recorded_sessions/camera_*'),
            reverse=True
        )
        if not camera_sessions:
            print("⚠️  저장된 카메라 세션이 없습니다.")
            return
        session_path = camera_sessions[0]
    
    try:
        loader = VideoDataLoader(session_path)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    if not loader.is_camera_data:
        print("⚠️  이 세션은 카메라 데이터가 아닙니다.")
        return
    
    print(f"\n🎥 카메라 깊이 분석: {session_path}")
    
    # 처음 10개 프레임 분석
    num_samples = min(10, loader.metadata['total_frames'])
    print(f"\n🔍 처음 {num_samples}개 프레임의 깊이 정보:\n")
    
    for frame_idx in range(num_samples):
        detections = loader.get_detections_for_frame(frame_idx)
        
        if not detections:
            continue
        
        # 이 프레임의 깊이 통계
        depths = [d['depth_meters'] for d in detections if d['depth_meters'] > 0]
        if not depths:
            continue
        
        frame_time = frame_idx / loader.metadata['average_fps']
        print(f"  프레임 {frame_idx} (시간: {frame_time:.2f}초):")
        print(f"    - 감지된 객체: {len(detections)}")
        print(f"    - 깊이 범위: {min(depths):.2f}m ~ {max(depths):.2f}m")
        print(f"    - 평균 깊이: {np.mean(depths):.2f}m")
        
        # 가장 가까운 객체
        closest = min(detections, key=lambda d: d['depth_meters'])
        if closest['depth_meters'] > 0:
            print(f"    - 가장 가까운: {closest['class_name']} at {closest['depth_meters']:.2f}m")
        print()


def example_4_distance_based_filtering(session_path=None):
    """
    예제 4: 거리 기반 객체 필터링
    """
    print("\n" + "="*60)
    print("예제 4: 거리 기반 객체 필터링")
    print("="*60)
    
    # 최신 카메라 세션 찾기
    if not session_path:
        import glob
        camera_sessions = sorted(
            glob.glob('data/recorded_sessions/camera_*'),
            reverse=True
        )
        if not camera_sessions:
            print("⚠️  저장된 카메라 세션이 없습니다.")
            return
        session_path = camera_sessions[0]
    
    try:
        loader = VideoDataLoader(session_path)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    if not loader.is_camera_data:
        print("⚠️  이 세션은 카메라 데이터가 아닙니다.")
        return
    
    print(f"\n📏 거리별 객체 분류:\n")
    
    # 거리 범위별 통계
    zones = {
        'very_close': (0, 0.5, '매우 가까움 (0~0.5m)'),
        'close': (0.5, 1.0, '가까움 (0.5~1.0m)'),
        'medium': (1.0, 2.0, '중간 (1.0~2.0m)'),
        'far': (2.0, 5.0, '멀음 (2.0~5.0m)'),
        'very_far': (5.0, 100, '매우 멀음 (5.0m 이상)')
    }
    
    zone_counts = {zone: {} for zone in zones}
    
    # 모든 프레임 분석
    for frame_idx in range(loader.metadata['total_frames']):
        detections = loader.get_detections_for_frame(frame_idx)
        
        for detection in detections:
            depth = detection['depth_meters']
            class_name = detection['class_name']
            
            if depth <= 0:
                continue
            
            # 해당 존에 추가
            for zone_name, (min_d, max_d, label) in zones.items():
                if min_d <= depth < max_d:
                    if class_name not in zone_counts[zone_name]:
                        zone_counts[zone_name][class_name] = 0
                    zone_counts[zone_name][class_name] += 1
    
    # 출력
    for zone_name, (min_d, max_d, label) in zones.items():
        count = sum(zone_counts[zone_name].values())
        if count > 0:
            print(f"  🔴 {label}: {count}개")
            for class_name, class_count in sorted(
                zone_counts[zone_name].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"     • {class_name}: {class_count}")
            print()


def example_5_frame_visualization(session_path=None, frame_idx=0):
    """
    예제 5: 프레임 시각화 (깊이맵 포함)
    """
    print("\n" + "="*60)
    print("예제 5: 프레임 시각화")
    print("="*60)
    
    # 최신 카메라 세션 찾기
    if not session_path:
        import glob
        camera_sessions = sorted(
            glob.glob('data/recorded_sessions/camera_*'),
            reverse=True
        )
        if not camera_sessions:
            print("⚠️  저장된 카메라 세션이 없습니다.")
            return
        session_path = camera_sessions[0]
    
    try:
        loader = VideoDataLoader(session_path)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print(f"\n🎨 프레임 시각화: {session_path}")
    print(f"   프레임 {frame_idx}\n")
    
    # 프레임 로드
    frame = loader.get_frame(frame_idx)
    depth_map = loader.get_depth_map(frame_idx)
    detections = loader.get_detections_for_frame(frame_idx)
    
    if frame is None:
        print("⚠️  프레임을 로드할 수 없습니다.")
        return
    
    print(f"✓ 로드 완료:")
    print(f"  - 프레임 크기: {frame.shape}")
    print(f"  - 감지 객체: {len(detections)}")
    
    # 깊이맵 통계
    valid_depths = depth_map[(depth_map > 0) & (depth_map < 100)]
    if len(valid_depths) > 0:
        print(f"  - 깊이 범위: {valid_depths.min():.2f}m ~ {valid_depths.max():.2f}m")
        print(f"  - 평균 깊이: {valid_depths.mean():.2f}m")
    
    # 시각화 정보
    print(f"\n📊 감지 정보:")
    for i, det in enumerate(detections[:5]):  # 처음 5개만
        bbox = det['bbox']
        print(f"  {i+1}. {det['class_name']} ({det['confidence']:.2f})")
        print(f"     - ID: {det['object_id']}")
        print(f"     - 위치: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
        print(f"     - 깊이: {det['depth_meters']:.2f}m")
    
    if len(detections) > 5:
        print(f"  ... 외 {len(detections)-5}개")
    
    print(f"\n💡 시각화 코드 예:")
    print("""
    # 깊이맵을 컬러맵으로 변환
    depth_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 영상과 합성
    result = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
    
    # 감지 영역 표시
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, det['class_name'], (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Camera with Depth', result)
    cv2.waitKey(0)
    """)


def main():
    """모든 예제 실행"""
    print("\n" + "="*60)
    print("🎥 실시간 D435i 카메라 기록 예제")
    print("="*60)
    
    example_1_basic_recording()
    example_2_load_and_inspect()
    example_3_depth_analysis()
    example_4_distance_based_filtering()
    example_5_frame_visualization()
    
    print("\n" + "="*60)
    print("✅ 모든 예제 완료!")
    print("="*60)
    print("""
    📚 다음 단계:
    1. REALTIME_CAMERA_GUIDE.md 문서 읽기
    2. realtime_camera_recorder.py로 데이터 기록
    3. video_data_loader.py로 데이터 분석
    4. 저장된 데이터로 테스트 수행
    """)


if __name__ == '__main__':
    main()
