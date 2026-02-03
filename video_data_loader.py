"""
기록된 테스트 데이터 로더
이전에 저장한 데이터를 로드하여 테스트 및 분석을 수행합니다.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class VideoDataLoader:
    """
    기록된 비디오/카메라 테스트 데이터 로드 및 제공
    """
    
    def __init__(self, session_path: str):
        """
        초기화
        
        Args:
            session_path (str): 저장된 세션 경로
        """
        self.session_path = Path(session_path)
        self.is_camera_data = False  # 카메라 데이터 여부
        
        if not self.session_path.exists():
            raise FileNotFoundError(f"세션 경로를 찾을 수 없습니다: {session_path}")
        
        # 메타데이터 로드
        metadata_path = self.session_path / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 감지 정보 로드
        detections_path = self.session_path / 'detections.json'
        with open(detections_path, 'r', encoding='utf-8') as f:
            self.detections = json.load(f)
        
        self.frames_dir = self.session_path / 'frames'
        self.depth_maps_dir = self.session_path / 'depth_maps'
        
        # 카메라 데이터인지 확인 (디렉토리 이름으로 판단)
        color_frames_dir = self.session_path / 'color_frames'
        if color_frames_dir.exists():
            self.is_camera_data = True
            self.frames_dir = color_frames_dir
            self.depth_frames_dir = self.session_path / 'depth_frames'
            self.depth_maps_dir = self.depth_frames_dir
        
        print(f"✓ 세션 로드 완료: {self.metadata['session_name']}")
        print(f"  - 프레임: {self.metadata['total_frames']}")
        print(f"  - 총 감지: {self.metadata['total_detections']}")
        print(f"  - 객체 타입: {list(self.metadata['class_distribution'].keys())}")
        print(f"  - 데이터 타입: {'카메라' if self.is_camera_data else '비디오'}")
    
    def get_metadata(self) -> Dict:
        """
        세션 메타데이터 반환
        
        Returns:
            dict: 메타데이터
        """
        return self.metadata
    
    def get_frame(self, frame_index: int) -> np.ndarray:
        """
        특정 프레임 로드
        
        Args:
            frame_index (int): 프레임 인덱스
            
        Returns:
            numpy.ndarray: 프레임 이미지 (BGR)
        """
        frame_filename = f"{frame_index:06d}.jpg"
        frame_path = self.frames_dir / frame_filename
        
        if not frame_path.exists():
            raise FileNotFoundError(f"프레임을 찾을 수 없습니다: {frame_filename}")
        
        return cv2.imread(str(frame_path))
    
    def get_depth_map(self, frame_index: int) -> np.ndarray:
        """
        특정 프레임의 깊이 맵 로드
        
        Args:
            frame_index (int): 프레임 인덱스
            
        Returns:
            numpy.ndarray: 깊이 맵 (0~1)
        """
        depth_filename = f"{frame_index:06d}.npy"
        depth_path = self.depth_maps_dir / depth_filename
        
        if not depth_path.exists():
            raise FileNotFoundError(f"깊이 맵을 찾을 수 없습니다: {depth_filename}")
        
        return np.load(str(depth_path))
    
    def get_detections_for_frame(self, frame_index: int) -> List[Dict]:
        """
        특정 프레임의 감지 결과 반환
        
        Args:
            frame_index (int): 프레임 인덱스
            
        Returns:
            list: 감지 정보 리스트
        """
        if frame_index >= len(self.detections):
            raise IndexError(f"프레임 인덱스 범위 초과: {frame_index}")
        
        return self.detections[frame_index]['detections']
    
    def get_frame_depth_meters(self, frame_index: int, bbox: List[float] = None) -> float:
        """
        프레임의 깊이값 반환 (카메라 데이터용)
        
        Args:
            frame_index (int): 프레임 인덱스
            bbox (list): 바운딩박스 [x1, y1, x2, y2] (선택)
            
        Returns:
            float: 깊이값 (미터) 또는 평균 깊이
        """
        if not self.is_camera_data:
            return None
        
        depth_map = self.get_depth_map(frame_index)
        
        if bbox is None:
            # 전체 깊이의 평균
            valid = depth_map[(depth_map > 0) & (depth_map < 100)]
            return float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            # ROI 깊이의 중앙값
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
            roi_depth = depth_map[y1:y2, x1:x2]
            valid = roi_depth[(roi_depth > 0) & (roi_depth < 100)]
            return float(np.median(valid)) if len(valid) > 0 else 0.0
    
    def get_all_frames_with_detections(self) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        모든 프레임과 감지 정보를 함께 반환 (제너레이터)
        
        Yields:
            tuple: (프레임 이미지, 감지 정보 리스트)
        """
        for frame_idx in range(self.metadata['total_frames']):
            frame = self.get_frame(frame_idx)
            detections = self.get_detections_for_frame(frame_idx)
            yield frame, detections
    
    def get_object_trajectory(self, object_id: int) -> List[Dict]:
        """
        특정 객체의 전체 궤적 반환
        
        Args:
            object_id (int): 객체 ID
            
        Returns:
            list: 각 프레임에서의 위치 정보
        """
        trajectory = []
        
        for frame_idx, frame_data in enumerate(self.detections):
            for detection in frame_data['detections']:
                if detection['object_id'] == object_id:
                    trajectory.append({
                        'frame_index': frame_idx,
                        'timestamp_ms': frame_data['timestamp_ms'],
                        'bbox': detection['bbox'],
                        'depth_value': detection['depth_value'],
                        'class_name': detection['class_name']
                    })
        
        return trajectory
    
    def get_statistics(self) -> Dict:
        """
        세션의 통계 정보 반환
        
        Returns:
            dict: 통계 정보
        """
        stats = {
            'total_frames': self.metadata['total_frames'],
            'total_detections': self.metadata['total_detections'],
            'duration_seconds': self.metadata['duration_seconds'],
            'fps': self.metadata['fps'],
            'class_distribution': self.metadata['class_distribution'],
            'average_detections_per_frame': (
                self.metadata['total_detections'] / max(1, self.metadata['total_frames'])
            ),
            'unique_objects': len(set(
                det['object_id'] 
                for frame_data in self.detections 
                for det in frame_data['detections']
                if det['object_id'] is not None
            ))
        }
        
        return stats
    
    def visualize_frame(self, frame_index: int, show_depth: bool = False) -> np.ndarray:
        """
        프레임을 감지 정보와 함께 시각화
        
        Args:
            frame_index (int): 프레임 인덱스
            show_depth (bool): 깊이 정보 표시 여부
            
        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        frame = self.get_frame(frame_index)
        detections = self.get_detections_for_frame(frame_index)
        
        # 감지 정보 그리기
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # 바운딩박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 레이블 그리기
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            if detection['object_id'] is not None:
                label = f"ID:{detection['object_id']} " + label
            
            label += f" (D:{detection['depth_value']:.2f})"
            
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 깊이 맵 오버레이 (선택)
        if show_depth:
            depth_map = self.get_depth_map(frame_index)
            depth_colored = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
        
        return frame


def demo_loader():
    """
    데이터 로더 데모
    """
    import glob
    
    # 최신 세션 찾기
    sessions = sorted(
        glob.glob('data/recorded_sessions/session_*'),
        reverse=True
    )
    
    if not sessions:
        print("저장된 세션이 없습니다.")
        print("먼저 video_data_recorder.py를 실행하여 데이터를 저장하세요.")
        return
    
    latest_session = sessions[0]
    print(f"\n최신 세션 로드: {latest_session}\n")
    
    loader = VideoDataLoader(latest_session)
    
    # 통계 출력
    stats = loader.get_statistics()
    print("📊 세션 통계:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n🎬 샘플 프레임 처리:")
    # 몇 개 프레임 샘플 처리
    for frame_idx in range(min(5, stats['total_frames'])):
        detections = loader.get_detections_for_frame(frame_idx)
        print(f"  프레임 {frame_idx}: {len(detections)}개 객체 감지")
        for det in detections:
            print(f"    - {det['class_name']} (ID:{det['object_id']}, 신뢰도:{det['confidence']:.2f})")
    
    print("\n🔍 객체 궤적 예제:")
    # 첫 번째 감지된 객체의 궤적 출력
    for frame_data in loader.detections:
        if frame_data['detections']:
            object_id = frame_data['detections'][0]['object_id']
            if object_id is not None:
                trajectory = loader.get_object_trajectory(object_id)
                print(f"  객체 ID {object_id} 궤적:")
                print(f"    - {len(trajectory)}개 프레임에서 감지됨")
                print(f"    - 시작: 프레임 {trajectory[0]['frame_index']}")
                print(f"    - 종료: 프레임 {trajectory[-1]['frame_index']}")
                break


if __name__ == '__main__':
    demo_loader()
