"""
비디오 파일로부터 테스트 데이터 기록
비디오를 처리하여 객체 감지, 깊이 정보, 프레임 등을 저장합니다.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import time
from object_detection_core import ObjectDetectionCore


class VideoDataRecorder:
    """
    비디오 파일을 처리하여 테스트용 데이터 기록
    
    저장되는 데이터:
    - frames/: 모든 프레임의 이미지 (jpg)
    - depth_maps/: 추정 깊이 맵 (npy)
    - detections.json: 모든 프레임의 객체 감지 결과
    - metadata.json: 비디오 정보 및 처리 결과
    """
    
    def __init__(self, output_dir='data/recorded_sessions', device='cpu'):
        """
        초기화
        
        Args:
            output_dir (str): 데이터 저장 디렉토리
            device (str): 실행 장치 ('cuda', 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 객체 감지 모델 초기화
        self.detector = ObjectDetectionCore(model_size='nano', device=device)
        
        # 타임스탬프 기반 세션 이름
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"session_{timestamp}"
        self.session_path = self.output_dir / self.session_name
        
        # 디렉토리 생성
        self.frames_dir = self.session_path / 'frames'
        self.depth_maps_dir = self.session_path / 'depth_maps'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.depth_maps_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장소
        self.detections = []
        self.metadata = {
            'session_name': self.session_name,
            'created_at': datetime.now().isoformat(),
            'device': device,
            'total_frames': 0,
            'width': 0,
            'height': 0,
            'fps': 0,
            'duration_seconds': 0,
            'total_detections': 0,
            'class_distribution': {}
        }
    
    def _estimate_depth_from_bbox(self, bbox, max_depth=5.0, min_depth=0.3):
        """
        바운딩박스 크기로부터 깊이값 추정
        
        Args:
            bbox (list): [x1, y1, x2, y2]
            max_depth (float): 최대 깊이 (미터)
            min_depth (float): 최소 깊이 (미터)
            
        Returns:
            float: 추정 깊이값 (0~1)
        """
        box_size = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2
        estimated_depth = max(min_depth, max_depth - (box_size / 100.0))
        return np.clip(estimated_depth / max_depth, 0.0, 1.0)
    
    def _generate_depth_map(self, frame, detections):
        """
        프레임으로부터 깊이 맵 생성
        객체의 깊이값을 활용하여 시뮬레이션 깊이 맵 생성
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            detections (list): 감지 결과
            
        Returns:
            numpy.ndarray: 깊이 맵 (0~1)
        """
        height, width = frame.shape[:2]
        depth_map = np.ones((height, width), dtype=np.float32) * 1.0  # 기본값: 멀리
        
        # 감지된 각 객체에 대해 깊이 정보 입력
        for detection in detections:
            bbox = detection['bbox']
            depth_value = detection.get('depth_value', 0.5)
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # 바운딩박스 영역에 깊이값 적용
            depth_map[y1:y2, x1:x2] = depth_value
        
        return depth_map
    
    def process_video(self, video_path, skip_frames=0):
        """
        비디오 파일 처리 및 데이터 기록
        
        Args:
            video_path (str): 비디오 파일 경로
            skip_frames (int): 건너뛸 프레임 수 (0=모든 프레임 처리)
            
        Returns:
            str: 저장된 세션 경로
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        print(f"🎥 비디오 처리 시작: {video_path.name}")
        print(f"📁 저장 경로: {self.session_path}")
        
        # 비디오 열기
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
        
        # 비디오 정보 획득
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.metadata['width'] = width
        self.metadata['height'] = height
        self.metadata['fps'] = fps
        self.metadata['duration_seconds'] = total_frames / fps
        self.metadata['source_video'] = str(video_path.name)
        
        print(f"  - 해상도: {width}x{height}")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - 총 프레임: {total_frames}")
        print(f"  - 예상 시간: {total_frames/fps:.1f}초")
        print(f"  - 건너뛸 프레임: {skip_frames}")
        
        frame_idx = 0
        saved_frames = 0
        start_time = time.time()
        
        # 프레임 처리 루프
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 건너뛰기 처리
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            # 객체 감지
            try:
                annotated, detections = self.detector.detect(frame, enable_tracking=True)
            except Exception as e:
                print(f"  ⚠ 프레임 {frame_idx} 처리 오류: {e}")
                frame_idx += 1
                continue
            
            # 깊이값 추정
            for detection in detections:
                detection['depth_value'] = self._estimate_depth_from_bbox(detection['bbox'])
            
            # 프레임 저장
            frame_filename = f"{saved_frames:06d}.jpg"
            frame_path = self.frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # 깊이 맵 생성 및 저장
            depth_map = self._generate_depth_map(frame, detections)
            depth_filename = f"{saved_frames:06d}.npy"
            depth_path = self.depth_maps_dir / depth_filename
            np.save(str(depth_path), depth_map)
            
            # 프레임 정보 저장
            frame_info = {
                'frame_index': saved_frames,
                'timestamp_ms': (frame_idx / fps) * 1000,
                'frame_filename': frame_filename,
                'depth_filename': depth_filename,
                'detections': []
            }
            
            # 감지 결과 저장
            for detection in detections:
                det_info = {
                    'class_name': detection['class_name'],
                    'class_id': detection['class_id'],
                    'confidence': float(detection['confidence']),
                    'object_id': detection['object_id'],
                    'bbox': [float(x) for x in detection['bbox']],
                    'depth_value': float(detection['depth_value'])
                }
                frame_info['detections'].append(det_info)
                
                # 클래스 분포 업데이트
                class_name = detection['class_name']
                if class_name not in self.metadata['class_distribution']:
                    self.metadata['class_distribution'][class_name] = 0
                self.metadata['class_distribution'][class_name] += 1
                
                self.metadata['total_detections'] += 1
            
            self.detections.append(frame_info)
            saved_frames += 1
            
            # 진행률 출력
            if saved_frames % max(1, total_frames // 20) == 0:
                elapsed = time.time() - start_time
                print(f"  ✓ {saved_frames} 프레임 저장 ({elapsed:.1f}초)")
            
            frame_idx += 1
        
        cap.release()
        
        # 메타데이터 업데이트
        self.metadata['total_frames'] = saved_frames
        elapsed_time = time.time() - start_time
        self.metadata['processing_time_seconds'] = elapsed_time
        
        # 데이터 저장
        self._save_data()
        
        print(f"\n✅ 처리 완료!")
        print(f"  - 저장된 프레임: {saved_frames}")
        print(f"  - 총 감지: {self.metadata['total_detections']}")
        print(f"  - 처리 시간: {elapsed_time:.1f}초")
        print(f"  - 저장 경로: {self.session_path}")
        
        return str(self.session_path)
    
    def _save_data(self):
        """
        수집된 데이터를 JSON 파일로 저장
        """
        # detections.json 저장
        detections_path = self.session_path / 'detections.json'
        with open(detections_path, 'w', encoding='utf-8') as f:
            json.dump(self.detections, f, indent=2, ensure_ascii=False)
        
        # metadata.json 저장
        metadata_path = self.session_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  📄 저장된 파일:")
        print(f"    - {detections_path.name}")
        print(f"    - {metadata_path.name}")
        print(f"    - {self.frames_dir.name}/ ({self.metadata['total_frames']} 프레임)")
        print(f"    - {self.depth_maps_dir.name}/ ({self.metadata['total_frames']} 깊이맵)")


def main():
    """
    테스트 실행 예제
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='비디오 파일로부터 테스트 데이터 기록')
    parser.add_argument('video_path', help='입력 비디오 파일 경로')
    parser.add_argument('--output-dir', default='data/recorded_sessions', 
                       help='출력 디렉토리 (기본값: data/recorded_sessions)')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='건너뛸 프레임 수 (0=모든 프레임 처리)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='실행 장치 (기본값: cpu)')
    
    args = parser.parse_args()
    
    # 레코더 초기화
    recorder = VideoDataRecorder(output_dir=args.output_dir, device=args.device)
    
    # 비디오 처리
    try:
        session_path = recorder.process_video(
            video_path=args.video_path,
            skip_frames=args.skip_frames
        )
        print(f"\n🎉 데이터 저장 성공!")
        print(f"저장 경로: {session_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
