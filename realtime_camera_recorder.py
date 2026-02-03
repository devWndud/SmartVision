"""
실시간 D435i 깊이 카메라 데이터 저장
카메라로 촬영되는 RGB, 깊이, 객체 감지 정보를 저장합니다.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import time
import threading

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    print("⚠️  pyrealsense2가 설치되지 않았습니다.")
    print("설치: pip install pyrealsense2")
    HAS_REALSENSE = False

from object_detection_core import ObjectDetectionCore


class RealtimeCameraRecorder:
    """
    D435i 카메라의 실시간 데이터 기록
    
    저장되는 데이터:
    - color_frames/: 컬러 영상 (jpg)
    - depth_frames/: 깊이 데이터 (npy - 미터 단위)
    - detections.json: 객체 감지 결과
    - metadata.json: 기록 정보 및 통계
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
        print("🔄 객체 감지 모델 로드 중...")
        self.detector = ObjectDetectionCore(model_size='nano', device=device)
        
        # 타임스탬프 기반 세션 이름
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"camera_{timestamp}"
        self.session_path = self.output_dir / self.session_name
        
        # 디렉토리 생성
        self.color_frames_dir = self.session_path / 'color_frames'
        self.depth_frames_dir = self.session_path / 'depth_frames'
        self.color_frames_dir.mkdir(parents=True, exist_ok=True)
        self.depth_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # 카메라 초기화
        self.pipeline = None
        self.depth_scale = None
        self._init_camera()
        
        # 데이터 저장소
        self.detections = []
        self.frame_count = 0
        self.is_recording = False
        self.metadata = {
            'session_name': self.session_name,
            'created_at': datetime.now().isoformat(),
            'device': device,
            'camera_model': 'D435i',
            'total_frames': 0,
            'total_detections': 0,
            'class_distribution': {},
            'recording_duration_seconds': 0,
            'average_fps': 0
        }
    
    def _init_camera(self):
        """D435i 카메라 초기화"""
        if not HAS_REALSENSE:
            print("❌ pyrealsense2가 설치되지 않았습니다.")
            print("설치 방법: pip install pyrealsense2")
            return False
        
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 스트림 설정
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            print("🎥 D435i 카메라 시작 중...")
            profile = self.pipeline.start(config)
            
            # Depth scale 획득
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            print(f"✓ D435i 카메라 초기화 완료")
            print(f"  - 해상도: 640x480")
            print(f"  - FPS: 30")
            print(f"  - Depth Scale: {self.depth_scale}")
            
            return True
        except Exception as e:
            print(f"❌ 카메라 초기화 실패: {e}")
            print("D435i 카메라가 연결되어 있는지 확인하세요.")
            return False
    
    def start_recording(self, duration_seconds=None):
        """
        실시간 기록 시작
        
        Args:
            duration_seconds (int): 기록 시간 (None=무한대, Ctrl+C로 중단)
        """
        if self.pipeline is None:
            print("❌ 카메라가 초기화되지 않았습니다.")
            return False
        
        self.is_recording = True
        self.frame_count = 0
        start_time = time.time()
        
        print(f"\n🔴 기록 시작!")
        print(f"📁 저장 경로: {self.session_path}")
        if duration_seconds:
            print(f"⏱️  기록 시간: {duration_seconds}초")
        else:
            print(f"⏱️  Ctrl+C를 눌러 기록 중단")
        
        try:
            while self.is_recording:
                # 프레임 획득
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # numpy 배열로 변환
                color_image = np.asanyarray(color_frame.get_data())
                depth_array = np.asanyarray(depth_frame.get_data())
                
                # 깊이값을 미터 단위로 변환
                depth_meters = depth_array * self.depth_scale
                
                # 객체 감지
                try:
                    annotated, detections = self.detector.detect(color_image, enable_tracking=True)
                except Exception as e:
                    print(f"  ⚠️  프레임 {self.frame_count} 감지 오류: {e}")
                    detections = []
                
                # 깊이값 추가 (바운딩박스 영역의 깊이)
                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(640, x2), min(480, y2)
                    
                    # ROI 깊이값 추출 (중앙값)
                    roi_depth = depth_meters[y1:y2, x1:x2]
                    valid_depths = roi_depth[(roi_depth > 0) & (roi_depth < 10)]
                    
                    if len(valid_depths) > 0:
                        detection['depth_meters'] = float(np.median(valid_depths))
                    else:
                        detection['depth_meters'] = 0.0
                
                # 데이터 저장
                self._save_frame(color_image, depth_meters, detections)
                
                # 진행률 출력
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"  ✓ {self.frame_count} 프레임 저장 ({fps:.1f} FPS, {elapsed:.1f}초)")
                
                # 시간 제한 확인
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    self.is_recording = False
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⏹️  Ctrl+C로 인해 기록 중단")
        except Exception as e:
            print(f"❌ 기록 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._finalize_recording(time.time() - start_time)
        
        return True
    
    def _save_frame(self, color_image, depth_array, detections):
        """
        프레임 및 감지 정보 저장
        
        Args:
            color_image (numpy.ndarray): 컬러 영상
            depth_array (numpy.ndarray): 깊이 배열 (미터 단위)
            detections (list): 감지 결과
        """
        # 컬러 프레임 저장
        color_filename = f"{self.frame_count:06d}.jpg"
        color_path = self.color_frames_dir / color_filename
        cv2.imwrite(str(color_path), color_image)
        
        # 깊이 프레임 저장
        depth_filename = f"{self.frame_count:06d}.npy"
        depth_path = self.depth_frames_dir / depth_filename
        np.save(str(depth_path), depth_array)
        
        # 프레임 정보 저장
        frame_info = {
            'frame_index': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'color_filename': color_filename,
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
                'depth_meters': detection.get('depth_meters', 0.0)
            }
            frame_info['detections'].append(det_info)
            
            # 클래스 분포 업데이트
            class_name = detection['class_name']
            if class_name not in self.metadata['class_distribution']:
                self.metadata['class_distribution'][class_name] = 0
            self.metadata['class_distribution'][class_name] += 1
            
            self.metadata['total_detections'] += 1
        
        self.detections.append(frame_info)
    
    def _finalize_recording(self, elapsed_time):
        """기록 완료 및 데이터 저장"""
        print(f"\n✅ 기록 완료!")
        print(f"  - 저장된 프레임: {self.frame_count}")
        print(f"  - 총 감지: {self.metadata['total_detections']}")
        print(f"  - 기록 시간: {elapsed_time:.1f}초")
        print(f"  - 평균 FPS: {self.frame_count/elapsed_time if elapsed_time > 0 else 0:.1f}")
        
        # 메타데이터 업데이트
        self.metadata['total_frames'] = self.frame_count
        self.metadata['recording_duration_seconds'] = elapsed_time
        self.metadata['average_fps'] = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # 데이터 저장
        self._save_metadata()
        
        print(f"\n📁 저장 경로: {self.session_path}")
        print(f"  - color_frames/: {self.frame_count}개 프레임")
        print(f"  - depth_frames/: {self.frame_count}개 깊이맵")
        print(f"  - detections.json: 감지 정보")
        print(f"  - metadata.json: 메타데이터")
    
    def _save_metadata(self):
        """메타데이터 저장"""
        # detections.json 저장
        detections_path = self.session_path / 'detections.json'
        with open(detections_path, 'w', encoding='utf-8') as f:
            json.dump(self.detections, f, indent=2, ensure_ascii=False)
        
        # metadata.json 저장
        metadata_path = self.session_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def stop_recording(self):
        """기록 중단"""
        self.is_recording = False
        print("\n⏹️  기록 중단 신호 전송...")
    
    def cleanup(self):
        """카메라 및 리소스 정리"""
        if self.pipeline:
            self.pipeline.stop()
            print("✓ 카메라 종료")


def main():
    """
    테스트 실행 예제
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='D435i 카메라 실시간 데이터 저장')
    parser.add_argument('--duration', type=int, default=None,
                       help='기록 시간 (초) - 기본값: Ctrl+C로 중단')
    parser.add_argument('--output-dir', default='data/recorded_sessions',
                       help='출력 디렉토리 (기본값: data/recorded_sessions)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='실행 장치 (기본값: cpu)')
    
    args = parser.parse_args()
    
    # 레코더 초기화
    recorder = RealtimeCameraRecorder(
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 기록 시작
    try:
        recorder.start_recording(duration_seconds=args.duration)
    finally:
        recorder.cleanup()


if __name__ == '__main__':
    main()
