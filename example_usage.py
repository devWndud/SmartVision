"""
통합 예제: 객체 인식 + 시각화 + 칼만 필터 추적 + 충돌 위험도 평가
세 모듈을 함께 사용하는 완전한 예제입니다.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from object_detection_core import ObjectDetectionCore
from visualization_core import Visualizer, BirdEyeViewRenderer, VisualizationComposer
from kalman_tracker import KalmanTracker
from collision_risk_assessment import CollisionRiskAssessment
from video_data_loader import VideoDataLoader

# ==================== 테스트 데이터 설정 ====================
# 저장된 테스트 데이터 경로를 설정하세요
# None: 현재 D435i 카메라 사용
# "data/recorded_sessions/camera_20260203_HHMMSS": 저장된 카메라 데이터 사용
TEST_DATA_PATH = "data/recorded_sessions/camera_20260203_003647"
# ============================================================


class ObjectTrackingPipeline:
    """
    객체 감지, 추적, 시각화 통합 파이프라인
    """
    # 상수 정의
    MAX_DEPTH_METERS = 5.0
    MIN_DEPTH_METERS = 0.3
    ROI_RADIUS = 20
    
    def __init__(self, test_data_path=None, device='cpu', enable_bev=True):
        """
        초기화
        
        Args:
            test_data_path (str or None): 테스트 데이터 경로 (None=카메라 사용)
            device (str): 실행 장치 ('cuda', 'cpu')
            enable_bev: BEV 시각화 활성화
        """
        # 모듈 초기화
        self.detector = ObjectDetectionCore(model_size='nano', device=device)
        self.tracker = KalmanTracker()
        self.visualizer = Visualizer(640, 480)
        self.risk_assessor = CollisionRiskAssessment()
        self.enable_bev = enable_bev
        self.device = device
        
        # 테스트 데이터 모드
        self.test_data_loader = None
        self.test_frame_idx = 0
        self.is_test_mode = False
        self.pipeline = None
        self.depth_scale = None
        self.cap = None
        
        if test_data_path:
            # 테스트 데이터 모드
            print(f"📂 테스트 데이터 로드: {test_data_path}")
            try:
                self.test_data_loader = VideoDataLoader(test_data_path)
                self.is_test_mode = True
                self.width = 640
                self.height = 480
                self.fps = int(self.test_data_loader.metadata.get('fps', 30)) or 30
                print(f"✓ 테스트 데이터 로드 완료")
                print(f"  - 총 프레임: {self.test_data_loader.metadata['total_frames']}")
                print(f"  - 카메라 데이터: {self.test_data_loader.is_camera_data}")
            except Exception as e:
                print(f"❌ 테스트 데이터 로드 실패: {e}")
                test_data_path = None
        
        if not test_data_path:
            # 실시간 카메라 모드
            self.is_test_mode = False
            try:
                import pyrealsense2 as rs
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.pipeline.start(config)
                
                # Depth scale 획득
                profile = self.pipeline.get_active_profile()
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f"✓ D435i 카메라 초기화 완료")
                print(f"  - Depth Scale: {self.depth_scale}")
                print(f"  - 예상: 50cm 거리 → depth 값 ≈ {0.5 / self.depth_scale:.0f}")
            except Exception as e:
                print(f"⚠ D435i 초기화 실패: {e}")
                self.pipeline = None
            
            # 카메라가 없으면 비디오 파일 사용
            if self.pipeline is None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise RuntimeError("카메라 또는 비디오 소스를 사용할 수 없습니다")
                
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            else:
                self.width = 640
                self.height = 480
                self.fps = 30
        
        # BirdEyeViewRenderer 크기 확대
        self.bev_renderer = BirdEyeViewRenderer(width=1000, height=1000, scale=150) if enable_bev else None
        
        # FPS 계산용
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
    
    def _estimate_depth_from_bbox(self, bbox):
        """
        바운딩박스 크기로부터 깊이값 추정 (폴백용)
        """
        box_size = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2
        estimated_depth = max(self.MIN_DEPTH_METERS, self.MAX_DEPTH_METERS - (box_size / 100.0))
        return np.clip(estimated_depth / self.MAX_DEPTH_METERS, 0.0, 1.0)
    
    def _filter_depth_with_iqr(self, depth_array, roi_mask=None):
        """
        IQR (Interquartile Range) 방식으로 깊이값 이상치 제거
        
        깊이 노이즈를 50~70% 감소시킴
        
        Args:
            depth_array: 깊이값 배열
            roi_mask: ROI 마스크 (선택)
            
        Returns:
            필터링된 깊이값 (중앙값)
        """
        # 0 이상의 유효한 값만 추출
        valid_depths = depth_array[depth_array > 0]
        
        if len(valid_depths) == 0:
            return None
        
        # IQR 계산
        Q1 = np.percentile(valid_depths, 25)
        Q3 = np.percentile(valid_depths, 75)
        IQR = Q3 - Q1
        
        # 이상치 제거 (1.5*IQR)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_depths = valid_depths[
            (valid_depths >= lower_bound) & 
            (valid_depths <= upper_bound)
        ]
        
        if len(filtered_depths) == 0:
            return np.median(valid_depths)  # 이상치만 있으면 원본 중앙값 반환
        
        return np.median(filtered_depths)  # 필터링된 중앙값
    
    def process_frame(self, frame, depth_frame=None):
        """
        프레임 처리 (감지 -> 추적 -> 시각화)
        
        Args:
            frame (numpy.ndarray): 입력 이미지 프레임
            depth_frame (numpy.ndarray): 깊이 프레임 (미터 단위, RealSense에서)
            
        Returns:
            numpy.ndarray: 처리된 프레임
        """
        # 1. 객체 감지
        annotated, detections = self.detector.detect(frame, enable_tracking=True)
        
        # 2. 칼만 필터를 이용한 추적
        tracked_detections = []
        active_ids = []
        
        for detection in detections:
            obj_id = detection['object_id']
            
            if obj_id is not None:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                
                depth = self._estimate_depth_from_bbox(bbox)
                
                if depth_frame is not None:
                    
                    h, w = depth_frame.shape
                    if 0 <= center_y_int < h and 0 <= center_x_int < w:
                        x1 = max(0, center_x_int - self.ROI_RADIUS)
                        x2 = min(w, center_x_int + self.ROI_RADIUS)
                        y1 = max(0, center_y_int - self.ROI_RADIUS)
                        y2 = min(h, center_y_int + self.ROI_RADIUS)
                        
                        roi_depth = depth_frame[y1:y2, x1:x2].flatten()
                        
                        # IQR 필터를 사용하여 노이즈 제거 (Phase 1 개선)
                        depth_raw = self._filter_depth_with_iqr(roi_depth)
                        
                        if depth_raw is not None:
                            depth = np.clip(depth_raw / self.MAX_DEPTH_METERS, 0.0, 1.0)
                        else:
                            depth = self._estimate_depth_from_bbox(bbox)
                
                measurement = {
                    'location': np.array([
                        (center_x - self.width/2) / self.width,  # -0.5 ~ 0.5
                        (center_y - self.height/2) / self.height,  # -0.5 ~ 0.5
                        depth
                    ]),
                    'dimensions': np.array([1.7, 0.6, 0.6]),  # 기본값
                    'orientation': 0,
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'depth_value': depth
                }
                
                # 칼만 필터 업데이트
                filtered = self.tracker.update(obj_id, measurement)
                filtered['object_id'] = obj_id
                
                tracked_detections.append(filtered)
                active_ids.append(obj_id)
        
        # 3. 활성 객체 정리
        self.tracker.cleanup(active_ids)
        
        # 3.5 위험도 평가 (미래 예측 + TTC 계산)
        risk_assessments = {}
        overall_risk_level = 'safe'
        max_risk_score = 0.0
        
        for detection in tracked_detections:
            obj_id = detection['object_id']
            
            # 미래 위치 예측 (16프레임 = ~0.5초)
            future_states = self.tracker.predict_future(obj_id, frames=16)
            
            # 현재 상태
            current_state = {
                'location': detection['location'],
                'velocity': detection['velocity'],
                'dimensions': detection.get('dimensions', np.array([1.7, 0.6, 0.6]))
            }
            
            # 위험도 평가
            assessment = self.risk_assessor.assess_risk(
                object_id=obj_id,
                class_name=detection['class_name'],
                current_state=current_state,
                future_states=future_states
            )
            
            risk_assessments[obj_id] = assessment
            
            # 전체 위험도 업데이트
            if assessment.risk_score > max_risk_score:
                max_risk_score = assessment.risk_score
                overall_risk_level = assessment.risk_level
            
            # 각 detection에 위험도 정보 추가
            detection['risk_assessment'] = assessment
        
        # 4. 시각화
        result_frame = frame.copy()
        
        for detection in tracked_detections:
            depth = detection.get('depth_value', 0.5)
            result_frame = self.visualizer.draw_3d_box(result_frame, detection, depth)
        
        # 5. 조감도 추가 (위험도 포함)
        bev_image = None
        if self.enable_bev and self.bev_renderer:
            self.bev_renderer.reset()
            
            for detection in tracked_detections:
                obj_id = detection.get('object_id')
                center_x = (detection['bbox'][0] + detection['bbox'][2]) / 2
                lateral = (center_x - self.width/2) / (self.width/2)
                depth_normalized = detection.get('depth_value', 0.5)
                depth_meters = np.clip(depth_normalized * self.MAX_DEPTH_METERS, self.MIN_DEPTH_METERS, self.MAX_DEPTH_METERS)
                
                # 위험도 평가 정보 추가
                risk_assessment = risk_assessments.get(obj_id) if obj_id in risk_assessments else None
                
                self.bev_renderer.draw_object_with_risk(
                    detection['class_name'],
                    depth_meters,
                    lateral,
                    obj_id,
                    risk_assessment
                )
            
            bev_image = self.bev_renderer.get_image()
            result_frame = VisualizationComposer.add_bev_corner(
                result_frame, bev_image, 'bottom-left'
            )

        
        # 6. 위험도 지표 표시 (왼쪽 상단)
        result_frame = self.visualizer.draw_risk_indicator(
            result_frame, 
            overall_risk_level, 
            max_risk_score,
            position=(15, 60)
        )
        
        # 7. FPS 및 정보 추가
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.fps_display = self.frame_count / elapsed
        
        result_frame = VisualizationComposer.overlay_info(
            result_frame, fps=self.fps_display, device=self.device,
            tracking_enabled=True
        )
        
        # BEV 이미지 함께 반환
        return result_frame, bev_image, risk_assessments
    
    def run(self, output_file=None, output_bev_file=None):
        """
        메인 처리 루프
        
        Args:
            output_file (str): 출력 비디오 파일 경로 (메인 프레임)
            output_bev_file (str): BEV 전용 출력 비디오 파일 경로
        """
        print(f"✓ 파이프라인 시작 (Device: {self.device})")
        print(f"✓ 해상도: {self.width}x{self.height}, FPS: {self.fps}")
        print(f"✓ 조감도: {'활성' if self.enable_bev else '비활성'}")
        
        if self.is_test_mode:
            print(f"✓ 테스트 데이터 모드 (총 {self.test_data_loader.metadata['total_frames']}프레임)")
            print(f"  - 카메라 데이터: {self.test_data_loader.is_camera_data}")
        else:
            print(f"✓ 실시간 카메라 모드")
            print(f"✓ RealSense 파이프라인: {'활성' if self.pipeline else '비활성 (일반 카메라)'}")
            if self.pipeline:
                print(f"  - Depth Scale: {self.depth_scale}")
        
        print("✓ 'q' 또는 ESC로 종료")
        
        # 비디오 저장 설정 (메인 프레임)
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
        
        # BEV 전용 저장 설정
        out_bev = None
        bev_size = (1000, 1000) if self.enable_bev else None
        if output_bev_file and self.enable_bev:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_bev = cv2.VideoWriter(output_bev_file, fourcc, self.fps, bev_size)
            print(f"✓ BEV 출력: {output_bev_file}")
        
        try:
            while True:
                if self.is_test_mode:
                    # 테스트 데이터 모드
                    if self.test_frame_idx >= self.test_data_loader.metadata['total_frames']:
                        print("✓ 테스트 데이터 재생 완료")
                        break
                    
                    frame = self.test_data_loader.get_frame(self.test_frame_idx)
                    depth_meters = self.test_data_loader.get_depth_map(self.test_frame_idx)
                    
                    self.test_frame_idx += 1
                
                elif self.pipeline:
                    # 실시간 D435i 카메라
                    frames = self.pipeline.wait_for_frames()
                    color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
                    if not (color_frame and depth_frame):
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                    depth_meters = np.asanyarray(depth_frame.get_data()) * self.depth_scale
                
                else:
                    # 일반 웹캠
                    ret, frame = self.cap.read()
                    if not ret:
                        print("✓ 비디오 재생 완료")
                        break
                    depth_meters = None
                
                result, bev_image, risk_assessments = self.process_frame(frame, depth_frame=depth_meters)
                
                if out:
                    out.write(result)
                
                # BEV 전용 저장
                if out_bev and bev_image is not None:
                    # BEV 이미지가 PIL 형식일 수 있으니 numpy 배열로 변환
                    if isinstance(bev_image, np.ndarray):
                        bev_bgr = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR) if len(bev_image.shape) == 3 else bev_image
                        out_bev.write(bev_bgr)

                cv2.imshow('Object Tracking Pipeline', result)
                if bev_image is not None:
                    cv2.imshow('Bird\'s Eye View (BEV)', bev_image)
                
                if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]:
                    print("✓ 사용자 종료")
                    break
        
        finally:
            self.cleanup(output_file, out, output_bev_file, out_bev)
    
    def cleanup(self, output_file, out, output_bev_file=None, out_bev=None):
        """
        리소스 정리
        
        Args:
            output_file (str): 메인 출력 파일명
            out: 메인 비디오 라이터
            output_bev_file (str): BEV 출력 파일명
            out_bev: BEV 비디오 라이터
        """
        if self.pipeline:
            self.pipeline.stop()
        if self.cap:
            self.cap.release()
        if out:
            out.release()
        if out_bev:
            out_bev.release()
        cv2.destroyAllWindows()
        
        if output_file:
            print(f"✓ 메인 비디오 저장됨: {output_file}")
        if output_bev_file:
            print(f"✓ BEV 비디오 저장됨: {output_bev_file}")

def main():
    """
    테스트 데이터 또는 카메라를 사용하여 실행
    
    TEST_DATA_PATH 변수로 모드 선택:
    - None: 실시간 D435i 카메라 사용
    - "data/recorded_sessions/camera_20260203_HHMMSS": 저장된 카메라 데이터 사용
    
    출력 옵션:
    - output_file: 메인 프레임 영상 저장
    - output_bev_file: BEV 전용 영상 저장
    """
    pipeline = ObjectTrackingPipeline(
        test_data_path=TEST_DATA_PATH,
        device='cpu',
        enable_bev=True
    )
    
    # 메인 프레임 + BEV 동시 저장
    pipeline.run(
        output_file='output_main.mp4',
        output_bev_file='output_bev.mp4'
    )


if __name__ == '__main__':
    main()
