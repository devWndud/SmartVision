"""
빠른 시작 가이드 (Quick Start)
"""

# ============================================================================
# 1. 객체 감지 모듈 (object_detection_core.py)
# ============================================================================

from object_detection_core import ObjectDetectionCore
import cv2

# 초기화
detector = ObjectDetectionCore(model_size='nano', device='cuda')

# 사용
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 객체 감지 (추적 활성화)
    annotated_image, detections = detector.detect(frame, enable_tracking=True)
    
    # detections는 다음과 같은 구조:
    # [
    #   {
    #       'bbox': [x1, y1, x2, y2],
    #       'confidence': 0.95,
    #       'class_id': 2,
    #       'object_id': 1,  # 추적 ID
    #       'class_name': 'car'
    #   },
    #   ...
    # ]
    
    cv2.imshow('Detection', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ============================================================================
# 2. 시각화 모듈 (visualization_core.py)
# ============================================================================

from visualization_core import Visualizer, BirdEyeViewRenderer, VisualizationComposer

# 2-1. 3D 박스 시각화
visualizer = Visualizer(image_width=640, image_height=480)

# detection 예제
detection = {
    'bbox': [100, 100, 200, 250],
    'class_name': 'car',
    'confidence': 0.95,
    'object_id': 1
}

result_image = visualizer.draw_3d_box(
    image=frame,
    detection=detection,
    depth_value=0.5,  # 0~1 범위
    color=(0, 0, 255)  # BGR
)

# 2-2. Bird's Eye View (조감도)
bev = BirdEyeViewRenderer(width=400, height=400, scale=30)

# 조감도 초기화
bev.reset()

# 객체 그리기
bev.draw_object(
    class_name='car',
    depth=2.5,              # 미터 단위
    lateral_offset=0.2,     # -1~1 범위
    object_id=1
)

bev_image = bev.get_image()

# 2-3. 통합 시각화
result = VisualizationComposer.overlay_info(
    image=result_image,
    fps=30.5,
    device='cuda',
    tracking_enabled=True
)

# 조감도를 메인 이미지에 추가
result = VisualizationComposer.add_bev_corner(
    main_image=result,
    bev_image=bev_image,
    position='bottom-left'  # top-left, top-right, bottom-left, bottom-right
)


# ============================================================================
# 3. 칼만 필터 추적 모듈 (kalman_tracker.py)
# ============================================================================

from kalman_tracker import KalmanTracker
import numpy as np

tracker = KalmanTracker(
    max_history=5,
    process_noise_scale=0.1,
    measurement_noise_scale=0.5
)

# 추적할 객체 ID
object_id = 1

# 측정값 (센서 또는 감지기로부터)
measurement = {
    'location': np.array([0.1, 0.2, 2.5]),      # [x, y, z] 미터
    'dimensions': np.array([1.7, 0.6, 0.6]),    # [height, width, length] 미터
    'orientation': 0.5                           # 라디안
}

# 1. 업데이트 (측정값으로 필터 업데이트)
filtered_state = tracker.update(object_id, measurement)
# 반환값:
# {
#     'location': np.array([...]),
#     'dimensions': np.array([...]),
#     'orientation': float,
#     'velocity': np.array([vx, vy, vz]),
#     'angular_velocity': float,
#     ... (원본 측정값도 포함)
# }

# 2. 예측 (다음 프레임 위치 예측)
predicted = tracker.predict(object_id)
# {
#     'location': np.array([...]),
#     'dimensions': np.array([...]),
#     'orientation': float
# }

# 3. 평활화 (시간적 평활화 적용)
smoothed = tracker.get_smoothed_state(object_id)
# 이전 프레임들의 데이터를 이용하여 부드러운 움직임 표현

# 4. 통계 (속도, 가속도 등)
stats = tracker.get_statistics(object_id)
# {
#     'velocity': 2.5,                           # m/s
#     'velocity_direction': np.array([...]),     # 단위 벡터
#     'movement_speed': 1.8,                     # m/s
#     'angular_velocity': 0.1,                   # rad/s
#     'frames_tracked': 15                       # 프레임 수
# }

# 5. 정리 (더 이상 추적하지 않는 객체 제거)
active_ids = [1, 2, 3]  # 현재 활성 객체 ID
tracker.cleanup(active_ids)


# ============================================================================
# 4. 완전한 통합 예제
# ============================================================================

import cv2
import numpy as np
from object_detection_core import ObjectDetectionCore
from visualization_core import Visualizer, BirdEyeViewRenderer, VisualizationComposer
from kalman_tracker import KalmanTracker

class SimpleTracker:
    def __init__(self):
        self.detector = ObjectDetectionCore(device='cuda')
        self.visualizer = Visualizer(640, 480)
        self.bev = BirdEyeViewRenderer()
        self.tracker = KalmanTracker()
    
    def process_frame(self, frame):
        # 1. 감지
        annotated, detections = self.detector.detect(frame)
        
        # 2. 추적
        active_ids = []
        for det in detections:
            if det['object_id']:
                # 깊이 추정 (박스 크기로부터)
                bbox = det['bbox']
                box_size = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2
                depth = 1.0 + (500 - box_size) / 500
                depth = max(0.1, min(depth, 1.0))
                
                # 칼만 필터 업데이트
                self.tracker.update(det['object_id'], {
                    'location': np.array([0, 0, depth]),
                    'dimensions': np.array([1.7, 0.6, 0.6]),
                    'orientation': 0
                })
                
                active_ids.append(det['object_id'])
        
        # 정리
        self.tracker.cleanup(active_ids)
        
        # 3. 시각화
        result = frame.copy()
        
        # BEV 그리기
        self.bev.reset()
        for det in detections:
            depth = (500 - ((det['bbox'][2] - det['bbox'][0]) + 
                           (det['bbox'][3] - det['bbox'][1])) / 2) / 500
            self.bev.draw_object(det['class_name'], max(0.1, min(depth * 4 + 1, 5)), 
                                object_id=det['object_id'])
        
        # 메인 이미지에 합성
        result = VisualizationComposer.add_bev_corner(result, self.bev.get_image())
        
        # 정보 오버레이
        result = VisualizationComposer.overlay_info(result, fps=30.5, device='cuda')
        
        return result

# 사용
tracker = SimpleTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = tracker.process_frame(frame)
    cv2.imshow('Simple Tracker', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ============================================================================
# 5. 모듈별 핵심 포인트
# ============================================================================

"""
객체 감지 (object_detection_core.py):
- detect()는 항상 (annotated_image, detections) 튜플 반환
- object_id는 추적이 활성화되었을 때만 할당됨
- class_name은 YOLO 클래스명

시각화 (visualization_core.py):
- Visualizer.draw_3d_box(): 단일 객체용
- BirdEyeViewRenderer: 공간적 관계 표현
- VisualizationComposer: 정보 오버레이 및 합성

칼만 필터 (kalman_tracker.py):
- 각 object_id마다 별도의 필터 유지
- update() 후 get_smoothed_state()로 평활화된 값 조회
- cleanup()으로 메모리 관리
- get_statistics()로 속도, 가속도 등 추적 정보 조회
"""

