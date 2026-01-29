# 객체 감지 + 시각화 + 칼만 필터 추적 모듈

세 가지 핵심 모듈로 구성된 객체 추적 시스템입니다.

## 📦 모듈 구성

### 1. **object_detection_core.py** - 객체 감지
- **클래스**: `ObjectDetectionCore`
- **기능**: 
  - YOLO v11 기반 실시간 객체 감지
  - 다중 객체 추적 (Tracking)
  - 궤적(Trajectory) 기록
  
**주요 메서드**:
```python
detector = ObjectDetectionCore(model_size='nano', device='cuda')
annotated_image, detections = detector.detect(frame, enable_tracking=True)
```

**반환 형식**:
```python
detections = [
    {
        'bbox': [x1, y1, x2, y2],
        'confidence': float,
        'class_id': int,
        'object_id': int,  # 추적 ID (None if no tracking)
        'class_name': str
    },
    ...
]
```

---

### 2. **visualization_core.py** - 시각화

#### 2.1 **Visualizer** - 3D 박스 시각화
```python
visualizer = Visualizer(image_width=640, image_height=480)
result = visualizer.draw_3d_box(image, detection, depth_value=0.5)
```

**기능**:
- 3D 효과의 바운딩 박스 그리기
- 클래스별 색상 자동 설정
- 깊이에 따른 오프셋 동적 조정

#### 2.2 **BirdEyeViewRenderer** - 조감도
```python
bev = BirdEyeViewRenderer(width=400, height=400, scale=30)
bev.reset()
bev.draw_object(class_name='car', depth=2.5, lateral_offset=0.2)
bev_image = bev.get_image()
```

**기능**:
- 위에서 본 시점의 객체 분포 표시
- 깊이 정보 표현 (그리드 및 거리 마커)
- 객체 ID 표시

#### 2.3 **VisualizationComposer** - 통합 시각화
```python
image = VisualizationComposer.overlay_info(image, fps=30.5, device='cuda')
image = VisualizationComposer.add_bev_corner(image, bev_image, 'bottom-left')
```

**기능**:
- FPS, 장치 정보 오버레이
- 조감도를 메인 이미지에 추가

---

### 3. **kalman_tracker.py** - 칼만 필터 추적

**클래스**: `KalmanTracker`

**기능**:
- 객체 위치, 크기, 방향 추적
- 속도 및 가속도 추정
- 시간적 평활화(Temporal Smoothing)

**주요 메서드**:
```python
tracker = KalmanTracker(max_history=5)

# 측정값으로 업데이트
filtered = tracker.update(object_id=1, measurement={
    'location': np.array([0.1, 0.2, 2.0]),
    'dimensions': np.array([1.7, 0.6, 0.6]),
    'orientation': 0.5
})

# 다음 프레임 예측
predicted = tracker.predict(object_id=1)

# 평활화된 상태 조회
smoothed = tracker.get_smoothed_state(object_id=1)

# 추적 통계
stats = tracker.get_statistics(object_id=1)
# {
#     'velocity': float,
#     'velocity_direction': np.array([x, y, z]),
#     'movement_speed': float,
#     'angular_velocity': float,
#     'frames_tracked': int
# }

# 불필요한 필터 정리
tracker.cleanup(active_ids=[1, 2, 3])
```

---

## 🚀 사용 예제

### 기본 사용법 (example_usage.py)
```python
from example_usage import ObjectTrackingPipeline

pipeline = ObjectTrackingPipeline(
    video_source=0,      # 웹캠
    device='cuda',       # CUDA 가속
    enable_bev=True      # 조감도 활성화
)

pipeline.run(output_file='output.mp4')
```

### 커스텀 파이프라인
```python
from object_detection_core import ObjectDetectionCore
from visualization_core import Visualizer, BirdEyeViewRenderer
from kalman_tracker import KalmanTracker
import cv2
import numpy as np

# 모듈 초기화
detector = ObjectDetectionCore(device='cuda')
visualizer = Visualizer(640, 480)
tracker = KalmanTracker()

# 비디오 루프
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. 감지
    annotated, detections = detector.detect(frame)
    
    # 2. 추적
    for det in detections:
        if det['object_id']:
            # 깊이 추정 (예: 박스 크기로부터)
            bbox = det['bbox']
            depth = 0.5  # 예제값
            
            # 칼만 필터 업데이트
            tracker.update(det['object_id'], {
                'location': np.array([0, 0, depth]),
                'dimensions': np.array([1.7, 0.6, 0.6]),
                'orientation': 0
            })
    
    # 3. 시각화
    for det in detections:
        annotated = visualizer.draw_3d_box(annotated, det)
    
    cv2.imshow('Result', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🔧 설정 및 튜닝

### 객체 감지 튜닝
```python
detector = ObjectDetectionCore(
    model_size='nano',        # nano, small, medium, large, extra
    conf_threshold=0.25,      # 낮을수록 많은 객체 감지 (속도 ↓)
    iou_threshold=0.45,       # NMS IOU 임계값
    device='cuda'             # cuda, cpu, mps
)
```

### 칼만 필터 튜닝
```python
tracker = KalmanTracker(
    max_history=5,                    # 과거 프레임 수
    process_noise_scale=0.1,          # 시스템 노이즈 (높을수록 민감)
    measurement_noise_scale=0.5       # 측정 노이즈 (높을수록 느림)
)
```

### 조감도 튜닝
```python
bev = BirdEyeViewRenderer(
    width=400,      # 픽셀 너비
    height=400,     # 픽셀 높이
    scale=30        # 픽셀/미터 (높을수록 줌인)
)
```

---

## 📊 출력 데이터 구조

### 감지 결과
```python
{
    'bbox': [x1, y1, x2, y2],      # 픽셀 좌표
    'confidence': 0.95,             # 신뢰도
    'class_id': 2,                  # 클래스 ID
    'object_id': 1,                 # 추적 ID
    'class_name': 'car',            # 클래스 이름
    'depth_value': 0.5              # 정규화된 깊이 (0~1)
}
```

### 칼만 필터 상태
```python
{
    'location': np.array([x, y, z]),         # 3D 위치 (미터)
    'dimensions': np.array([h, w, l]),       # 크기 (미터)
    'orientation': 0.5,                      # 방향 각도 (라디안)
    'velocity': np.array([vx, vy, vz]),      # 3D 속도
    'angular_velocity': 0.1                  # 회전 속도
}
```

---

## ⚙️ 의존성

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
ultralytics>=8.0.0
numpy>=1.22.0
filterpy>=1.4.5  # Kalman filter
scipy>=1.10.0
```

설치:
```bash
pip install torch torchvision opencv-python ultralytics filterpy scipy numpy
```

---

## 💡 주요 특징

1. **모듈식 설계**: 각 모듈을 독립적으로 사용 가능
2. **칼만 필터**: 안정적인 객체 추적 및 속도 추정
3. **3D 시각화**: 깊이 기반 3D 박스 표현
4. **조감도**: 공간적 관계 이해 용이
5. **확장 가능**: 추가 기능 쉽게 통합 가능

---

## 📝 라이선스

MIT License
