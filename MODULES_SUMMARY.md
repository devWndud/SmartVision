# 세 가지 핵심 모듈 요약

## 📋 파일 구조

```
SmartVision/
├── object_detection_core.py      ← 객체 감지 모듈
├── visualization_core.py         ← 시각화 모듈
├── kalman_tracker.py            ← 칼만 필터 추적 모듈
├── example_usage.py             ← 통합 예제
├── QUICK_START.py               ← 빠른 시작 (코드 예제)
├── USAGE_GUIDE.md               ← 상세 사용 설명서
└── MODULES_SUMMARY.md           ← 이 파일
```

---

## 🎯 각 모듈의 역할

### 1️⃣ **object_detection_core.py**

```
목적: YOLO 기반 실시간 객체 감지 및 추적
핵심 클래스: ObjectDetectionCore

입력: 이미지 (numpy array, BGR)
┌─────────────────────────────┐
│ 이미지 프레임                  │
└──────────────┬────────────────┘
               │
         ┌─────▼──────┐
         │  YOLO v11  │
         │ 감지 & 추적 │
         └─────┬──────┘
               │
   ┌───────────┴───────────┐
   │                       │
   ▼                       ▼
바운딩박스            객체ID + 궤적
신뢰도
클래스정보

출력: annotated_image, detections 리스트
```

**핵심 메서드**:
```python
detector = ObjectDetectionCore(model_size='nano', device='cuda')
annotated_img, dets = detector.detect(frame, enable_tracking=True)

# 반환 detections:
# [
#   {'bbox': [x1, y1, x2, y2], 'confidence': 0.95, 'class_id': 2, 
#    'object_id': 1, 'class_name': 'car'},
#   ...
# ]
```

---

### 2️⃣ **visualization_core.py**

```
목적: 감지 결과를 다양한 방식으로 시각화
핵심 클래스: 
  - Visualizer (3D 박스)
  - BirdEyeViewRenderer (조감도)
  - VisualizationComposer (합성)

처리 흐름:
입력 이미지
    │
    ├─▶ Visualizer.draw_3d_box()
    │      └─ 깊이에 따른 3D 효과 박스 그리기
    │
    ├─▶ BirdEyeViewRenderer
    │      ├─ 객체 위치를 위에서 본 시점으로 표현
    │      ├─ 거리, 방향 정보 표현
    │      └─ 객체 궤적 표현
    │
    └─▶ VisualizationComposer
           ├─ FPS, 장치 정보 오버레이
           ├─ 조감도 코너 추가
           └─ 통합 이미지 반환
```

**핵심 메서드**:
```python
# 3D 박스 그리기
visualizer = Visualizer(640, 480)
result = visualizer.draw_3d_box(frame, detection, depth_value=0.5)

# 조감도
bev = BirdEyeViewRenderer(width=400, height=400, scale=30)
bev.reset()
bev.draw_object(class_name='car', depth=2.5, lateral_offset=0.2)

# 합성
result = VisualizationComposer.add_bev_corner(result, bev.get_image())
result = VisualizationComposer.overlay_info(result, fps=30.5)
```

---

### 3️⃣ **kalman_tracker.py**

```
목적: 칼만 필터를 이용한 부드러운 객체 추적
핵심 클래스: KalmanTracker

원리:
1. 예측(Predict): 이전 상태와 속도로부터 다음 위치 예측
2. 측정(Measure): 센서/감지기로부터 현재 값 획득
3. 업데이트(Update): 예측과 측정값을 가중 평균하여 최적값 계산

   ┌──────────────────────────────┐
   │ 칼만 필터 상태 공간           │
   ├──────────────────────────────┤
   │ 위치(x,y,z)                  │
   │ 크기(height, width, length)  │
   │ 방향(orientation)            │
   │ 속도(vx, vy, vz)             │
   │ 회전속도(v_yaw)              │
   └──────────────────────────────┘
       │
       ├─ predict(): 다음 상태 예측
       ├─ update(): 측정값으로 업데이트
       └─ get_smoothed_state(): 평활화된 결과

과정:
측정값 ─┐
        ├─▶ [예측] ─▶ [업데이트] ─▶ 필터링된 상태
시간-→  ─┘                        (부드러운 움직임)
```

**핵심 메서드**:
```python
tracker = KalmanTracker()

# 업데이트 (매 프레임마다)
measurement = {
    'location': np.array([0.1, 0.2, 2.5]),
    'dimensions': np.array([1.7, 0.6, 0.6]),
    'orientation': 0.5
}
filtered = tracker.update(object_id=1, measurement=measurement)

# 예측
predicted = tracker.predict(object_id=1)

# 평활화 (이전 프레임들 고려)
smoothed = tracker.get_smoothed_state(object_id=1)

# 통계
stats = tracker.get_statistics(object_id=1)
# {'velocity': 2.5, 'velocity_direction': [...], ...}

# 정리 (활성 객체 유지)
tracker.cleanup(active_ids=[1, 2, 3])
```

---

## 🔄 세 모듈의 통합 흐름

```
┌─────────────────────────────────────────────┐
│          입력 프레임                          │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────▼──────────────┐
    │  ObjectDetectionCore      │
    │  .detect(frame)           │
    └────────────┬──────────────┘
                 │
      ┌──────────▼──────────┐
      │ Detections:         │
      │ - bbox              │
      │ - confidence        │
      │ - object_id         │
      │ - class_name        │
      └──────────┬──────────┘
                 │
    ┌────────────▼──────────────┐
    │  KalmanTracker            │
    │  .update(id, measurement) │
    │  .get_smoothed_state(id)  │
    └────────────┬──────────────┘
                 │
      ┌──────────▼──────────┐
      │ Filtered State:     │
      │ - smooth location   │
      │ - smooth dimensions │
      │ - velocity          │
      │ - statistics        │
      └──────────┬──────────┘
                 │
    ┌────────────▼──────────────┐
    │  Visualizer               │
    │  .draw_3d_box()           │
    │  BirdEyeView.draw_object()│
    │  Composer.overlay_info()  │
    └────────────┬──────────────┘
                 │
┌────────────────▼────────────────┐
│   최종 시각화 이미지              │
│  - 3D 박스                       │
│  - 조감도                        │
│  - FPS, 정보                     │
└────────────────────────────────┘
```

---

## 📊 데이터 흐름 예제

```python
import cv2
import numpy as np
from object_detection_core import ObjectDetectionCore
from visualization_core import Visualizer, BirdEyeViewRenderer
from kalman_tracker import KalmanTracker

# 모듈 초기화
detector = ObjectDetectionCore(device='cuda')
visualizer = Visualizer(640, 480)
bev = BirdEyeViewRenderer()
tracker = KalmanTracker()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 1️⃣ 감지
    annotated, detections = detector.detect(frame, enable_tracking=True)
    # detections: [{'bbox': [100, 150, 200, 300], 'object_id': 1, ...}, ...]
    
    # 2️⃣ 추적
    active_ids = []
    for det in detections:
        if det['object_id']:
            # 깊이 추정 (예: 박스 크기로부터)
            bbox = det['bbox']
            box_size = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2
            depth = max(0.1, (500 - box_size) / 500)
            
            # 칼만 필터 업데이트
            filtered = tracker.update(det['object_id'], {
                'location': np.array([0, 0, depth]),
                'dimensions': np.array([1.7, 0.6, 0.6]),
                'orientation': 0
            })
            
            # 또는 평활화된 상태 조회
            smoothed = tracker.get_smoothed_state(det['object_id'])
            
            # 또는 통계 조회
            stats = tracker.get_statistics(det['object_id'])
            
            active_ids.append(det['object_id'])
    
    tracker.cleanup(active_ids)
    
    # 3️⃣ 시각화
    result = frame.copy()
    
    # 3D 박스 그리기
    for det in detections:
        result = visualizer.draw_3d_box(result, det, depth_value=0.5)
    
    # 조감도 그리기
    bev.reset()
    for det in detections:
        depth = max(0.1, (500 - ((det['bbox'][2] - det['bbox'][0]) + 
                                 (det['bbox'][3] - det['bbox'][1])) / 2) / 500)
        bev.draw_object(det['class_name'], depth * 4 + 1, object_id=det['object_id'])
    
    # 최종 합성
    result = VisualizationComposer.add_bev_corner(result, bev.get_image())
    result = VisualizationComposer.overlay_info(result, fps=30.5, device='cuda')
    
    cv2.imshow('Complete Pipeline', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✨ 각 모듈의 장점

### ObjectDetectionCore
✅ YOLO v11의 빠른 속도  
✅ 자동 객체 추적 (ID 할당)  
✅ 궤적 기록  
✅ 여러 장치 지원 (CPU, CUDA, MPS)

### Visualization
✅ 깊이 기반 3D 효과  
✅ 공간적 관계 표현 (조감도)  
✅ 클래스별 자동 색상  
✅ 정보 오버레이

### KalmanTracker
✅ 부드러운 추적  
✅ 속도/가속도 추정  
✅ 시간적 평활화  
✅ 메모리 효율적 (활성 객체만 추적)

---

## 🚀 실행 방법

### 최소 코드
```python
from example_usage import ObjectTrackingPipeline
pipeline = ObjectTrackingPipeline(video_source=0, device='cuda', enable_bev=True)
pipeline.run()
```

### 상세 제어
[USAGE_GUIDE.md](USAGE_GUIDE.md) 참고

### 코드 예제
[QUICK_START.py](QUICK_START.py) 참고

---

## 📦 설치

```bash
pip install torch torchvision opencv-python ultralytics filterpy scipy numpy
```

---

## 🎓 학습 경로

1. **QUICK_START.py** - 기본 사용 방법 학습
2. **example_usage.py** - 통합 파이프라인 이해
3. **각 모듈 코드** - 세부 구현 이해
4. **USAGE_GUIDE.md** - 고급 튜닝 방법

---

