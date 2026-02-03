# 🎥 실시간 D435i 카메라 데이터 기록

현재 촬영 중인 D435i 깊이 카메라의 모든 정보를 저장하고, 나중에 카메라 없이도 테스트할 수 있습니다.

## 📋 개요

### 저장되는 데이터
- **RGB 영상**: 모든 컬러 프레임 (jpg)
- **깊이맵**: 실제 깊이값 (미터 단위, npy 형식)
- **객체 감지**: 바운딩박스, 신뢰도, 객체 ID, 실제 깊이값
- **메타데이터**: 기록 시간, FPS, 감지 통계 등

## 🚀 사용 방법

### Step 1: 카메라로 데이터 기록

#### 명령어:
```bash
# 기본 사용 (Ctrl+C로 중단)
python realtime_camera_recorder.py

# 10초 자동 기록
python realtime_camera_recorder.py --duration 10

# GPU 사용 (빠른 처리)
python realtime_camera_recorder.py --device cuda

# 사용자 정의 출력 디렉토리
python realtime_camera_recorder.py --output-dir my_recordings
```

#### 옵션:
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--duration` | 기록 시간 (초) | 무한대 (Ctrl+C 중단) |
| `--output-dir` | 데이터 저장 디렉토리 | `data/recorded_sessions` |
| `--device` | 실행 장치 (cpu/cuda) | `cpu` |

#### 출력 예:
```
🎥 D435i 카메라 시작 중...
✓ D435i 카메라 초기화 완료
  - 해상도: 640x480
  - FPS: 30
  - Depth Scale: 0.0010000000474974513

🔴 기록 시작!
📁 저장 경로: data/recorded_sessions/camera_20260203_145230
⏱️  Ctrl+C를 눌러 기록 중단
  ✓ 30 프레임 저장 (29.8 FPS, 1.0초)
  ✓ 60 프레임 저장 (30.2 FPS, 2.0초)
  ...
⏹️  Ctrl+C로 인해 기록 중단

✅ 기록 완료!
  - 저장된 프레임: 450
  - 총 감지: 1823
  - 기록 시간: 15.0초
  - 평균 FPS: 30.0

📁 저장 경로: data/recorded_sessions/camera_20260203_145230
  - color_frames/: 450개 프레임
  - depth_frames/: 450개 깊이맵
  - detections.json: 감지 정보
  - metadata.json: 메타데이터
```

### Step 2: 저장된 카메라 데이터 로드

#### Python 코드:
```python
from video_data_loader import VideoDataLoader

# 카메라 세션 로드 (자동 감지)
loader = VideoDataLoader('data/recorded_sessions/camera_20260203_145230')

# 1. 메타데이터 조회
print(f"총 프레임: {loader.metadata['total_frames']}")
print(f"기록 시간: {loader.metadata['recording_duration_seconds']}초")
print(f"평균 FPS: {loader.metadata['average_fps']:.1f}")

# 2. 프레임과 실제 깊이맵 로드
frame = loader.get_frame(0)  # RGB 영상
depth_map = loader.get_depth_map(0)  # 깊이맵 (미터 단위)

# 3. 감지 정보 조회
detections = loader.get_detections_for_frame(0)
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
    print(f"  - 위치: {det['bbox']}")
    print(f"  - 실제 깊이: {det['depth_meters']:.2f}m")  # 카메라 실제 깊이

# 4. 특정 ROI의 깊이값 조회
roi_depth = loader.get_frame_depth_meters(0, bbox=[100, 100, 300, 300])
print(f"ROI 깊이: {roi_depth:.2f}m")

# 5. 전체 프레임 순회
for frame, detections in loader.get_all_frames_with_detections():
    print(f"프레임: {len(detections)}개 객체, 깊이 범위: {frame.shape}")

# 6. 통계 정보
stats = loader.get_statistics()
print(stats)
```

## 📁 저장된 데이터 구조

```
data/recorded_sessions/
└── camera_20260203_145230/          # 카메라 세션명
    ├── color_frames/                # RGB 컬러 프레임
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    ├── depth_frames/                # 실제 깊이맵 (미터 단위)
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    ├── detections.json              # 감지 결과 + 실제 깊이값
    └── metadata.json                # 기록 메타데이터
```

## 📄 JSON 파일 형식

### `metadata.json` (카메라 데이터)
```json
{
  "session_name": "camera_20260203_145230",
  "created_at": "2026-02-03T14:52:30.123456",
  "device": "cpu",
  "camera_model": "D435i",
  "total_frames": 450,
  "total_detections": 1823,
  "recording_duration_seconds": 15.0,
  "average_fps": 30.0,
  "class_distribution": {
    "car": 850,
    "person": 600,
    "bicycle": 373
  }
}
```

### `detections.json` (카메라 데이터)
```json
[
  {
    "frame_index": 0,
    "timestamp": "2026-02-03T14:52:30.123456",
    "color_filename": "000000.jpg",
    "depth_filename": "000000.npy",
    "detections": [
      {
        "class_name": "car",
        "class_id": 2,
        "confidence": 0.92,
        "object_id": 1,
        "bbox": [100.5, 200.3, 350.1, 450.7],
        "depth_meters": 2.34  # 실제 깊이값 (미터)
      }
    ]
  }
]
```

## 🎯 활용 예제

### 1. 카메라와 동일한 환경 재현

```python
loader = VideoDataLoader('data/recorded_sessions/camera_20260203_145230')

# 저장된 데이터를 마치 카메라에서 받는 것처럼 처리
for frame_idx in range(loader.metadata['total_frames']):
    frame = loader.get_frame(frame_idx)
    depth_map = loader.get_depth_map(frame_idx)
    detections = loader.get_detections_for_frame(frame_idx)
    
    # 원본 카메라 데이터를 사용한 테스트 수행
    # 예: 추적, 거리 측정, 위험도 평가 등
```

### 2. 깊이값 기반 분석

```python
loader = VideoDataLoader('data/recorded_sessions/camera_20260203_145230')

# 가장 가까운 객체 찾기
min_depth = float('inf')
closest_object = None

for frame_idx in range(loader.metadata['total_frames']):
    detections = loader.get_detections_for_frame(frame_idx)
    for det in detections:
        if det['depth_meters'] > 0 and det['depth_meters'] < min_depth:
            min_depth = det['depth_meters']
            closest_object = {
                'frame': frame_idx,
                'class': det['class_name'],
                'distance': det['depth_meters']
            }

print(f"가장 가까운 객체: {closest_object['class']} at {closest_object['distance']:.2f}m")
```

### 3. 깊이 기반 시각화

```python
import cv2

loader = VideoDataLoader('data/recorded_sessions/camera_20260203_145230')

for frame_idx in range(loader.metadata['total_frames']):
    frame = loader.get_frame(frame_idx)
    depth_map = loader.get_depth_map(frame_idx)
    
    # 깊이맵을 컬러맵으로 변환
    depth_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 영상과 깊이맵 합성
    result = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
    
    cv2.imshow('Camera Recording', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### 4. 거리 기반 필터링

```python
loader = VideoDataLoader('data/recorded_sessions/camera_20260203_145230')

# 1m 이내의 객체만 분석
for frame_idx in range(loader.metadata['total_frames']):
    detections = loader.get_detections_for_frame(frame_idx)
    
    # 1m 이내
    close_objects = [d for d in detections if 0 < d['depth_meters'] < 1.0]
    
    # 1~3m
    mid_objects = [d for d in detections if 1.0 <= d['depth_meters'] < 3.0]
    
    # 3m 이상
    far_objects = [d for d in detections if d['depth_meters'] >= 3.0]
    
    if close_objects:
        print(f"프레임 {frame_idx}: {len(close_objects)}개 객체가 1m 이내")
```

## 🔄 카메라와 비디오 파일 호환성

동일한 `VideoDataLoader`로 둘 다 로드 가능:

```python
# 카메라 데이터
camera_loader = VideoDataLoader('data/recorded_sessions/camera_xyz')

# 비디오 파일 데이터
video_loader = VideoDataLoader('data/recorded_sessions/session_xyz')

# 같은 메서드로 접근
frame = camera_loader.get_frame(0)
frame = video_loader.get_frame(0)

# 실제 깊이값 조회 (카메라만 지원)
if camera_loader.is_camera_data:
    depth_m = camera_loader.get_frame_depth_meters(0)
    print(f"깊이: {depth_m}m")
```

## ⚠️ 주의사항

1. **카메라 연결**: D435i가 연결되어 있어야 합니다
2. **저장 용량**: 30초 기록 → 약 500MB
3. **깊이값**: `depth_meters` 필드는 실제 D435i 카메라의 깊이값 (비디오는 추정값)
4. **타임스탬프**: 각 프레임에 정확한 기록 시간이 저장됨

## 🐛 문제 해결

### "카메라를 찾을 수 없습니다"
- D435i가 USB로 연결되었는지 확인
- `pip install pyrealsense2` 재설치

### "이미지 오류"
- 저장된 색상 프레임 확인: `color_frames/` 디렉토리
- 깊이맵 확인: `depth_frames/` 디렉토리

### "깊이값이 0"
- 카메라가 객체를 감지하지 못함
- 조명 확인, 카메라 거리 조정 필요

## 📚 다른 모듈과의 연동

- **video_data_recorder.py**: 비디오 파일 처리용
- **video_data_loader.py**: 통합 로더 (카메라/비디오 모두 지원)
- **example_usage.py**: 저장된 데이터로 추적/분석 수행
