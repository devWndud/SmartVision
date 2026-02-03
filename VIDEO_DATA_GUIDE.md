# 비디오 테스트 데이터 기록 및 재현

카메라 없이도 비디오 파일을 사용하여 테스트 데이터를 기록하고, 저장된 데이터로 테스트를 수행할 수 있습니다.

## 📋 개요

이 시스템은 두 가지 주요 모듈로 구성됩니다:

### 1. **VideoDataRecorder** (`video_data_recorder.py`)
- 비디오 파일을 입력으로 받아 처리
- 모든 프레임의 이미지 저장
- 각 프레임의 깊이 맵 생성 및 저장
- 객체 감지 결과 저장 (바운딩박스, 신뢰도, 깊이값 등)
- 메타데이터와 통계 정보 저장

### 2. **VideoDataLoader** (`video_data_loader.py`)
- 저장된 데이터를 로드하여 사용
- 프레임별 접근, 감지 정보 조회
- 객체 궤적 추적
- 데이터 시각화 및 분석

## 🚀 사용 방법

### Step 1: 비디오 파일에서 테스트 데이터 기록

#### 명령어 실행:
```bash
python video_data_recorder.py <비디오_파일_경로> [옵션]
```

#### 예제:
```bash
# 기본 사용 (모든 프레임 처리)
python video_data_recorder.py data/test_videos/my_video.mp4

# GPU 사용 (빠른 처리)
python video_data_recorder.py data/test_videos/my_video.mp4 --device cuda

# 프레임 건너뛰기 (매 2번째 프레임만 처리 - 저장 용량 줄이기)
python video_data_recorder.py data/test_videos/my_video.mp4 --skip-frames 1

# 사용자 정의 출력 디렉토리
python video_data_recorder.py data/test_videos/my_video.mp4 --output-dir my_sessions
```

#### 옵션:
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--output-dir` | 데이터 저장 디렉토리 | `data/recorded_sessions` |
| `--skip-frames` | 건너뛸 프레임 수 (0=모두 처리) | `0` |
| `--device` | 실행 장치 (cpu/cuda) | `cpu` |

#### 출력:
```
🎥 비디오 처리 시작: my_video.mp4
📁 저장 경로: data/recorded_sessions/session_20260203_145230
  - 해상도: 1280x720
  - FPS: 30.00
  - 총 프레임: 300
  - 예상 시간: 10.0초

  ✓ 50 프레임 저장 (12.5초)
  ✓ 100 프레임 저장 (25.1초)
  ...

✅ 처리 완료!
  - 저장된 프레임: 300
  - 총 감지: 756
  - 처리 시간: 50.3초
  - 저장 경로: data/recorded_sessions/session_20260203_145230
```

### Step 2: 저장된 데이터 로드 및 사용

#### Python 코드에서 사용:

```python
from video_data_loader import VideoDataLoader

# 세션 로드
loader = VideoDataLoader('data/recorded_sessions/session_20260203_145230')

# 1. 메타데이터 조회
metadata = loader.get_metadata()
print(f"총 프레임: {metadata['total_frames']}")
print(f"해상도: {metadata['width']}x{metadata['height']}")
print(f"총 감지: {metadata['total_detections']}")

# 2. 특정 프레임 로드
frame = loader.get_frame(0)  # 첫 번째 프레임
depth_map = loader.get_depth_map(0)  # 깊이 맵

# 3. 프레임의 감지 정보 조회
detections = loader.get_detections_for_frame(0)
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
    print(f"  - 위치: {det['bbox']}")
    print(f"  - 깊이: {det['depth_value']:.2f}m")

# 4. 모든 프레임 순회
for frame, detections in loader.get_all_frames_with_detections():
    print(f"프레임: {len(detections)}개 객체")

# 5. 특정 객체의 궤적 추적
trajectory = loader.get_object_trajectory(object_id=1)
for point in trajectory:
    print(f"프레임 {point['frame_index']}: {point['bbox']}")

# 6. 통계 정보 조회
stats = loader.get_statistics()
print(f"고유 객체 수: {stats['unique_objects']}")
print(f"프레임당 평균 감지: {stats['average_detections_per_frame']:.2f}")

# 7. 시각화
visualized = loader.visualize_frame(0, show_depth=True)
```

#### 데모 실행:
```bash
python video_data_loader.py
```

자동으로 최신 세션을 로드하여 통계, 샘플 프레임, 객체 궤적을 출력합니다.

## 📁 저장된 데이터 구조

```
data/recorded_sessions/
└── session_20260203_145230/          # 타임스탬프 기반 세션명
    ├── frames/                       # 모든 프레임 이미지
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    ├── depth_maps/                   # 깊이 맵 (NumPy 배열)
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    ├── detections.json               # 모든 감지 결과
    └── metadata.json                 # 메타데이터 및 통계
```

## 📄 JSON 파일 형식

### `metadata.json`
```json
{
  "session_name": "session_20260203_145230",
  "created_at": "2026-02-03T14:52:30.123456",
  "device": "cpu",
  "source_video": "my_video.mp4",
  "total_frames": 300,
  "width": 1280,
  "height": 720,
  "fps": 30.0,
  "duration_seconds": 10.0,
  "processing_time_seconds": 50.3,
  "total_detections": 756,
  "class_distribution": {
    "car": 450,
    "person": 200,
    "bicycle": 106
  }
}
```

### `detections.json`
```json
[
  {
    "frame_index": 0,
    "timestamp_ms": 0.0,
    "frame_filename": "000000.jpg",
    "depth_filename": "000000.npy",
    "detections": [
      {
        "class_name": "car",
        "class_id": 2,
        "confidence": 0.92,
        "object_id": 1,
        "bbox": [100.5, 200.3, 350.1, 450.7],
        "depth_value": 0.65
      },
      {
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.87,
        "object_id": 2,
        "bbox": [500.2, 150.1, 600.8, 480.5],
        "depth_value": 0.45
      }
    ]
  },
  ...
]
```

## 🔧 고급 사용

### 테스트 파이프라인에 통합

```python
from video_data_loader import VideoDataLoader
from example_usage import ObjectTrackingPipeline

loader = VideoDataLoader('data/recorded_sessions/session_xyz')

# 저장된 데이터로 추가 처리 수행
for frame_idx in range(loader.metadata['total_frames']):
    frame = loader.get_frame(frame_idx)
    detections = loader.get_detections_for_frame(frame_idx)
    
    # 여기서 추가 분석, 검증 등 수행
    for det in detections:
        # 예: 신뢰도 필터링
        if det['confidence'] > 0.8:
            print(f"{det['class_name']} detected")
```

### 벤치마크 수행

```python
loader = VideoDataLoader('data/recorded_sessions/session_xyz')
stats = loader.get_statistics()

print(f"처리된 프레임: {stats['total_frames']}")
print(f"총 감지: {stats['total_detections']}")
print(f"FPS: {stats['fps']}")
print(f"비디오 길이: {stats['duration_seconds']}초")
print(f"객체 종류별 분포: {stats['class_distribution']}")
```

## ⚠️ 주의사항

1. **저장 용량**: 모든 프레임을 저장하므로 긴 비디오는 많은 디스크 용량이 필요합니다.
   - 10분 1280x720 비디오 → ~2-3GB

2. **프레임 건너뛰기**: 용량을 줄이려면 `--skip-frames` 옵션 사용
   - `--skip-frames 1`: 매 2번째 프레임만 (50% 용량 감소)
   - `--skip-frames 2`: 매 3번째 프레임만 (33% 용량 감소)

3. **GPU 가속**: NVIDIA GPU가 있으면 `--device cuda` 사용으로 빠른 처리 가능

4. **깊이값**: 카메라가 없으므로 바운딩박스 크기를 기반으로 깊이 추정

## 📚 활용 예제

### 1. 특정 객체 추적하기
```python
loader = VideoDataLoader('data/recorded_sessions/session_xyz')
trajectory = loader.get_object_trajectory(object_id=1)
print(f"객체 ID 1의 이동 경로: {trajectory}")
```

### 2. 객체 종류별 통계
```python
stats = loader.get_statistics()
for class_name, count in stats['class_distribution'].items():
    percentage = (count / stats['total_detections']) * 100
    print(f"{class_name}: {count} ({percentage:.1f}%)")
```

### 3. 시각화 및 저장
```python
loader = VideoDataLoader('data/recorded_sessions/session_xyz')
import cv2

output = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, (1280, 720))

for frame, _ in loader.get_all_frames_with_detections():
    vis_frame = loader.visualize_frame(frame_idx, show_depth=True)
    output.write(vis_frame)

output.release()
```

## 🐛 문제 해결

### "세션 경로를 찾을 수 없습니다"
- 올바른 세션 경로를 확인하세요
- `data/recorded_sessions/` 디렉토리 확인

### "프레임을 찾을 수 없습니다"
- 프레임 인덱스 범위가 올바른지 확인
- `loader.metadata['total_frames']` 확인

### "비디오를 열 수 없습니다"
- 비디오 파일 경로 확인
- 지원되는 비디오 형식: MP4, AVI, MOV 등

## 📞 추가 정보

더 많은 정보는 `example_usage.py`와 다른 모듈들을 참고하세요.
