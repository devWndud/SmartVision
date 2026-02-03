# 📹 카메라 데이터 기록 및 재현 시스템

**목표**: D435i 깊이 카메라로 촬영 중인 모든 정보를 저장하고, 나중에 카메라 없이도 동일한 테스트 환경을 재현

## 🎯 핵심 요약

### 작동 원리
```
┌─────────────────────────┐
│  D435i 카메라 실시간    │ → RGB 프레임
│  촬영                    │ → 깊이맵 (미터)
│                         │ → 객체 감지
└──────────────┬──────────┘
               │
               ↓
      ┌────────────────────┐
      │ 실시간 저장         │
      │ (realtime_camera   │
      │  _recorder.py)     │
      └────────┬───────────┘
               │
               ↓
      ┌────────────────────────────┐
      │ data/recorded_sessions/    │
      │ camera_20260203_HHMMSS/    │
      │ ├── color_frames/          │
      │ ├── depth_frames/          │
      │ ├── detections.json        │
      │ └── metadata.json          │
      └────────┬───────────────────┘
               │
               ↓
      ┌────────────────────────────┐
      │ 나중에 카메라 없이 로드      │
      │ (video_data_loader.py)     │
      │ 테스트/분석/시뮬레이션      │
      └────────────────────────────┘
```

## 📂 생성된 파일

| 파일 | 설명 |
|------|------|
| `realtime_camera_recorder.py` | 🎬 카메라 실시간 데이터 기록 (핵심) |
| `video_data_loader.py` | 📖 저장된 데이터 로드 (카메라/비디오 모두 지원) |
| `REALTIME_CAMERA_GUIDE.md` | 📚 상세 사용 설명서 |
| `quick_camera_guide.py` | 🚀 5가지 실용 예제 |

## ⚡ 빠른 시작 (3단계)

### 1️⃣ 카메라로 데이터 기록 (터미널)
```bash
# 10초 기록
python realtime_camera_recorder.py --duration 10

# 또는 Ctrl+C로 중단
python realtime_camera_recorder.py
```

**결과**: `data/recorded_sessions/camera_20260203_HHMMSS/` 생성

### 2️⃣ 저장된 데이터 로드 (Python)
```python
from video_data_loader import VideoDataLoader

loader = VideoDataLoader('data/recorded_sessions/camera_20260203_HHMMSS')

# 첫 프레임 접근
frame = loader.get_frame(0)  # RGB 영상
depth_map = loader.get_depth_map(0)  # 깊이맵 (미터)
detections = loader.get_detections_for_frame(0)  # 감지 정보

# 깊이값 조회 (카메라 데이터만)
for det in detections:
    print(f"{det['class_name']}: {det['depth_meters']:.2f}m")  # 실제 깊이
```

### 3️⃣ 테스트 수행
```python
# 저장된 데이터로 원하는 테스트 수행
# - 객체 추적
# - 거리 측정
# - 위험도 평가
# - 성능 벤치마크
```

## 📊 저장되는 데이터

### 저장 위치
```
data/recorded_sessions/camera_20260203_145230/
├── color_frames/         # RGB 컬러 프레임 (jpg)
│   ├── 000000.jpg       # 프레임 0
│   ├── 000001.jpg       # 프레임 1
│   └── ...
├── depth_frames/        # 깊이맵 (npy, 미터 단위)
│   ├── 000000.npy       # 프레임 0의 깊이
│   ├── 000001.npy       # 프레임 1의 깊이
│   └── ...
├── detections.json      # 모든 감지 결과 + 실제 깊이값
└── metadata.json        # 기록 정보 및 통계
```

### 저장 크기 (참고)
- 1분 기록 → 약 2GB
- 10초 기록 → 약 330MB
- 30프레임 × 100프레임 해상도 640x480

## 🎯 주요 기능

### ✅ `realtime_camera_recorder.py`
- D435i 카메라 자동 감지 및 초기화
- RGB + 깊이 실시간 저장
- YOLO 객체 감지 (자동 감지)
- 실제 깊이값 저장 (픽셀 기반 중앙값)
- 진행률 실시간 표시
- 타임스탬프 기반 자동 폴더 생성

### ✅ `video_data_loader.py` (업그레이드)
- 카메라 데이터 자동 감지
- 프레임 & 깊이맵 로드
- 감지 정보 조회
- 객체 궤적 추적
- 깊이값 기반 분석
- 통계 정보 제공

## 💡 활용 시나리오

### 시나리오 1: 성능 테스트
```python
loader = VideoDataLoader('data/recorded_sessions/camera_xyz')

# 같은 데이터로 여러 번 테스트 (일관된 결과)
for iteration in range(10):
    for frame_idx in range(loader.metadata['total_frames']):
        frame = loader.get_frame(frame_idx)
        # 테스트 코드 실행
```

### 시나리오 2: 깊이 기반 필터링
```python
# 2m 이내 객체만 처리
for frame_idx in range(loader.metadata['total_frames']):
    detections = loader.get_detections_for_frame(frame_idx)
    close_objects = [d for d in detections if 0 < d['depth_meters'] < 2.0]
    # 처리 로직
```

### 시나리오 3: 객체 추적 재현
```python
# 카메라와 정확히 같은 조건으로 추적 테스트
trajectory = loader.get_object_trajectory(object_id=1)
for point in trajectory:
    print(f"프레임 {point['frame_index']}: {point['bbox']}")
```

### 시나리오 4: 시각화
```python
# 깊이맵 가시화
for frame_idx in range(loader.metadata['total_frames']):
    frame = loader.get_frame(frame_idx)
    depth_map = loader.get_depth_map(frame_idx)
    
    # 깊이맵 컬러맵
    depth_colored = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # 병렬 표시
    result = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
```

## 🔧 명령어 참고

### 카메라 데이터 기록
```bash
# 기본 (무한 기록, Ctrl+C 중단)
python realtime_camera_recorder.py

# 10초 자동 기록
python realtime_camera_recorder.py --duration 10

# GPU 가속 (빠른 객체 감지)
python realtime_camera_recorder.py --device cuda

# 사용자 정의 폴더
python realtime_camera_recorder.py --output-dir my_recordings --duration 30
```

### 데이터 분석 예제
```bash
# 5가지 예제 실행
python quick_camera_guide.py
```

## 📖 상세 문서

| 문서 | 내용 |
|------|------|
| `REALTIME_CAMERA_GUIDE.md` | 완전한 API 문서, 사용 예제, 문제 해결 |
| `quick_camera_guide.py` | 5가지 실용 예제 코드 |
| `VIDEO_DATA_GUIDE.md` | 비디오 파일 처리 (참고) |

## ⚠️ 중요 사항

1. **카메라 필수**: D435i가 USB로 연결되어야 함
2. **저장 공간**: 장시간 기록 시 충분한 디스크 공간 필요
3. **깊이값**: 
   - 카메라 데이터: 실제 깊이값 (미터)
   - 비디오 파일: 추정된 깊이값 (상대값 0~1)
4. **타임스탐프**: 각 프레임에 기록 시간 저장됨

## 🎓 예제 실행

```bash
# 1. 카메라로 데이터 기록 (10초)
python realtime_camera_recorder.py --duration 10

# 2. 저장된 데이터 분석 (자동으로 최신 세션 로드)
python quick_camera_guide.py

# 3. 수동 분석 (Python)
from video_data_loader import VideoDataLoader
loader = VideoDataLoader('data/recorded_sessions/camera_20260203_HHMMSS')
print(loader.get_statistics())
```

## 🚀 다음 단계

1. ✅ 카메라로 테스트 데이터 기록
2. ✅ 저장된 데이터 로드 & 검증
3. ✅ 깊이값 기반 분석
4. ✅ 기존 `example_usage.py` 와 통합
5. ✅ 성능 벤치마크 & 최적화

---

**생성일**: 2026-02-03  
**카메라**: D435i (RealSense)  
**해상도**: 640x480  
**FPS**: 30  
**저장 형식**: JPG + NPY + JSON
