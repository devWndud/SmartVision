# BEV 출력 가이드

## 📺 세 가지 출력 방식

### 1️⃣ **BEV만 저장** (추천)
```bash
python bev_only_output.py --source 0 --output output_bev.mp4
```

**결과:**
- `output_bev.mp4` - BEV 조감도 영상만 (1000x1000)
- 메인 영상은 저장 안 함 (파일 크기 작음)

**사용 시기:**
- ISEF 발표용 깔끔한 조감도
- 분석 및 시각화 전용

---

### 2️⃣ **메인 + BEV 동시 저장**
```bash
python both_outputs.py --source 0 --output-main output_main.mp4 --output-bev output_bev.mp4
```

**결과:**
- `output_main.mp4` - 메인 프레임 (640x480) + 위험도 지표
- `output_bev.mp4` - BEV 조감도 (1000x1000)

**사용 시기:**
- 완전한 기록 및 분석
- 논문/보고서에 두 가지 시각 모두 필요할 때

---

### 3️⃣ **코드로 직접 호출**

#### BEV만 저장:
```python
from bev_only_output import create_bev_only_output

create_bev_only_output(
    video_source=0,              # 웹캠
    output_bev_file='bev.mp4',
    device='cpu'
)
```

#### 메인 + BEV:
```python
from both_outputs import create_both_outputs

create_both_outputs(
    video_source=0,
    output_main='main.mp4',
    output_bev='bev.mp4',
    device='cpu'
)
```

#### 또는 직접 파이프라인:
```python
from example_usage import ObjectTrackingPipeline

pipeline = ObjectTrackingPipeline(video_source=0, device='cpu', enable_bev=True)

# BEV만
pipeline.run(output_file=None, output_bev_file='bev.mp4')

# 또는 메인 + BEV
pipeline.run(output_file='main.mp4', output_bev_file='bev.mp4')
```

---

## 🎬 출력 파일 사양

### BEV 영상
```
해상도: 1000x1000 픽셀
코덱: MP4V (H.264)
FPS: 30 (입력과 동일)
색상: BGR (OpenCV)

내용:
├─ 객체 위치 (원형)
├─ 위험도 표시 (색상)
│  ├─ 빨강: DANGER
│  ├─ 주황: CAUTION
│  └─ 초록: SAFE
├─ 거리 마커 (격자)
├─ 안전거리 원 (2m)
└─ 객체 ID + 정보
```

### 메인 프레임 영상
```
해상도: 640x480 픽셀
코덱: MP4V
FPS: 30

내용:
├─ 3D 바운딩박스
├─ 객체 ID + 클래스명
├─ 위험도 지표 (좌상단)
├─ FPS 정보
└─ 시스템 상태
```

---

## 💾 파일 크기 예상

```
웹캠 30초 녹화 (30fps):

BEV만:              ~50MB  (1000x1000)
메인만:             ~30MB  (640x480)
메인 + BEV:         ~80MB  (총합)
```

---

## 🎯 ISEF 발표용 권장 설정

### 데모 영상 만들기:

```bash
# 30초 정도의 깔끔한 BEV 영상 생성
python bev_only_output.py \
  --source 0 \
  --output isef_bev_demo.mp4 \
  --device cpu
```

### 분석 자료용:

```bash
# 메인 + BEV 함께 저장 (논문/포스터에 사용)
python both_outputs.py \
  --source test_video.mp4 \
  --output-main isef_main.mp4 \
  --output-bev isef_bev.mp4 \
  --device cpu
```

---

## ⚙️ 트러블슈팅

### Q: BEV가 저장되지 않음
**A:** 다음 확인:
- `enable_bev=True` 설정 확인
- 디스크 공간 충분한지 확인
- 출력 경로 접근 권한 확인

### Q: 파일 크기가 너무 큼
**A:** 
- 짧은 영상만 저장
- BEV만 저장하기 (메인 영상보다 작음)
- 해상도 낮추기 (코드 수정 필요)

### Q: 재생이 안 됨
**A:**
- MP4 코덱 지원 확인
- VLC Media Player로 시도
- 프레임 크기 맞는지 확인

---

## 📝 참고

- **웹캠 입력:** `--source 0`
- **파일 입력:** `--source video.mp4`
- **GPU 사용:** `--device cuda` (NVIDIA 그래픽카드 필요)
