# RealSense D435i 통합 가이드

## 개요
ObjectTrackingPipeline은 이제 **Intel RealSense D435i 깊이 카메라**를 완전히 지원합니다. 실제 카메라에서 얻은 깊이값을 사용하여 정확한 3D 객체 추적을 수행합니다.

## 필요한 구성 요소

### 1. 하드웨어
- **Intel RealSense D435i** 카메라 (또는 호환 RealSense 깊이 카메라)
- USB 3.0 이상 포트
- 충분한 USB 전원 공급

### 2. 소프트웨어 패키지
```bash
pip install pyrealsense2
pip install opencv-python
pip install ultralytics
pip install filterpy
pip install numpy
```

모든 패키지는 이미 설치되어 있습니다.

## 프로그램 실행

### 기본 실행
```bash
python example_usage.py
```

### 설정 옵션
`example_usage.py` 마지막 부분에서 다음을 수정할 수 있습니다:

```python
pipeline = ObjectTrackingPipeline(
    video_source=0,      # 0 = RealSense (기본값) 또는 웹캠
    device='cpu',        # 'cpu' 또는 'cuda' (GPU 가능하면)
    enable_bev=True      # 조감도(BEV) 활성화/비활성화
)
pipeline.run(output_file='output.mp4')  # 비디오 저장 (선택)
```

## 동작 원리

### 깊이 데이터 수집
1. **RealSense 파이프라인 초기화**
   - 컬러 스트림: 640×480 @ 30fps
   - 깊이 스트림: 640×480 @ 30fps (Z16 형식)

2. **깊이값 변환**
   ```
   깊이(미터) = 깊이배열(uint16) × depth_scale(0.001)
   ```

3. **ROI 기반 중간값 필터링**
   - 각 객체의 바운딩박스 중앙에서 ±20픽셀 영역 검사
   - 유효한 깊이값(0 초과)들의 중간값 사용
   - 장점: 단일 픽셀 노이즈에 강건함

### 거리 정규화
```
정규화 깊이 = 실제 거리(미터) / 5.0
범위: 0.0 ~ 1.0 (5미터까지 표현)
```

## 출력 해석

### 콘솔 출력 예
```
✓ 파이프라인 시작 (Device: cpu)
✓ 해상도: 640x480, FPS: 30
✓ 조감도: 활성
✓ RealSense 파이프라인: 활성
  - Depth Scale: 0.0010000000474974513
✓ 'q' 또는 ESC로 종료
```

### 비디오 표시
- **상단**: 객체 감지 및 추적 (바운딩박스 + ID)
- **하단-좌**: 조감도(BEV) 시각화
- **하단-우**: 프레임별 정보

### 조감도 해석
- **X축**: 카메라로부터 좌우 위치 (중앙 = 0)
- **Y축**: 카메라로부터 거리 (위쪽 = 더 멀음)
- **색상**: 객체 클래스별로 다름
- **숫자**: 추적 객체 ID

## 카메라 센서 문제 해결

### 문제: "깊이값이 0이거나 너무 크다"
**원인**: 카메라가 너무 멀거나 센서 보정 필요

**해결책**:
1. 카메라가 피사체로부터 30cm~3미터 범위에 위치
2. 카메라 렌즈에서 먼지 제거
3. RealSense Viewer로 깊이값 확인:
   ```bash
   realsense-viewer  # RealSense SDK 설치 필요
   ```

### 문제: "카메라를 찾을 수 없음"
**해결책**:
1. USB 케이블 재연결
2. 다른 USB 포트 시도
3. PowerShell에서 확인:
   ```powershell
   wmic logicaldisk get name  # USB 드라이브 확인
   ```

### 문제: "ImportError: No module named 'pyrealsense2'"
**해결책**: 패키지 재설치
```bash
pip install --upgrade pyrealsense2
```

## 성능 최적화

### 1. 깊이값 필터링 강화
코드의 ROI 크기를 조정하여 노이즈 제거 개선:

```python
# example_usage.py, 약 113줄
x1 = max(0, center_x - 20)  # ← 20을 더 큰 값으로 변경 (예: 30, 40)
x2 = min(depth_frame.shape[1], center_x + 20)
y1 = max(0, center_y - 20)
y2 = min(depth_frame.shape[0], center_y + 20)
```

### 2. 깊이 범위 조정
카메라 운영 환경에 맞게 최대 거리 수정:

```python
depth = depth_raw / 5.0  # ← 5.0을 다른 값으로 변경
# 예: 사용 범위가 0~2미터면 / 2.0 사용
# 예: 사용 범위가 0~10미터면 / 10.0 사용
```

### 3. 프레임 스킵
모든 프레임 처리 대신 2개 중 1개만 처리 (속도 향상):

```python
# run() 메서드 내에서
frame_count = 0
for ...:
    frame_count += 1
    if frame_count % 2 == 0:  # 2 프레임마다 처리
        continue
    # ... 프레임 처리 코드
```

## 주요 특징

✅ **실시간 처리**: 30fps로 카메라 스트림 처리
✅ **다중 객체 추적**: Kalman 필터로 안정적인 ID 유지  
✅ **3D 시각화**: 
   - 원근감 있는 3D 바운딩박스
   - 조감도(BEV) 뷰
   - 객체 궤적 표시
✅ **실제 깊이값**: 박스 크기 추정 대신 카메라 센서값 사용
✅ **노이즈 강건성**: 중간값 필터로 단일 픽셀 오류 제거

## 참고 자료

- [RealSense Python API 문서](https://intelrealsense.github.io/librealsense/python_docs/)
- [YOLO v11 문서](https://docs.ultralytics.com/models/yolov11/)
- [Kalman 필터 가이드](https://filterpy.readthedocs.io/)

## 라이센스
이 프로젝트의 모든 코드는 MIT 라이센스 하에 배포됩니다.
