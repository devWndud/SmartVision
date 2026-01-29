"""
시각화 모듈
객체 검출 결과를 다양한 방식으로 시각화합니다.
"""

import numpy as np
import cv2
import math

# ==================== 색상 정의 (BGR 형식) ====================
# 모든 시각화에서 동일하게 사용하는 색상
COLOR_RED = (0, 0, 255)        # 빨강 - DANGER
COLOR_ORANGE = (0, 140, 255)   # 주황 - CAUTION  
COLOR_GREEN = (0, 255, 0)      # 초록 - SAFE
COLOR_BLUE = (255, 100, 0)     # 파랑 - 차량 기본색
COLOR_CYAN = (128, 128, 0)     # 시안 - 자전거 기본색
COLOR_YELLOW = (0, 255, 255)   # 노랑 - 식물 기본색
COLOR_WHITE = (255, 255, 255)  # 흰색
COLOR_GRAY = (200, 200, 200)   # 회색 - 기본값


class Visualizer:
    """
    3D 객체 시각화 및 Bird's Eye View 렌더링
    """
    
    def __init__(self, image_width, image_height):
        """
        초기화
        
        Args:
            image_width (int): 이미지 너비
            image_height (int): 이미지 높이
        """
        self.image_width = image_width
        self.image_height = image_height
    
    def draw_3d_box(self, image, detection, depth_value=0.5, color=None):
        """
        3D 효과의 바운딩 박스 그리기
        
        Args:
            image (numpy.ndarray): 대상 이미지
            detection (dict): 감지 정보
                - bbox: [x1, y1, x2, y2]
                - class_name: 클래스명
                - confidence: 신뢰도
                - object_id: 객체 ID (선택)
            depth_value (float): 깊이값 (0~1)
            color (tuple): BGR 색상
            
        Returns:
            numpy.ndarray: 그려진 이미지
        """
        x1, y1, x2, y2 = [int(c) for c in detection['bbox']]
        width = x2 - x1
        height = y2 - y1
        
        # 색상 결정
        if color is None:
            class_name = detection['class_name'].lower()
            color = self._get_color_by_class(class_name)
        
        # 깊이에 따른 오프셋 계산
        offset_factor = max(0.1, 1.0 - depth_value)
        offset_x = int(max(15, min(width * 0.3 * offset_factor, 50)))
        offset_y = int(max(15, min(height * 0.3 * offset_factor, 50)))
        
        # 프론트 페이스 (2D 바운딩 박스)
        front_tl, front_tr = (x1, y1), (x2, y1)
        front_br, front_bl = (x2, y2), (x1, y2)
        
        # 백 페이스 (깊이 오프셋)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        
        # 프론트 페이스 그리기
        cv2.rectangle(image, front_tl, front_br, color, 2)
        
        # 연결선 그리기
        for (ft, bt) in [(front_tl, back_tl), (front_tr, back_tr),
                         (front_br, back_br), (front_bl, back_bl)]:
            cv2.line(image, ft, bt, color, 2)
        
        # 백 페이스 그리기
        cv2.line(image, back_tl, back_tr, color, 2)
        cv2.line(image, back_tr, back_br, color, 2)
        cv2.line(image, back_br, back_bl, color, 2)
        cv2.line(image, back_bl, back_tl, color, 2)
        
        # 투명 오버레이 (3D 효과)
        overlay = image.copy()
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        # 텍스트 정보 그리기
        text_y = y1 - 10
        
        if detection.get('object_id'):
            cv2.putText(image, f"ID:{detection['object_id']}", (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 18
        
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(image, label, (x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 지면 연결선 그리기 (깊이 표현)
        center_x = (x1 + x2) // 2
        ground_y = y2 + int(height * 0.2)
        cv2.line(image, (center_x, y2), (center_x, ground_y), color, 2)
        cv2.circle(image, (center_x, ground_y), 3, color, -1)
        
        return image
    
    def draw_risk_indicator(self, image, risk_level, risk_score, position=(10, 30)):
        """
        위험도 지표 표시 (왼쪽 상단)
        
        Args:
            image: 대상 이미지
            risk_level: 'safe', 'caution', 'danger'
            risk_score: 0.0~100.0
            position: 표시 위치 (x, y)
        """
        x, y = position
        
        # 색상 결정 (COLOR_* 상수 사용)
        if risk_level == 'danger':
            color = COLOR_RED  # 빨강
            level_text = "DANGER !!"
        elif risk_level == 'caution':
            color = COLOR_ORANGE  # 주황
            level_text = "CAUTION !"
        else:
            color = COLOR_GREEN  # 초록
            level_text = "SAFE"
        
        # 배경 박스
        cv2.rectangle(image, (x-5, y-25), (x+200, y+35), (50, 50, 50), -1)
        cv2.rectangle(image, (x-5, y-25), (x+200, y+35), color, 2)
        
        # 텍스트
        cv2.putText(image, level_text, (x+5, y-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Score: {risk_score:.1f}/100", (x+5, y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 위험도 바 그리기
        bar_width = 100
        bar_height = 8
        bar_x = x + 5
        bar_y = y + 25
        
        # 배경 바
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # 위험도 바 (채우기)
        fill_width = int(bar_width * min(risk_score / 100.0, 1.0))
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        return image
    
    def _get_color_by_class(self, class_name):
        """클래스에 따른 색상 반환"""
        if any(x in class_name for x in ['car', 'vehicle', 'truck', 'bus']):
            return COLOR_BLUE  # 파랑
        elif 'person' in class_name:
            return COLOR_GREEN  # 초록
        elif any(x in class_name for x in ['bicycle', 'motorcycle']):
            return COLOR_CYAN  # 시안
        elif any(x in class_name for x in ['plant', 'potted']):
            return COLOR_YELLOW  # 노랑
        else:
            return COLOR_GRAY  # 회색


class BirdEyeViewRenderer:
    """
    Bird's Eye View (조감도) 시각화
    """
    
    def __init__(self, width=400, height=400, scale=30):
        """
        초기화
        
        Args:
            width (int): 조감도 이미지 너비 (픽셀)
            height (int): 조감도 이미지 높이 (픽셀)
            scale (float): 스케일 (픽셀/미터)
        """
        self.width = width
        self.height = height
        self.scale = scale
        
        # 원점: 이미지 아래 중앙
        self.origin_x = width // 2
        self.origin_y = height - 50
        
        self.image = self._create_base_image()
    
    def _create_base_image(self):
        """기본 조감도 이미지 생성"""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 20
        
        # 그리드 그리기
        grid_spacing = max(int(self.scale), 20)
        for y in range(self.origin_y, 0, -grid_spacing):
            cv2.line(image, (0, y), (self.width, y), (50, 50, 50), 1)
        for x in range(0, self.width, grid_spacing):
            cv2.line(image, (x, 0), (x, self.height), (50, 50, 50), 1)
        
        # 좌표축 그리기
        axis_length = 80
        cv2.line(image, (self.origin_x, self.origin_y),
                (self.origin_x, self.origin_y - axis_length), (0, 200, 0), 2)  # X축
        cv2.line(image, (self.origin_x, self.origin_y),
                (self.origin_x + axis_length, self.origin_y), (0, 0, 200), 2)  # Y축
        
        # 거리 마커 (1~5미터)
        for dist in range(1, 6):
            y = self.origin_y - int(dist * self.scale)
            cv2.line(image, (self.origin_x - 5, y), (self.origin_x + 5, y), (120, 120, 120), 2)
            cv2.putText(image, f"{dist}m", (self.origin_x + 10, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        return image
    
    def reset(self):
        """조감도 초기화"""
        self.image = self._create_base_image()
    
    def draw_object(self, class_name, depth, lateral_offset=0, object_id=None):
        """
        객체를 조감도에 그리기
        
        Args:
            class_name (str): 클래스명
            depth (float): 깊이 (미터)
            lateral_offset (float): 측면 오프셋 (정규화됨, -1~1)
            object_id (int): 객체 ID (선택)
        """
        class_name = class_name.lower()
        
        # 색상 결정
        if any(x in class_name for x in ['car', 'vehicle', 'truck', 'bus']):
            color = (0, 0, 255)  # 빨강
        elif 'person' in class_name:
            color = (0, 255, 0)  # 초록
        elif any(x in class_name for x in ['bicycle', 'motorcycle']):
            color = (255, 0, 0)  # 파랑
        elif 'plant' in class_name or 'potted' in class_name:
            color = (0, 255, 255)  # 노랑
        else:
            color = (255, 255, 255)  # 흰색
        
        # 픽셀 좌표 계산
        bev_y = int(self.origin_y - depth * self.scale)
        bev_x = int(self.origin_x + lateral_offset * self.width * 0.3)
        
        # 범위 체크
        bev_x = max(20, min(bev_x, self.width - 20))
        bev_y = max(20, min(bev_y, self.origin_y - 10))
        
        # 형태에 따라 그리기
        if 'person' in class_name:
            cv2.circle(self.image, (bev_x, bev_y), 4, color, -1)
        elif any(x in class_name for x in ['car', 'vehicle', 'truck', 'bus']):
            rect_width = 12 if 'truck' not in class_name else 15
            rect_length = 18 if 'truck' not in class_name else 24
            cv2.rectangle(self.image,
                         (bev_x - rect_width//2, bev_y - rect_length//2),
                         (bev_x + rect_width//2, bev_y + rect_length//2),
                         color, -1)
        elif 'plant' in class_name or 'potted' in class_name:
            cv2.circle(self.image, (bev_x, bev_y), 8, color, -1)
        else:
            cv2.rectangle(self.image,
                         (bev_x - 6, bev_y - 6),
                         (bev_x + 6, bev_y + 6),
                         color, -1)
        
        # 객체 ID 그리기 (숫자만 사용)
        if object_id is not None:
            try:
                id_str = str(int(object_id))
                cv2.putText(self.image, id_str, (bev_x - 5, bev_y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except (ValueError, TypeError):
                pass
        
        # 원점에서 객체로의 거리선 그리기
        cv2.line(self.image, (self.origin_x, self.origin_y), (bev_x, bev_y),
                (70, 70, 70), 1)
    
    def get_image(self):
        """조감도 이미지 반환"""
        return self.image.copy()
    
    def draw_object_with_risk(self, class_name, depth, lateral_offset=0, 
                             object_id=None, risk_assessment=None):
        """
        위험도와 함께 객체를 조감도에 그리기
        
        Args:
            class_name (str): 클래스명
            depth (float): 깊이 (미터)
            lateral_offset (float): 측면 오프셋 (-1~1)
            object_id (int): 객체 ID
            risk_assessment: RiskAssessment 객체 (선택)
        """
        try:
            class_name = class_name.lower()
            
            # 기본 색상
            base_color = self._get_object_color(class_name)
            
            # 위험도에 따라 색상 오버라이드
            if risk_assessment:
                if risk_assessment.risk_level == 'danger':
                    color = COLOR_RED  # 빨강
                    risk_symbol = "!!"
                elif risk_assessment.risk_level == 'caution':
                    color = COLOR_ORANGE  # 주황
                    risk_symbol = "!"
                else:
                    color = base_color
                    risk_symbol = None
            else:
                color = base_color
                risk_symbol = None
            
            # 픽셀 좌표 계산
            bev_y = int(self.origin_y - depth * self.scale)
            bev_x = int(self.origin_x + lateral_offset * self.width * 0.3)
            
            # 범위 체크
            bev_x = max(20, min(bev_x, self.width - 20))
            bev_y = max(20, min(bev_y, self.origin_y - 10))
            
            # 미래 궤적 그리기 (위험도 평가가 있을 때) - 간소화
            # future_positions는 정규화된 좌표이므로 직접 깊이만 사용
            if risk_assessment and risk_assessment.future_positions:
                try:
                    for i, future_pos in enumerate(risk_assessment.future_positions[:8]):  # 최대 8프레임만
                        # 미래 깊이만 사용 (Z 축)
                        future_depth = future_pos[2] if len(future_pos) > 2 else depth
                        future_y = int(self.origin_y - future_depth * self.scale)
                        
                        # 범위 체크
                        if 20 <= future_y <= self.origin_y - 10:
                            # 점 크기 감소 (과거 → 현재로 갈수록 커짐)
                            radius = max(1, 2 - i // 4)
                            overlay = self.image.copy()
                            cv2.circle(overlay, (bev_x, future_y), radius, color, -1)
                            cv2.addWeighted(overlay, 0.15, self.image, 0.85, 0, self.image)
                except Exception:
                    pass  # 미래 궤적 그리기 실패 무시
            
            # 형태에 따라 그리기
            if 'person' in class_name:
                cv2.circle(self.image, (bev_x, bev_y), 4, color, -1)
            elif any(x in class_name for x in ['car', 'vehicle', 'truck', 'bus']):
                rect_width = 12 if 'truck' not in class_name else 15
                rect_length = 18 if 'truck' not in class_name else 24
                cv2.rectangle(self.image,
                             (bev_x - rect_width//2, bev_y - rect_length//2),
                             (bev_x + rect_width//2, bev_y + rect_length//2),
                             color, -1)
            elif 'plant' in class_name or 'potted' in class_name:
                cv2.circle(self.image, (bev_x, bev_y), 8, color, -1)
            else:
                cv2.rectangle(self.image,
                             (bev_x - 6, bev_y - 6),
                             (bev_x + 6, bev_y + 6),
                             color, -1)
            
            # 객체 ID 그리기 (숫자만 사용, 인코딩 문제 해결)
            if object_id is not None:
                try:
                    # 정수로 변환해서 순수 숫자만 표시
                    id_str = str(int(object_id))
                    cv2.putText(self.image, id_str, (bev_x - 6, bev_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except (ValueError, TypeError):
                    pass
            
            # 위험도 기호 표시 (BEV에 명확하게)
            if risk_symbol:
                # !! = 빨강 (DANGER)
                # ! = 노랑 (CAUTION)
                symbol_color = color  # 같은 색상 사용
                cv2.putText(self.image, risk_symbol, (bev_x + 8, bev_y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, symbol_color, 2)
            
            # 원점에서 객체로의 거리선 그리기
            cv2.line(self.image, (self.origin_x, self.origin_y), (bev_x, bev_y),
                    (70, 70, 70), 1)
        
        except Exception as e:
            # 그리기 실패 무시
            pass
    
    def _get_object_color(self, class_name):
        """클래스명에 따른 기본 색상 반환"""
        # BGR 형식 (OpenCV)
        if any(x in class_name for x in ['car', 'vehicle', 'truck', 'bus']):
            return COLOR_BLUE  # 파란색
        elif 'person' in class_name:
            return COLOR_GREEN  # 초록
        elif any(x in class_name for x in ['bicycle', 'motorcycle']):
            return COLOR_CYAN  # 시안
        elif 'plant' in class_name or 'potted' in class_name:
            return COLOR_YELLOW  # 노랑
        else:
            return COLOR_GRAY  # 회색


class VisualizationComposer:
    """
    여러 시각화를 조합하여 표시
    """
    
    @staticmethod
    def overlay_info(image, fps=None, device=None, **kwargs):
        """
        이미지에 정보 오버레이
        
        Args:
            image (numpy.ndarray): 대상 이미지
            fps (float): FPS 값
            device (str): 실행 장치
            **kwargs: 추가 정보 (예: tracking_enabled=True)
            
        Returns:
            numpy.ndarray: 정보가 추가된 이미지
        """
        h, w = image.shape[:2]
        
        # FPS 정보
        if fps:
            text = f"FPS: {fps:.1f}"
            if device:
                text += f" | Device: {device}"
            cv2.putText(image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 추적 상태
        if kwargs.get('tracking_enabled'):
            cv2.putText(image, "Tracking: ON", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image
    
    @staticmethod
    def add_bev_corner(main_image, bev_image, position='bottom-left'):
        """
        메인 이미지의 코너에 조감도 추가
        
        Args:
            main_image (numpy.ndarray): 메인 이미지
            bev_image (numpy.ndarray): 조감도 이미지
            position (str): 위치 ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            
        Returns:
            numpy.ndarray: 조감도가 추가된 이미지
        """
        h, w = main_image.shape[:2]
        bev_h = h // 4
        bev_w = bev_h
        
        # 조감도 리사이징
        bev_resized = cv2.resize(bev_image, (bev_w, bev_h))
        
        # 위치별 좌표 계산
        positions = {
            'top-left': (0, 0),
            'top-right': (w - bev_w, 0),
            'bottom-left': (0, h - bev_h),
            'bottom-right': (w - bev_w, h - bev_h)
        }
        
        x, y = positions.get(position, (0, h - bev_h))
        
        # 조감도 오버레이
        main_image[y:y+bev_h, x:x+bev_w] = bev_resized
        
        # 테두리 그리기
        cv2.rectangle(main_image, (x, y), (x + bev_w, y + bev_h), (255, 255, 255), 1)
        
        # 제목 추가
        cv2.putText(main_image, "Bird's Eye View", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return main_image

