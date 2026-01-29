"""
객체 인식 모듈 (YOLO 기반)
간단하고 실용적인 객체 감지 기능을 제공합니다.
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque


class ObjectDetectionCore:
    """
    YOLOv11 기반 객체 감지 및 추적
    """
    
    def __init__(self, model_size='nano', conf_threshold=0.25, iou_threshold=0.45, 
                 device=None):
        """
        초기화
        
        Args:
            model_size (str): 모델 크기 ('nano', 'small', 'medium', 'large', 'extra')
            conf_threshold (float): 신뢰도 임계값
            iou_threshold (float): IoU 임계값
            device (str): 실행 장치 ('cuda', 'cpu', 'mps')
        """
        # 장치 결정
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 모델 로드
        model_map = {
            'nano': 'yolo11n',
            'small': 'yolo11s',
            'medium': 'yolo11m',
            'large': 'yolo11l',
            'extra': 'yolo11x'
        }
        
        model_name = model_map.get(model_size.lower(), 'yolo11n')
        self.model = YOLO(model_name)
        self.model.overrides['conf'] = conf_threshold
        self.model.overrides['iou'] = iou_threshold
        
        # 추적 궤적 저장
        self.tracking_trajectories = {}
        
        print(f"✓ 객체 감지 모델 로드 완료 (Device: {self.device})")
    
    def detect(self, image, enable_tracking=True):
        """
        이미지에서 객체 감지
        
        Args:
            image (numpy.ndarray): 입력 이미지 (BGR)
            enable_tracking (bool): 추적 활성화 여부
            
        Returns:
            tuple: (annotated_image, detections)
                - annotated_image: 객체가 그려진 이미지
                - detections: 감지 결과 리스트
                  각 감지: [bbox, confidence, class_id, object_id]
                  bbox: [x1, y1, x2, y2]
        """
        annotated_image = image.copy()
        detections = []
        
        try:
            if enable_tracking:
                results = self.model.track(image, verbose=False, device=self.device, persist=True)
            else:
                results = self.model.predict(image, verbose=False, device=self.device)
        except RuntimeError as e:
            if self.device == 'mps' and "not currently implemented" in str(e):
                # MPS 오류 시 CPU로 폴백
                results = self.model.track(image, verbose=False, device='cpu', persist=True) if enable_tracking else self.model.predict(image, verbose=False, device='cpu')
            else:
                raise
        
        if enable_tracking:
            # 더 이상 추적되지 않는 객체 정리
            active_ids = []
            for predictions in results:
                if predictions and predictions.boxes:
                    for bbox in predictions.boxes:
                        if hasattr(bbox, 'id') and bbox.id is not None:
                            active_ids.append(int(bbox.id))
            
            for obj_id in list(self.tracking_trajectories.keys()):
                if obj_id not in active_ids:
                    del self.tracking_trajectories[obj_id]
        
        # 결과 처리
        for predictions in results:
            if not predictions or not predictions.boxes:
                continue
            
            for bbox in predictions.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()
                score = float(bbox.conf)
                class_id = int(bbox.cls)
                obj_id = int(bbox.id) if (enable_tracking and hasattr(bbox, 'id') and bbox.id is not None) else None
                
                # 감지 정보 추가
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': score,
                    'class_id': class_id,
                    'object_id': obj_id,
                    'class_name': predictions.names[class_id]
                })
                
                # 바운딩 박스 그리기
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 225), 2)
                
                # 레이블 그리기
                label = f"ID:{obj_id} " if obj_id else ""
                label += f"{predictions.names[class_id]} {score:.2f}"
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                dim, baseline = text_size[0], text_size[1]
                
                cv2.rectangle(annotated_image, 
                             (int(x1), int(y1)), 
                             (int(x1) + dim[0], int(y1) - dim[1] - baseline), 
                             (30, 30, 30), cv2.FILLED)
                cv2.putText(annotated_image, label, 
                           (int(x1), int(y1) - 7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 궤적 업데이트
                if obj_id is not None:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    if obj_id not in self.tracking_trajectories:
                        self.tracking_trajectories[obj_id] = deque(maxlen=20)
                    
                    self.tracking_trajectories[obj_id].append((center_x, center_y))
        
        # 궤적 그리기
        for obj_id, trajectory in self.tracking_trajectories.items():
            for i in range(1, len(trajectory)):
                thickness = int(2 * (i / len(trajectory)) + 1)
                cv2.line(annotated_image,
                        (int(trajectory[i-1][0]), int(trajectory[i-1][1])),
                        (int(trajectory[i][0]), int(trajectory[i][1])),
                        (255, 255, 255), thickness)
        
        return annotated_image, detections
    
    def get_class_names(self):
        """YOLO 클래스 이름 반환"""
        return self.model.names
