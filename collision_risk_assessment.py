"""
충돌 위험도 평가 모듈

기능:
1. Time-to-Collision (TTC) 계산
2. 상대 속도 및 이동 방향 분석
3. 미래 위치 예측 (1~2초)
4. 위험도 등급 판단 (안전/주의/위험)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class RiskAssessment:
    """위험도 평가 결과"""
    object_id: int
    class_name: str
    
    # 현재 상태
    current_position: np.ndarray  # [x, y, z]
    current_distance: float
    current_velocity: float  # m/s
    
    # 미래 예측
    future_positions: List[np.ndarray]  # 1~2초 후 위치들
    predicted_distance: float
    
    # 위험도 분석
    time_to_collision: Optional[float]  # 초 단위, None이면 충돌 없음
    relative_velocity: float  # 상대 속도 (음수 = 가까워짐)
    approach_direction: np.ndarray  # 단위 벡터
    collision_probability: float  # 0.0~1.0
    risk_level: str  # 'safe', 'caution', 'danger'
    risk_score: float  # 0.0~100.0
    
    # 물리적 영향도
    collision_impact: float  # 예상 충돌 에너지 (상대 질량 고려)


class CollisionRiskAssessment:
    """충돌 위험도 평가 엔진"""
    
    def __init__(self, 
                 prediction_frames: int = 16,  # 1~2초 (30fps 기준)
                 ego_position: np.ndarray = np.array([0, 0, 0]),
                 ego_velocity: np.ndarray = np.array([0, 0, 0]),
                 ego_size: np.ndarray = np.array([1.7, 0.6, 0.6])):  # h, w, l
        """
        초기화
        
        Args:
            prediction_frames: 예측 프레임 수 (기본 16 = ~0.5초)
            ego_position: 자차 위치 [x, y, z]
            ego_velocity: 자차 속도 [vx, vy, vz]
            ego_size: 자차 크기 [h, w, l]
        """
        self.prediction_frames = prediction_frames
        self.ego_position = ego_position
        self.ego_velocity = ego_velocity
        self.ego_size = ego_size
        
        # 위험도 판정 기준
        self.TTC_DANGER = 2.0  # 2초 이내 = 위험
        self.TTC_CAUTION = 3.5  # 3.5초 이내 = 주의
        self.VELOCITY_THRESHOLD = 0.3  # m/s (가까워지는 속도 기준)
        self.SAFE_DISTANCE = 2.0  # 안전 거리 (미터)
    
    def set_ego_state(self, position: np.ndarray, velocity: np.ndarray):
        """자차 상태 업데이트"""
        self.ego_position = position.copy()
        self.ego_velocity = velocity.copy()
    
    def assess_risk(self, 
                   object_id: int,
                   class_name: str,
                   current_state: Dict,
                   future_states: List[Dict]) -> RiskAssessment:
        """
        단일 객체의 위험도 평가
        
        Args:
            object_id: 객체 ID
            class_name: 클래스명
            current_state: 현재 상태 {'location', 'velocity', 'dimensions'}
            future_states: 미래 상태들 리스트
        
        Returns:
            RiskAssessment 객체
        """
        # 현재 위치 및 속도
        obj_position = np.array(current_state['location'])
        obj_velocity = np.array(current_state['velocity'])
        obj_size = np.array(current_state['dimensions'])
        
        # 1. 현재 거리 계산
        current_distance = np.linalg.norm(obj_position - self.ego_position)
        
        # 2. 현재 속도 (크기)
        current_velocity = np.linalg.norm(obj_velocity)
        
        # 3. 상대 속도 (음수 = 가까워짐)
        relative_velocity = self._calculate_relative_velocity(
            obj_position, obj_velocity,
            self.ego_position, self.ego_velocity
        )
        
        # 4. 접근 방향
        approach_direction = self._calculate_approach_direction(
            obj_position, obj_velocity,
            self.ego_position, self.ego_velocity
        )
        
        # 5. 미래 위치 예측
        future_positions = self._predict_future_positions(
            obj_position, obj_velocity
        )
        
        # 6. 예측된 거리 (1~2초 후)
        predicted_distance = np.linalg.norm(
            future_positions[-1] - self.ego_position
        )
        
        # 7. Time-to-Collision 계산
        ttc = self._calculate_ttc(
            obj_position, obj_velocity,
            self.ego_position, self.ego_velocity,
            obj_size, self.ego_size
        )
        
        # 8. 충돌 확률
        collision_probability = self._calculate_collision_probability(
            current_distance, predicted_distance, ttc, relative_velocity
        )
        
        # 9. 위험도 등급
        risk_level, risk_score = self._determine_risk_level(
            ttc, current_distance, collision_probability, relative_velocity
        )
        
        # 10. 충돌 영향도 (질량 고려 가중치)
        collision_impact = self._calculate_collision_impact(
            current_velocity, obj_class=class_name
        )
        
        return RiskAssessment(
            object_id=object_id,
            class_name=class_name,
            current_position=obj_position,
            current_distance=current_distance,
            current_velocity=current_velocity,
            future_positions=future_positions,
            predicted_distance=predicted_distance,
            time_to_collision=ttc,
            relative_velocity=relative_velocity,
            approach_direction=approach_direction,
            collision_probability=collision_probability,
            risk_level=risk_level,
            risk_score=risk_score,
            collision_impact=collision_impact
        )
    
    def assess_all_risks(self,
                        tracked_objects: Dict) -> List[RiskAssessment]:
        """
        모든 객체의 위험도 평가
        
        Args:
            tracked_objects: {object_id: {'class_name', 'state', 'future_states'}, ...}
        
        Returns:
            RiskAssessment 리스트
        """
        assessments = []
        for obj_id, obj_data in tracked_objects.items():
            assessment = self.assess_risk(
                object_id=obj_id,
                class_name=obj_data.get('class_name', 'unknown'),
                current_state=obj_data['state'],
                future_states=obj_data.get('future_states', [])
            )
            assessments.append(assessment)
        
        # 위험도 순서로 정렬
        assessments.sort(key=lambda x: x.risk_score, reverse=True)
        return assessments
    
    # ======================== 내부 계산 함수 ========================
    
    def _calculate_relative_velocity(self,
                                    obj_pos: np.ndarray,
                                    obj_vel: np.ndarray,
                                    ego_pos: np.ndarray,
                                    ego_vel: np.ndarray) -> float:
        """
        상대 속도 계산 (음수 = 가까워짐)
        """
        # 자차에서 객체로의 벡터
        relative_pos = obj_pos - ego_pos
        distance = np.linalg.norm(relative_pos)
        
        if distance < 0.001:  # 0에 가까우면 방지
            return 0.0
        
        # 정규화된 방향
        direction = relative_pos / distance
        
        # 상대 속도
        rel_vel_vector = obj_vel - ego_vel
        relative_speed = np.dot(rel_vel_vector, direction)
        
        return relative_speed  # 음수 = 가까워짐
    
    def _calculate_approach_direction(self,
                                     obj_pos: np.ndarray,
                                     obj_vel: np.ndarray,
                                     ego_pos: np.ndarray,
                                     ego_vel: np.ndarray) -> np.ndarray:
        """
        접근 방향 벡터 계산 (단위 벡터)
        """
        relative_pos = obj_pos - ego_pos
        distance = np.linalg.norm(relative_pos)
        
        if distance < 0.001:
            return np.array([0, 0, 0])
        
        return relative_pos / distance  # 단위 벡터
    
    def _predict_future_positions(self,
                                 position: np.ndarray,
                                 velocity: np.ndarray) -> List[np.ndarray]:
        """
        미래 위치 예측 (등속 모델)
        
        Returns:
            16 프레임(~0.5초)의 미래 위치들
        """
        dt = 1/30.0  # 30fps 기준
        future_positions = []
        
        current_pos = position.copy()
        for _ in range(self.prediction_frames):
            current_pos = current_pos + velocity * dt
            future_positions.append(current_pos.copy())
        
        return future_positions
    
    def _calculate_ttc(self,
                      obj_pos: np.ndarray,
                      obj_vel: np.ndarray,
                      ego_pos: np.ndarray,
                      ego_vel: np.ndarray,
                      obj_size: np.ndarray,
                      ego_size: np.ndarray) -> Optional[float]:
        """
        Time-to-Collision 계산
        
        Returns:
            충돌까지 시간(초), None이면 충돌 없음
        """
        relative_pos = obj_pos - ego_pos
        relative_vel = obj_vel - ego_vel
        
        distance = np.linalg.norm(relative_pos)
        
        # 안전 거리 계산 (두 물체 크기 고려)
        safety_distance = (np.linalg.norm(obj_size) + 
                          np.linalg.norm(ego_size)) / 4.0
        
        # 상대 속도 (가까워지는 방향)
        if distance < 0.001:
            return 0.0
        
        direction = relative_pos / distance
        closing_speed = -np.dot(relative_vel, direction)
        
        # 가까워지지 않으면 충돌 없음
        if closing_speed <= 0:
            return None
        
        # TTC = (현재 거리 - 안전거리) / 가까워지는 속도
        ttc = (distance - safety_distance) / closing_speed
        
        if ttc < 0:
            return 0.0  # 이미 충돌 범위 내
        
        return ttc
    
    def _calculate_collision_probability(self,
                                        current_distance: float,
                                        predicted_distance: float,
                                        ttc: Optional[float],
                                        relative_velocity: float) -> float:
        """
        충돌 확률 계산 (0.0~1.0)
        """
        prob = 0.0
        
        # 1. TTC 기반
        if ttc is not None:
            if ttc < self.TTC_DANGER:
                prob += 0.6  # 위험 시간 범위면 60%
            elif ttc < self.TTC_CAUTION:
                prob += 0.3  # 주의 범위면 30%
        
        # 2. 거리 기반
        if current_distance < self.SAFE_DISTANCE:
            prob += 0.2  # 안전거리 이하면 20%
        
        # 3. 접근 속도 기반
        if relative_velocity < -self.VELOCITY_THRESHOLD:
            prob += 0.2  # 빠르게 가까워지면 20%
        
        return min(prob, 1.0)  # 0.0~1.0으로 클립
    
    def _determine_risk_level(self,
                             ttc: Optional[float],
                             distance: float,
                             collision_prob: float,
                             relative_velocity: float) -> Tuple[str, float]:
        """
        위험도 등급 판정 (안전/주의/위험)
        
        Returns:
            (risk_level, risk_score)
        """
        risk_score = 0.0
        
        # TTC 점수
        if ttc is not None:
            if ttc < self.TTC_DANGER:
                risk_score += 40
            elif ttc < self.TTC_CAUTION:
                risk_score += 20
            else:
                risk_score += 5
        
        # 거리 점수
        if distance < self.SAFE_DISTANCE:
            risk_score += 30
        elif distance < self.SAFE_DISTANCE * 1.5:
            risk_score += 15
        else:
            risk_score += 5
        
        # 접근 속도 점수
        if relative_velocity < -0.5:  # 빠르게 가까워짐
            risk_score += 20
        elif relative_velocity < -self.VELOCITY_THRESHOLD:
            risk_score += 10
        
        # 충돌 확률 점수
        risk_score += collision_prob * 10
        
        # 등급 결정
        if risk_score >= 60:
            risk_level = 'danger'
        elif risk_score >= 30:
            risk_level = 'caution'
        else:
            risk_level = 'safe'
        
        return risk_level, min(risk_score, 100.0)
    
    def _calculate_collision_impact(self,
                                   object_velocity: float,
                                   obj_class: str) -> float:
        """
        충돌 영향도 계산 (예상 에너지)
        
        클래스별 예상 질량:
        - person: 70kg
        - bicycle: 15kg
        - motorcycle: 150kg
        - car: 1500kg
        - truck: 5000kg
        - bus: 5000kg
        """
        # 클래스별 질량 (kg)
        mass_map = {
            'person': 70,
            'bicycle': 15,
            'motorcycle': 150,
            'car': 1500,
            'truck': 5000,
            'bus': 5000,
            'potted plant': 5,
            'plant': 5,
        }
        
        mass = mass_map.get(obj_class, 100)  # 기본값 100kg
        
        # 운동 에너지: E = 0.5 * m * v^2
        # 정규화: 기준값 대비 비율
        reference_energy = 0.5 * 1500 * (5.0 ** 2)  # 자차 표준값
        actual_energy = 0.5 * mass * (object_velocity ** 2)
        
        impact = (actual_energy / reference_energy) * 100
        
        return min(impact, 100.0)
