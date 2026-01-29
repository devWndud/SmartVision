"""
칼만 필터 기반 객체 추적 모듈
객체의 위치, 크기, 방향 등을 부드럽게 추적합니다.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from collections import defaultdict


class KalmanTracker:
    """
    칼만 필터를 이용한 3D 객체 추적
    """
    
    def __init__(self, max_history=5, process_noise_scale=0.1, measurement_noise_scale=0.5):
        """
        초기화
        
        Args:
            max_history (int): 유지할 최대 과거 프레임 수
            process_noise_scale (float): 프로세스 노이즈 스케일
            measurement_noise_scale (float): 측정 노이즈 스케일
        """
        self.filters = {}  # 객체별 칼만 필터
        self.history = defaultdict(list)  # 객체별 위치 이력
        self.max_history = max_history
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
    
    def _create_filter(self, initial_state):
        """
        새로운 칼만 필터 생성
        
        Args:
            initial_state (dict): 초기 상태
                - location: [x, y, z] 3D 위치 (미터)
                - dimensions: [height, width, length] 크기 (미터)
                - orientation: 회전 각도 (라디안)
                
        Returns:
            filterpy.kalman.KalmanFilter: 초기화된 칼만 필터
        """
        # 상태: [x, y, z, w, h, l, yaw, vx, vy, vz, v_yaw]
        # (위치, 크기, 방향, 속도)
        kf = KalmanFilter(dim_x=11, dim_z=7)
        
        # 초기 상태 설정
        location = initial_state.get('location', np.array([0, 0, 1]))
        dimensions = initial_state.get('dimensions', np.array([1.7, 0.6, 0.6]))
        orientation = initial_state.get('orientation', 0)
        
        kf.x = np.array([
            location[0],          # x
            location[1],          # y
            location[2],          # z (깊이)
            dimensions[1],        # width
            dimensions[0],        # height
            dimensions[2],        # length
            orientation,          # yaw
            0, 0, 0, 0           # 초기 속도 = 0
        ], dtype=np.float32)
        
        # 상태 전이 행렬 (motion model)
        dt = 1.0  # 프레임 간 시간 간격
        kf.F = np.eye(11, dtype=np.float32)
        kf.F[0, 7] = dt   # x += vx * dt
        kf.F[1, 8] = dt   # y += vy * dt
        kf.F[2, 9] = dt   # z += vz * dt
        kf.F[6, 10] = dt  # yaw += v_yaw * dt
        
        # 측정 행렬 (관측 가능한 상태)
        kf.H = np.zeros((7, 11), dtype=np.float32)
        kf.H[0, 0] = 1  # x 측정
        kf.H[1, 1] = 1  # y 측정
        kf.H[2, 2] = 1  # z 측정
        kf.H[3, 3] = 1  # width 측정
        kf.H[4, 4] = 1  # height 측정
        kf.H[5, 5] = 1  # length 측정
        kf.H[6, 6] = 1  # yaw 측정
        
        # 측정 노이즈 공분산
        kf.R = np.eye(7, dtype=np.float32) * self.measurement_noise_scale
        kf.R[0:3, 0:3] *= 0.8   # 위치 노이즈
        kf.R[3:6, 3:6] *= 0.2   # 크기 노이즈
        kf.R[6, 6] *= 0.5       # 방향 노이즈
        
        # 프로세스 노이즈 공분산
        kf.Q = np.eye(11, dtype=np.float32) * self.process_noise_scale
        kf.Q[7:11, 7:11] *= 0.5  # 속도 노이즈
        
        # 초기 추정 오차 공분산
        kf.P = np.eye(11, dtype=np.float32)
        kf.P[7:11, 7:11] *= 10.0  # 속도 불확실성 (큼)
        
        return kf
    
    def update(self, object_id, measurement):
        """
        객체 측정값으로 칼만 필터 업데이트
        
        Args:
            object_id (int): 객체 ID
            measurement (dict): 측정값
                - location: [x, y, z]
                - dimensions: [height, width, length]
                - orientation: yaw
                - bbox: [x1, y1, x2, y2] (선택)
                
        Returns:
            dict: 필터링된 상태
        """
        # 새로운 객체일 경우 필터 생성
        if object_id not in self.filters:
            self.filters[object_id] = self._create_filter(measurement)
        
        kf = self.filters[object_id]
        
        # 예측 단계
        kf.predict()
        
        # 측정값 준비
        location = measurement.get('location', np.array([0, 0, 1]))
        dimensions = measurement.get('dimensions', np.array([1.7, 0.6, 0.6]))
        orientation = measurement.get('orientation', 0)
        
        z = np.array([
            location[0],    # x
            location[1],    # y
            location[2],    # z
            dimensions[1],  # width
            dimensions[0],  # height
            dimensions[2],  # length
            orientation     # yaw
        ], dtype=np.float32)
        
        # 업데이트 단계
        kf.update(z)
        
        # 이력에 추가
        filtered_state = {
            'location': np.array([kf.x[0], kf.x[1], kf.x[2]]),
            'dimensions': np.array([kf.x[4], kf.x[3], kf.x[5]]),  # height, width, length
            'orientation': kf.x[6],
            'velocity': np.array([kf.x[7], kf.x[8], kf.x[9]]),
            'angular_velocity': kf.x[10]
        }
        
        # 원본 측정값도 포함
        filtered_state.update(measurement)
        
        self.history[object_id].append(filtered_state)
        if len(self.history[object_id]) > self.max_history:
            self.history[object_id].pop(0)
        
        return filtered_state
    
    def predict(self, object_id):
        """
        객체의 다음 프레임 위치 예측
        
        Args:
            object_id (int): 객체 ID
            
        Returns:
            dict: 예측된 상태 (객체가 없으면 None)
        """
        if object_id not in self.filters:
            return None
        
        kf = self.filters[object_id]
        
        # 예측만 수행 (업데이트 없음)
        kf.predict()
        
        return {
            'location': np.array([kf.x[0], kf.x[1], kf.x[2]]),
            'dimensions': np.array([kf.x[4], kf.x[3], kf.x[5]]),
            'orientation': kf.x[6]
        }
    
    def get_smoothed_state(self, object_id):
        """
        시간적 평활화를 적용한 객체 상태 반환
        
        Args:
            object_id (int): 객체 ID
            
        Returns:
            dict: 평활화된 상태 (객체가 없으면 None)
        """
        if object_id not in self.history or not self.history[object_id]:
            return None
        
        history = self.history[object_id]
        
        if len(history) < 2:
            return history[-1]
        
        # 지수 이동 평균(EMA) 적용
        alpha = 0.7  # 현재 측정값의 가중치
        
        smoothed = history[-1].copy()
        
        # 위치 평활화
        location = smoothed['location'].copy()
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            location = location * (1 - weight) + history[i]['location'] * weight
        
        smoothed['location'] = location
        
        return smoothed
    
    def cleanup(self, active_ids):
        """
        더 이상 추적되지 않는 객체의 필터 정리
        
        Args:
            active_ids (list): 현재 활성 객체 ID 리스트
        """
        active_set = set(active_ids)
        
        # 필터 정리
        for obj_id in list(self.filters.keys()):
            if obj_id not in active_set:
                del self.filters[obj_id]
        
        # 이력 정리
        for obj_id in list(self.history.keys()):
            if obj_id not in active_set:
                del self.history[obj_id]
    
    def get_statistics(self, object_id):
        """
        객체의 추적 통계 반환
        
        Args:
            object_id (int): 객체 ID
            
        Returns:
            dict: 속도, 가속도, 방향 변화 등의 통계
        """
        if object_id not in self.history or len(self.history[object_id]) < 2:
            return None
        
        history = self.history[object_id]
        
        # 최근 속도
        recent_velocity = history[-1].get('velocity', np.array([0, 0, 0]))
        velocity_magnitude = np.linalg.norm(recent_velocity)
        
        # 속도 방향
        if velocity_magnitude > 0:
            velocity_direction = recent_velocity / velocity_magnitude
        else:
            velocity_direction = np.array([0, 0, 0])
        
        # 위치 변화 (최근 2프레임)
        if len(history) >= 2:
            delta_location = history[-1]['location'] - history[-2]['location']
            movement_speed = np.linalg.norm(delta_location)
        else:
            movement_speed = 0
        
        return {
            'velocity': velocity_magnitude,
            'velocity_direction': velocity_direction,
            'movement_speed': movement_speed,
            'angular_velocity': history[-1].get('angular_velocity', 0),
            'frames_tracked': len(history)
        }
    
    def predict_future(self, object_id, frames=16, dt=1/30.0):
        """
        미래 위치 예측 (1~2초)
        
        Args:
            object_id (int): 객체 ID
            frames (int): 예측 프레임 수 (기본 16 = ~0.5초)
            dt (float): 프레임 간 시간 간격 (초)
        
        Returns:
            list: 미래 상태들 리스트
                [{'location': [x,y,z], 'velocity': [vx,vy,vz], ...}, ...]
        """
        if object_id not in self.filters:
            return []
        
        kf = self.filters[object_id]
        future_states = []
        
        # 현재 상태 복사
        x_current = kf.x.copy()
        
        for _ in range(frames):
            # 상태 전이 (A @ x + B @ u)
            x_next = kf.F @ x_current
            
            # 현재 상태 업데이트
            x_current = x_next
            
            # 상태 벡터 추출
            location = x_current[0:3].flatten()
            dimensions = x_current[3:6].flatten()
            orientation = x_current[6]
            velocity = x_current[7:10].flatten()
            angular_velocity = x_current[10]
            
            future_states.append({
                'location': location,
                'dimensions': dimensions,
                'orientation': orientation,
                'velocity': velocity,
                'angular_velocity': angular_velocity,
                'frame_offset': (_ + 1)
            })
        
        return future_states
