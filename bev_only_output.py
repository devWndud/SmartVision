"""
BEV (Bird's Eye View) 전용 출력 스크립트

조감도만 별도로 녹화하는 간단한 스크립트입니다.
"""

import cv2
import numpy as np
import time
from example_usage import ObjectTrackingPipeline


def create_bev_only_output(video_source=0, output_bev_file='output_bev_only.mp4', device='cpu'):
    """
    BEV만 녹화합니다
    
    Args:
        video_source: 0=웹캠, 또는 비디오 파일 경로
        output_bev_file: BEV 저장 경로
        device: 'cpu' 또는 'cuda'
    """
    print(f"BEV 전용 녹화 시작")
    print(f"출력: {output_bev_file}")
    
    pipeline = ObjectTrackingPipeline(
        video_source=video_source,
        device=device,
        enable_bev=True
    )
    
    # BEV만 저장하려면 output_bev_file 사용
    # output_file은 None으로 (메인 영상 저장 안 함)
    pipeline.run(
        output_file=None,
        output_bev_file=output_bev_file
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BEV 전용 녹화')
    parser.add_argument('--source', type=str, default='0', help='비디오 소스 (0=웹캠, 또는 파일 경로)')
    parser.add_argument('--output', type=str, default='output_bev_only.mp4', help='BEV 출력 파일')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='실행 장치')
    
    args = parser.parse_args()
    
    # 소스 변환 (문자열 "0"을 정수 0으로)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    create_bev_only_output(
        video_source=source,
        output_bev_file=args.output,
        device=args.device
    )
