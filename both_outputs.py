"""
메인 프레임 + BEV 동시 녹화 스크립트

메인 영상과 BEV 조감도를 동시에 별개의 파일로 저장합니다.
"""

import cv2
import numpy as np
from example_usage import ObjectTrackingPipeline


def create_both_outputs(video_source=0, output_main='output_main.mp4', output_bev='output_bev.mp4', device='cpu'):
    """
    메인 프레임과 BEV를 동시에 녹화합니다
    
    Args:
        video_source: 0=웹캠, 또는 비디오 파일 경로
        output_main: 메인 프레임 저장 경로
        output_bev: BEV 저장 경로
        device: 'cpu' 또는 'cuda'
    """
    print(f"메인 + BEV 동시 녹화 시작")
    print(f"메인 출력: {output_main}")
    print(f"BEV 출력: {output_bev}")
    
    pipeline = ObjectTrackingPipeline(
        video_source=video_source,
        device=device,
        enable_bev=True
    )
    
    # 메인과 BEV 동시 저장
    pipeline.run(
        output_file=output_main,
        output_bev_file=output_bev
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='메인 + BEV 동시 녹화')
    parser.add_argument('--source', type=str, default='0', help='비디오 소스 (0=웹캠, 또는 파일 경로)')
    parser.add_argument('--output-main', type=str, default='output_main.mp4', help='메인 프레임 출력 파일')
    parser.add_argument('--output-bev', type=str, default='output_bev.mp4', help='BEV 출력 파일')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='실행 장치')
    
    args = parser.parse_args()
    
    # 소스 변환
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    create_both_outputs(
        video_source=source,
        output_main=args.output_main,
        output_bev=args.output_bev,
        device=args.device
    )
