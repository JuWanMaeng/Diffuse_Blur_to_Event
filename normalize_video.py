import cv2
import os

def extract_frames(video_path, output_folder):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 출력 폴더 확인 및 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    print(f'Total frames: {frame_count}, Frame rate: {frame_rate}')
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 프레임 파일명 생성 및 저장
        frame_filename = os.path.join(output_folder, f'frame_{frame_idx:06d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1
    
    # 객체 해제
    cap.release()
    cv2.destroyAllWindows()

# 사용 예제
extract_frames('/workspace/data/LIVE-HFR/library/library_crf_0_120fps.webm', '/workspace/data/LIVE-HFR/library/frames')
