import os
import glob
import imageio

def make_gif(image_folder, output_path='animation.gif', duration=0.5):
    """
    이미지 폴더의 모든 PNG/JPG 파일을 읽어 GIF로 저장합니다.
    
    Args:
        image_folder (str): 이미지들이 들어있는 디렉터리 경로.
        output_path  (str): 생성할 GIF 파일 경로.
        duration     (float): 각 프레임당 지속 시간(초).
    """
    # PNG, JPG 파일 리스트 추출 및 정렬
    patterns = [os.path.join(image_folder, f'*.{ext}') for ext in ('png','jpg','jpeg')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files, reverse=True)
    if not files:
        raise FileNotFoundError(f"No images found in {image_folder}")

    # 이미지 읽어서 리스트에 저장
    frames = []
    for img_path in files:
        frames.append(imageio.imread(img_path))

    # GIF 생성
    imageio.mimsave(output_path, frames, duration=duration)
    print(f"✔ Saved GIF: {output_path}")

if __name__ == "__main__":
    # 예시: debug/original 폴더의 이미지를 모아 0.5초 프레임으로 GIF 생성
    make_gif('debug/NAFVAE', 'debug/NAFVAE/animation.gif', duration=0.5)