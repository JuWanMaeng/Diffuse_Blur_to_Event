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


def make_mp4(image_folder, output_path='animation.mp4', duration=0.5):
    """
    이미지 폴더의 모든 PNG/JPG 파일을 읽어 MP4 비디오로 저장합니다.
    
    Args:
        image_folder (str): 이미지들이 들어있는 디렉터리 경로.
        output_path  (str): 생성할 MP4 파일 경로.
        duration     (float): 각 프레임당 지속 시간(초).
    """
    # PNG, JPG 파일 리스트 추출 및 정렬
    patterns = [os.path.join(image_folder, f'*.{ext}') for ext in ('png','jpg','jpeg')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files, reverse=True)  # 순서대로 정렬
    if not files:
        raise FileNotFoundError(f"No images found in {image_folder}")

    # 이미지 읽어서 리스트에 저장
    frames = []
    for img_path in files:
        frames.append(imageio.imread(img_path))

    # MP4 생성: fps = 1 / duration
    fps = 1.0 / duration
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264')
    print(f"✔ Saved MP4: {output_path}")

if __name__ == "__main__":
    # 예시: debug/original 폴더의 이미지를 모아 0.5초 프레임으로 GIF 생성
    make_gif('debug/NAFVAE_8', 'debug/NAFVAE_8/animation.gif', duration=0.5)
    # make_mp4('debug/NAFVAE_8', 'debug/NAFVAE_8/animation.mp4', duration=0.5)