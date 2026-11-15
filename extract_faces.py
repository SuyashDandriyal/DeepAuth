from facenet_pytorch import MTCNN
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

# Global MTCNN instance for fast face extraction
global_mtcnn = MTCNN(keep_all=False)

# Haar Cascade fallback detector
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else None
global_cascade = cv2.CascadeClassifier(_cascade_path) if _cascade_path and os.path.exists(_cascade_path) else None


def extract_face_from_image(img_pil: Image.Image):
    """Takes a PIL Image, extracts the face, and returns a cropped PIL Image (224x224)."""
    if img_pil is None:
        return None

    # 1. Try MTCNN
    try:
        face_tensor = global_mtcnn(img_pil)
        if face_tensor is not None:
            if face_tensor.min() < 0:
                face_tensor = (face_tensor + 1.0) / 2.0
            return to_pil(face_tensor.clamp(0, 1)).resize((224, 224))
    except Exception:
        pass

    # 2. Try OpenCV Haar Cascade
    try:
        if global_cascade is not None and not global_cascade.empty():
            cv_img = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = global_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                h_img, w_img = cv_img.shape[:2]
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + w + pad_x)
                y2 = min(h_img, y + h + pad_y)
                crop = cv_img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                return Image.fromarray(crop_rgb).resize((224, 224))
    except Exception:
        pass

    # 3. Fallback: Center crop & resize to (224, 224)
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return img_pil.crop((left, top, left + min_dim, top + min_dim)).resize((224, 224))


def extract_faces_from_video(video_path, output_dir, max_frames=60):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        total_frames = 300
        fps = 30

    duration = total_frames / fps
    target_frames = int(duration * 2)

    if target_frames > max_frames:
        target_frames = max_frames
    elif target_frames < 5:
        target_frames = min(5, max(1, total_frames))

    frame_interval = max(1, total_frames // max(1, target_frames))
    frames_extracted = 0
    all_read_frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            all_read_frames.append(img_pil)

            # Try MTCNN
            face_tensor = None
            try:
                face_tensor = global_mtcnn(img_pil)
            except Exception:
                pass

            if face_tensor is not None:
                if face_tensor.min() < 0:
                    face_tensor = (face_tensor + 1.0) / 2.0
                face_img = to_pil(face_tensor.clamp(0, 1)).resize((224, 224))
                save_path = os.path.join(output_dir, f"frame{frames_extracted}.jpg")
                face_img.save(save_path)
                frames_extracted += 1
            else:
                # Try Haar cascade fallback
                if global_cascade is not None and not global_cascade.empty():
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = global_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        pad_x = int(w * 0.15)
                        pad_y = int(h * 0.15)
                        h_img, w_img = frame.shape[:2]
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(w_img, x + w + pad_x)
                        y2 = min(h_img, y + h + pad_y)
                        crop = frame[y1:y2, x1:x2]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        face_img = Image.fromarray(crop_rgb).resize((224, 224))
                        save_path = os.path.join(output_dir, f"frame{frames_extracted}.jpg")
                        face_img.save(save_path)
                        frames_extracted += 1

            if frames_extracted >= target_frames:
                break

    cap.release()

    # Fallback if no faces were detected in the video:
    # Save up to 5 evenly spaced center-cropped frames
    if frames_extracted == 0 and len(all_read_frames) > 0:
        step = max(1, len(all_read_frames) // 5)
        for idx, sample_img in enumerate(all_read_frames[::step][:5]):
            w, h = sample_img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            crop = sample_img.crop((left, top, left + min_dim, top + min_dim)).resize((224, 224))
            save_path = os.path.join(output_dir, f"frame{idx}.jpg")
            crop.save(save_path)
