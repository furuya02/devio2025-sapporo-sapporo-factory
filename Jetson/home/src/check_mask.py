# 参考にさせて頂きました
# https://github.com/ultralytics/ultralytics/issues/561

from ultralytics import YOLO
import numpy as np
import cv2

MODEL_PATH = "./best.pt"
CONF_MINIMUM = 0.96
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

COLORS = [(0, 0, 200)]
LINE_WIDTH = 5
FONT_SCALE = 1.5
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 5


def get_detect_img(image, target_mask):
    """マスク領域のみ元画像を残し、他を黒塗りにする"""
    if target_mask is not None:
        h, w, _ = image.shape
        detect_img = np.where(target_mask[:h, :w, np.newaxis], image, (0, 0, 0))
        return detect_img.astype(np.uint8)
    return np.zeros_like(image)


def create_mask(results, frame, conf_minimum=CONF_MINIMUM):
    """検出結果から信頼度が高い1つのアヒルのマスク画像を返す"""
    result = results[0]
    if result.masks is not None:
        for r in results:
            boxes = r.boxes
            conf_list = r.boxes.conf.tolist()
        if len(conf_list) == 1 and conf_list[0] > conf_minimum:
            for seg, _ in zip(result.masks.data.cpu().numpy(), boxes):
                print(f"conf_list[0]={conf_list[0]}")
                return get_detect_img(frame, seg)
    return None


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise IOError("カメラが開けません")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise IOError("カメラから画像が取得できません")
    print(f"frame.shape: {frame.shape}")
    print(f"Camera resolution is sufficient: {frame.shape}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise IOError("カメラから画像が取得できません")
            results = model(frame, conf=0.80, iou=0.5)
            mask_img = create_mask(results, frame)
            if mask_img is None:
                mask_img = np.zeros_like(frame)
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            mask_img_resized = cv2.resize(mask_img, (FRAME_WIDTH, FRAME_HEIGHT))
            merge_img = np.hstack((frame_resized, mask_img_resized))
            cv2.imshow("YOLO", merge_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
