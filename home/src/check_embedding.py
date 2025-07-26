# 参考にさせて頂きました
# https://github.com/ultralytics/ultralytics/issues/561

import os
import shutil
import json
import cv2
import boto3
import base64
import numpy as np
from ultralytics import YOLO
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


class DuckCenterer:
    def __init__(
        self,
        model_path,
        conf_minimum,
        frame_width,
        frame_height,
        center_image_width,
        center_image_height,
        conf_threshold,
        iou_threshold,
    ):
        self.model = YOLO(model_path)
        self.conf_minimum = conf_minimum
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_image_width = center_image_width
        self.center_image_height = center_image_height
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def get_largest_contour(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, binary
        largest = max(contours, key=cv2.contourArea)
        return largest, binary

    def move_target_to_center(self, org_img):
        oh = self.center_image_height
        ow = self.center_image_width
        center_img = np.zeros((oh, ow, 3), dtype=np.uint8)

        contour, _ = self.get_largest_contour(org_img)
        if contour is None:
            return center_img
        x, y, w, h = cv2.boundingRect(contour)
        print(f"x:{x}  y:{y} w:{w} h:{h}")
        target_img = org_img[y : y + h, x : x + w]
        if target_img.size == 0 or len(target_img.shape) != 3:
            return center_img
        th, tw = target_img.shape[:2]
        print(f"th:{th} tw:{tw}")
        dx = int((ow - tw) / 2)
        dy = int((oh - th) / 2)
        print(f"dx:{dx} dy:{dy}")
        center_img[dy : dy + th, dx : dx + tw] = target_img
        return center_img

    def get_detect_img(self, image, target_mask):
        if target_mask is not None:
            h, w, _ = image.shape
            detect_img = np.where(target_mask[:h, :w, np.newaxis], image, (0, 0, 0))
            return detect_img.astype(np.uint8)
        return np.zeros_like(image)

    def create_mask(self, results, frame):
        result = results[0]
        if result.masks is not None:
            for r in results:
                boxes = r.boxes
                conf_list = r.boxes.conf.tolist()
            if len(conf_list) == 1 and conf_list[0] > self.conf_minimum:
                for seg, _ in zip(result.masks.data.cpu().numpy(), boxes):
                    seg = self.remove_noise(frame, seg)
                    return self.get_detect_img(frame, seg)
        return None

    def get_centered_duck(self, frame):
        results = self.model(frame, conf=0.80, iou=0.5)
        mask_img = self.create_mask(results, frame)
        if mask_img is None:
            mask_img = np.zeros_like(frame)
        mask_img_resized = cv2.resize(mask_img, (self.frame_width, self.frame_height))
        center_img = self.move_target_to_center(mask_img_resized)
        if center_img.ndim == 2:
            center_img = cv2.cvtColor(center_img, cv2.COLOR_GRAY2BGR)
        return center_img

    # ノイズ除去
    def remove_noise(self, image, mask):
        # 2値画像（白及び黒）を生成する
        height, width, _ = image.shape
        tmp_black_image = np.full(np.array([height, width, 1]), 0, dtype=np.uint8)
        tmp_white_image = np.full(np.array([height, width, 1]), 255, dtype=np.uint8)
        # マスクによって黒画像の上に白を描画する
        tmp_black_image[:] = np.where(
            mask[:height, :width, np.newaxis] == True, tmp_white_image, tmp_black_image
        )

        # 領域の穴埋め（オプション: 穴がある場合）
        tmp_black_image = cv2.morphologyEx(
            tmp_black_image, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8)
        )

        # 輪郭の取得
        contours, _ = cv2.findContours(
            tmp_black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 最も面積が大きい輪郭を選択
        max_contours = max(contours, key=lambda x: cv2.contourArea(x))
        # 黒画面に一番大きい輪郭だけ塗りつぶして描画する
        black_image = np.full(np.array([height, width, 1]), 0, dtype=np.uint8)
        black_image = cv2.drawContours(
            black_image, [max_contours], -1, color=255, thickness=-1
        )
        # 輪郭を保存
        self._contours, _ = cv2.findContours(
            black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # マスクを作り直す
        new_mask = np.full(np.array([height, width, 1]), False, dtype=np.bool_)
        new_mask[::] = np.where(black_image[:height, :width] == 0, False, True)
        new_mask = np.squeeze(new_mask)

        return new_mask


class Embedding:
    def __init__(
        self, base_img_path: str, dimensions: int = 1024, region: str = "us-east-1"
    ):
        self.dimensions = dimensions
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )
        self.base_embedding = self._create_embedding(base_img_path)
        print(f"base_embedding.len: {len(self.base_embedding)}")

    def _create_embedding(self, img_path) -> List[float]:
        with open(img_path, "rb") as image:
            body = image.read()
        response = self.bedrock.invoke_model(
            body=json.dumps(
                {
                    "inputImage": base64.b64encode(body).decode("utf8"),
                    "embeddingConfig": {"outputEmbeddingLength": self.dimensions},
                }
            ),
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")

    def compare(self, target_img_path) -> float:
        target_embedding = self._create_embedding(target_img_path)
        cosine = cosine_similarity([self.base_embedding], [target_embedding])
        return cosine[0][0]


# --- サンプル実行部 ---
def main():
    CAMERA_ID = 0
    # カメラ画像のサイズ
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    # 切り出す画像のサイズ
    CENTER_IMAGE_WIDTH = 300
    CENTER_IMAGE_HEIGHT = 260
    # 物体検出モデル
    MODEL_PATH = "./best.pt"
    CONF_THRESHOLD = 0.75
    CONF_MINIMUM = 0.75
    IOU_THRESHOLD = 0.5
    # Embeddingの設定
    BASE_IMAGE_PATH = "./base_image.jpg"
    # 物体検出画像の保存先
    DETECT_IMAGE_PATH = "./detect_image"
    # 判定基準
    EMBEDDING_THRESHOLD = 0.93

    # フォルダを再作成
    if os.path.exists(DETECT_IMAGE_PATH):
        shutil.rmtree(DETECT_IMAGE_PATH)
    os.makedirs(DETECT_IMAGE_PATH)
    base_image = cv2.imread(BASE_IMAGE_PATH)
    if base_image is None:
        raise FileNotFoundError(f"{BASE_IMAGE_PATH} が見つかりません")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise IOError("カメラが開けません")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise IOError("カメラから画像が取得できません")
    centerer = DuckCenterer(
        model_path=MODEL_PATH,
        conf_minimum=CONF_MINIMUM,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        center_image_width=CENTER_IMAGE_WIDTH,
        center_image_height=CENTER_IMAGE_HEIGHT,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
    embedding = Embedding(base_img_path=BASE_IMAGE_PATH)
    image_counter = 0
    frame_counter = 0
    embedding_score = 0.0
    center_img = np.zeros_like(base_image)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise IOError("カメラから画像が取得できません")
            center_img = centerer.get_centered_duck(frame)
            frame_counter += 1
            color = (0, 255, 0)
            if frame_counter % 12 == 0:
                detect_img_path = os.path.join(
                    DETECT_IMAGE_PATH, f"{image_counter:010d}.jpg"
                )
                image_counter += 1
                if image_counter > 10000000:
                    image_counter = 0
                print(f"detect_img_path: {detect_img_path}")
                cv2.imwrite(detect_img_path, center_img)
                embedding_score = embedding.compare(detect_img_path)
                print(f"embedding_score: {embedding_score:.4f}")

            if embedding_score < EMBEDDING_THRESHOLD:
                color = (0, 0, 255)

            cv2.putText(
                center_img,
                f"Score: {embedding_score:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            merge_img2 = np.hstack((base_image, center_img))
            cv2.imshow("embedding", merge_img2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
