import cv2
import numpy as np
import threading
from duck_centerer import DuckCenterer
from embedding import Embedding
from detect_images import DetectImages
from detect_scores import DetectScores
from base_image import BaseImage
from sensor import Sensor
from servo import Servo
import boto3
import os
import time
import json

s3_client = boto3.client("s3", region_name="ap-northeast-1")
iot_data_client = boto3.client("iot-data", region_name="ap-northeast-1")
BUCKET_NAME = "duck-factory-229914322323"
PREFIX = "img/"
LINE_NAME = "line-01"  # ライン名を設定してください（例: "Line-01"）
MQTT_TOPIC = f"duck-factory/{LINE_NAME}"  # MQTT トピックを設定してください（例: "$aws/things/duck-factory/shadow/update")


CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_IMAGE_WIDTH = 300
CENTER_IMAGE_HEIGHT = 260
MODEL_PATH = "./best.pt"
CONF_THRESHOLD = 0.75
CONF_MINIMUM = 0.75
IOU_THRESHOLD = 0.5
BASE_IMAGE_PATH = "./base_image.jpg"
DETECT_IMAGE_PATH = "./detect_image"
DETECT_SCORE_PATH = "./detect_score"
EMBEDDING_THRESHOLD = 0.92


def worker_thread(embedding, save_image_path_list, detect_scores, sensor):
    best_score = 0
    best_image_path = None
    score_list = []
    for save_image_path in save_image_path_list:
        embedding_score = embedding.compare(save_image_path)
        print(f"embedding_score: {save_image_path} {embedding_score:.4f}")
        score_list.append(embedding_score)
        if embedding_score > best_score:
            best_score = embedding_score
            best_image_path = save_image_path

    detect_scores.save_score(
        best_score, cv2.imread(best_image_path), EMBEDDING_THRESHOLD
    )

    if best_score < EMBEDDING_THRESHOLD:
        print(f"\033[91mNG {best_image_path} {best_score:.4f}\033[0m")
    else:
        print(f"\033[96mOK {best_image_path} {best_score:.4f}\033[0m")

    # 画像送信
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge = "OK" if best_score >= EMBEDDING_THRESHOLD else "NG"
    s3_key = f"{PREFIX}{timestamp}_{judge}_{best_score:.3f}.jpg"

    with open(best_image_path, "rb") as f:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=f)
    imageUrl = s3_key
    # MQTT送信
    iot_data_client.publish(
        topic=MQTT_TOPIC,
        qos=1,
        payload=json.dumps(
            {
                "state": {
                    "reported": {
                        "score": best_score,
                        "judge": "OK" if best_score >= EMBEDDING_THRESHOLD else "NG",
                        "imageUrl": imageUrl,
                    }
                }
            }
        ),
    )

    time.sleep(2)
    sensor.reset()
    print("sensor off")


def main():

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise IOError("カメラが開けません")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise IOError("カメラから画像が取得できません")

    detect_images = DetectImages(detect_img_path=DETECT_IMAGE_PATH)
    detect_scores = DetectScores(detect_score_path=DETECT_SCORE_PATH)
    base_image = BaseImage(BASE_IMAGE_PATH)
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
    embedding = Embedding(base_img_path=base_image.get_path())
    sensor = Sensor()
    servo = Servo()
    duck_img = base_image.create_blank_image()

    shoot_counter = 0
    save_image_path_list = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise IOError("カメラから画像が取得できません")
            duck_img = centerer.get_centered_duck(frame)
            merge_img2 = np.hstack((base_image.get_image(), duck_img))
            cv2.imshow("embedding", merge_img2)
            cv2.imshow("frame", frame)

            if sensor.check() == "on":
                print("センサー ON")
                shoot_counter = 4
            if shoot_counter > 0:
                save_image_path = detect_images.save_image(duck_img)
                save_image_path_list.append(save_image_path)
                print(f"shooting...{save_image_path}")

                shoot_counter -= 1
                if shoot_counter == 0:
                    t = threading.Thread(
                        target=worker_thread,
                        args=(embedding, save_image_path_list, detect_scores, sensor),
                    )
                    t.start()
                    save_image_path_list = []

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
