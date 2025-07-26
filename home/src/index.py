import cv2
import numpy as np
import threading
from duck_centerer import DuckCenterer
from embedding import Embedding
from detect_images import DetectImages
from base_image import BaseImage
from sensor import Sensor
from servo import Servo
import boto3
import os
import time
import json

s3_client = boto3.client("s3", region_name="ap-northeast-1")
iot_data_client = boto3.client("iot-data", region_name="ap-northeast-1")
BUCKET_NAME = ""
PREFIX = "img/"
CLOUD_FRONT_URL = ""
MQTT_TOPIC = ""
LINE_NAME = ""
EQUIPMENT_NAME = ""


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
EMBEDDING_THRESHOLD = 0.94


def adjust_timing(message: str, sec: float) -> None:
    print(f"{message} {sec} 秒待機")
    threading.Event().wait(sec)


def process_sensor_event(embedding, save_image_path, sensor, servo):
    embedding_score = embedding.compare(save_image_path)
    print(f"embedding_score: {embedding_score:.4f}")
    adjust_timing("センサーのチャタリング防止のためのウエイト", 1.0)
    sensor.reset()
    adjust_timing("排除バーのタイミング調整", 2.0)
    # if embedding_score < EMBEDDING_THRESHOLD:
    #     print("アヒルを排除します")
    #     servo.close_gate()
    ##################################
    # 判定結果をクラウドに送信するなら、ここ
    # 判定画像のパス（save_image_path）
    # 判定結果(embedding_score < EMBEDDING_THRESHOLD)
    ##################################
    # imageUrl = ""
    # decision = 0
    # if embedding_score < EMBEDDING_THRESHOLD:
    #     # エラー画像を送る
    #     f = open(save_image_path, "rb")
    #     basename = os.path.basename(save_image_path)
    #     s3_client.put_object(Bucket=BUCKET_NAME, Key=f"{PREFIX}{basename}", Body=f)
    #     imageUrl = f"{CLOUD_FRONT_URL}{PREFIX}{basename}"
    #     decision = 1

    # payload = {
    #     "messageTime": int(time.time() * 1000),
    #     "lineName": LINE_NAME,
    #     "equipmentName": EQUIPMENT_NAME,
    #     "imageUrl": imageUrl,
    #     "decision": decision,
    # }
    # iot_data_client.publish(
    #     topic=MQTT_TOPIC,
    #     qos=1,
    #     payload=json.dumps(payload),
    # )


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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise IOError("カメラから画像が取得できません")
            duck_img = centerer.get_centered_duck(frame)
            save_image_path = detect_images.save_image(duck_img)
            merge_img2 = np.hstack((base_image.get_image(), duck_img))
            cv2.imshow("embedding", merge_img2)
            cv2.imshow("frame", frame)

            if sensor.check() == "on":
                t = threading.Thread(
                    target=process_sensor_event,
                    args=(embedding, save_image_path, sensor, servo),
                )
                t.start()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
