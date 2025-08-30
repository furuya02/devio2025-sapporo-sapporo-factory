# -*- coding: utf-8 -*-
import boto3
import json
import time
import random
import os
from typing import Dict, Any
from datetime import datetime

# AWS設定
IOT_DATA_CLIENT = boto3.client("iot-data", region_name="ap-northeast-1")
S3_CLIENT = boto3.client("s3", region_name="ap-northeast-1")
DYNAMODB_CLIENT = boto3.client("dynamodb", region_name="ap-northeast-1")
MQTT_TOPIC = "duck-factory/line_1"
LINE_NAME = "Production Line 1"
IMAGE_NAME = "base_image.jpg"
S3_BUCKET = "duck-factory-229914322323"
DYNAMODB_TABLE = "duck-factory-table"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), IMAGE_NAME)


def generate_sensor_data(s3_key: str) -> Dict[str, Any]:
    """センサーデータを生成する"""
    current_time = datetime.now()
    device_id = "jetson-duck-factory-001"
    
    # ファクトリーのセンサーデータを模擬
    sensor_data = {
        "messageTime": int(current_time.timestamp() * 1000),
        "deviceId": device_id,
        "lineName": LINE_NAME,
        "qualityScore": round(random.uniform(20.0, 25.0), 2),
        "status": random.choice(["good", "defective"]),
        "timestamp": current_time.isoformat(),
        "imageS3Key": s3_key,
        "datePartition": current_time.strftime("%Y-%m-%d"),  # For daily partitioning
        "ttl": int((current_time.timestamp() + (365 * 24 * 3600)))  # TTL: 1年後
    }
    return sensor_data


def publish_to_iot(data: Dict[str, Any]) -> bool:
    """AWS IoT Coreにメッセージをパブリッシュ"""
    try:
        IOT_DATA_CLIENT.publish(
            topic=MQTT_TOPIC, qos=1, payload=json.dumps(data, ensure_ascii=False)
        )
        print(f"✓ メッセージ送信成功: {data['messageTime']}")
        return True
    except Exception as e:
        print(f"✗ メッセージ送信失敗: {e}")
        return False


def upload_image_to_s3() -> str:
    """base_image.jpgをS3にアップロード"""
    try:
        if not os.path.exists(IMAGE_PATH):
            print(f"✗ 画像ファイルが見つかりません: {IMAGE_PATH}")
            return ""

        # タイムスタンプ付きのキー名を生成
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        s3_key = f"base_images/{timestamp}.jpg"

        # S3にアップロード
        S3_CLIENT.upload_file(
            IMAGE_PATH,
            S3_BUCKET,
            s3_key,
            ExtraArgs={
                "ContentType": "image/jpeg",
                "Metadata": {
                    "upload_time": current_time.isoformat(),
                    "device_id": "jetson-duck-factory-001",
                    "line_name": LINE_NAME,
                },
            },
        )
        print(f"✓ S3アップロード成功: s3://{S3_BUCKET}/{s3_key}")
        return s3_key
    except Exception as e:
        print(f"✗ S3アップロード失敗: {e}")
        return ""


def save_to_dynamodb(data: Dict[str, Any]) -> bool:
    """センサーデータをDynamoDBに保存"""
    try:
        # DynamoDB形式に変換
        item = {}
        for key, value in data.items():
            if isinstance(value, str):
                item[key] = {"S": value}
            elif isinstance(value, (int, float)):
                item[key] = {"N": str(value)}
            else:
                item[key] = {"S": str(value)}
        
        # DynamoDBに書き込み
        DYNAMODB_CLIENT.put_item(
            TableName=DYNAMODB_TABLE,
            Item=item
        )
        print(f"✓ DynamoDB保存成功: {data['deviceId']} - {data['timestamp']}")
        return True
    except Exception as e:
        print(f"✗ DynamoDB保存失敗: {e}")
        return False


def main():
    """定期的にIoTデータをパブリッシュし、DynamoDBに保存、画像をS3にアップロードするメインループ"""
    print(f"AWS IoT Core + DynamoDB + S3 定期送信開始")
    print(f"IoTトピック: {MQTT_TOPIC}")
    print(f"DynamoDBテーブル: {DYNAMODB_TABLE}")
    print(f"S3バケット: {S3_BUCKET}")
    print(f"画像パス: {IMAGE_PATH}")
    print(f"送信間隔: 10秒")
    print("Ctrl+Cで停止\n")

    cycle_count = 0

    try:
        while True:
            cycle_count += 1

            # 画像をS3にアップロード
            print("📸 画像アップロード実行中...")
            s3_key = upload_image_to_s3()

            # センサーデータ生成
            sensor_data = generate_sensor_data(s3_key)

            # データ表示
            print(f"サイクル {cycle_count} - 時刻: {sensor_data['timestamp']}")
            print(f"デバイス: {sensor_data['deviceId']}")
            print(f"品質スコア: {sensor_data['qualityScore']}")
            print(f"ステータス: {sensor_data['status']}")
            print(f"画像: {sensor_data['imageS3Key']}")

            # IoT Coreにパブリッシュ
            iot_success = publish_to_iot(sensor_data)
            
            # DynamoDBに保存
            db_success = save_to_dynamodb(sensor_data)

            # 結果表示
            results = []
            if iot_success:
                results.append("IoT Core✓")
            else:
                results.append("IoT Core✗")
                
            if db_success:
                results.append("DynamoDB✓")
            else:
                results.append("DynamoDB✗")
                
            if s3_key:
                results.append("S3✓")
            else:
                results.append("S3✗")

            print(f"→ 送信結果: {' | '.join(results)}\n")

            # 10秒待機
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n定期送信を停止しました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
