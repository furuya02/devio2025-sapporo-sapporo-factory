# AHIRU Factory


## Jetson

* Copy Contents to Jetson

```
% scp -r Jetson/home/* sin@192.168.1.31:~/home2/2025.07.26_DevIO_SAPPORO_Ahiru_Factory/"
```

* Docker Start

```
$ ./docker-build.sh
$ ./docker-run.sh

root@ubuntu:/src#
```

* Check

```

```


 メインテーブル: duck-factory-table
  - パーティションキー: deviceId (デバイス単位でデータを分割)
  - ソートキー: timestamp (時系列ソート用)
  - TTL: ttl 属性で1年後の自動削除

  Global Secondary Indexes (GSI):
  1. LineNameTimeIndex: 生産ライン別での最新データ取得
  2. StatusTimeIndex: ステータス別での品質管理分析
  3. DateTimeIndex: 日別でのデータ集約

  📊 データ構造例

  {
    "deviceId": "jetson-duck-factory-001",
    "timestamp": "2024-01-15T10:30:45.123Z",
    "messageTime": 1705317045123,
    "lineName": "Production Line 1",
    "qualityScore": 23.45,
    "status": "good",
    "imageS3Key": "base_images/20240115_103045.jpg",
    "datePartition": "2024-01-15",
    "ttl": 1737939845
  }

  🔍 最新100件取得のクエリ方法

  デバイス別最新100件:
  # 特定デバイスの最新100件
  response = dynamodb.query(
      TableName="duck-factory-table",
      KeyConditionExpression="deviceId = :deviceId",
      ExpressionAttributeValues={":deviceId": {"S":
  "jetson-duck-factory-001"}},
      ScanIndexForward=False,  # 新しい順
      Limit=100
  )

  生産ライン別最新100件:
  # GSIを使用した生産ライン別クエリ
  response = dynamodb.query(
      TableName="duck-factory-table",
      IndexName="LineNameTimeIndex",
      KeyConditionExpression="lineName = :lineName",
      ExpressionAttributeValues={":lineName": {"S": "Production Line
   1"}},
      ScanIndexForward=False,
      Limit=100
  )

## CDK

