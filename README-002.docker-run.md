##  docker run

```
$ export AWS_REGION=ap-northeast-1
$ export AWS_ACCESS_KEY_ID=xxxxxxxx
$ export AWS_SECRET_ACCESS_KEY=xxxxxxxx

$ env | grep AWS_
AWS_REGION=ap-northeast-1
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxx
AWS_ACCESS_KEY_ID=xxxxxxxxxx
```


```
$ ./docker-run.sh

# env | grep AWS_
AWS_REGION=ap-northeast-1
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxx
AWS_ACCESS_KEY_ID=xxxxxxxxxxxxx

# python3 -c "import cv2;print(cv2.__version__)"
4.11.0
# python3 -c "import torch;print(torch.__version__)"
2.4.0
# python3 -c "import torch;print(torch.cuda.is_available())"
True
# python3 -c "import ultralytics;print(ultralytics.__version__)"
8.3.169

```

```
#  python3 check_torch.py

=== PyTorch CUDA テスト ===
PyTorch バージョン: 2.4.0
OpenCV バージョン: 4.11.0
JetPack 6.2.1 (L4T R36.4.4) 環境での動作確認
CUDA利用可能: True
CUDA デバイス数: 1
現在のCUDAデバイス: 0
デバイス名: Orin
CUDAバージョン: 12.6
GPU メモリ: 30.0 GB
cuDNN利用可能: True
cuDNNバージョン: 90400

--- CUDA テンソル操作テスト ---
CPU テンソル作成成功: torch.Size([100, 100])
GPU テンソル移動成功: cuda:0
GPU 行列積計算成功: torch.Size([100, 100])
CPU への移動成功: cpu
✅ PyTorch CUDA 動作テスト成功

=== Ultralytics インポートテスト ===
✅ Ultralytics インポート成功
✅ YOLOv8n モデル読み込み成功

テスト結果:
PyTorch CUDA: ✅ OK
Ultralytics: ✅ OK

🎉 すべてのテストが成功しました！

```

