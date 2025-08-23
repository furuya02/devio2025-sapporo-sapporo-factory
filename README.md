# AHIRU Factory

## 動作確認




### Jetson

* Jetson AGX Orin 
* jetPack 6.2.1
* 要AWS認証情報

```
.
├── Jetson
│   ├── README-001.assembly.md
│   ├── README-002.docker-run.md
│   ├── README-003.check-device.md
│   ├── README-004.check-app.md
│   ├── home
│   │   ├── Dockerfile
│   │   ├── docker-build.sh
│   │   ├── docker-run.sh
│   │   └── src
│   │       ├── base_image.jpg
│   │       ├── base_image.py
│   │       ├── best.pt
│   │       ├── best.pt.keep
│   │       ├── check_center.py
│   │       ├── check_embedding.py
│   │       ├── check_inference.py
│   │       ├── check_mask.py
│   │       ├── check_sensor.py
│   │       ├── check_servo.py
│   │       ├── check_torch.py
│   │       ├── check_webcam.py
│   │       ├── detect_images.py
│   │       ├── duck_centerer.py
│   │       ├── embedding.py
│   │       ├── index.py
│   │       ├── sensor.py
│   │       └── servo.py
│   ├── images
│   │   ├── 001.png
│   │   ├── 002.png
│   │   ├── 003.png
│   │   └── chcec_servo.mp4
│   └── memo.md
├── LICENSE
└── README.md

```

### CDK

