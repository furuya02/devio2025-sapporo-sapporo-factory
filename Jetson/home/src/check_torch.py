import torch
import sys
import cv2


def test_pytorch_cuda():
    """PyTorchのCUDA動作をテストする"""
    print("=== PyTorch CUDA テスト ===")
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"OpenCV バージョン: {cv2.__version__}")
    print("JetPack 6.2.1 (L4T R36.4.4) 環境での動作確認")

    # CUDA利用可能性をチェック
    print(f"CUDA利用可能: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA デバイス数: {torch.cuda.device_count()}")
        print(f"現在のCUDAデバイス: {torch.cuda.current_device()}")
        print(f"デバイス名: {torch.cuda.get_device_name()}")
        print(f"CUDAバージョン: {torch.version.cuda}")
        print(
            f"GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        # CUDNNチェック
        print(f"cuDNN利用可能: {torch.backends.cudnn.enabled}")
        if torch.backends.cudnn.enabled:
            print(f"cuDNNバージョン: {torch.backends.cudnn.version()}")

        # 簡単なテンソル操作
        try:
            print("\n--- CUDA テンソル操作テスト ---")
            device = torch.device("cuda")

            # CPU上でテンソルを作成
            x = torch.randn(100, 100)
            print(f"CPU テンソル作成成功: {x.shape}")

            # GPUに移動
            x_gpu = x.to(device)
            print(f"GPU テンソル移動成功: {x_gpu.device}")

            # GPU上で計算
            y_gpu = torch.mm(x_gpu, x_gpu.t())
            print(f"GPU 行列積計算成功: {y_gpu.shape}")

            # CPUに戻す
            y_cpu = y_gpu.cpu()
            print(f"CPU への移動成功: {y_cpu.device}")

            print("✅ PyTorch CUDA 動作テスト成功")
            return True

        except Exception as e:
            print(f"❌ CUDA テンソル操作エラー: {e}")
            return False
    else:
        print("❌ CUDAが利用できません")
        return False


def test_ultralytics_import():
    """Ultralyticsのインポートをテストする"""
    print("\n=== Ultralytics インポートテスト ===")
    try:
        from ultralytics import YOLO

        print("✅ Ultralytics インポート成功")

        # YOLOモデルの初期化（事前学習済みモデル）
        try:
            model = YOLO("yolov8n.pt")
            print("✅ YOLOv8n モデル読み込み成功")

            # モデルの情報表示
            print(f"モデル: {model.model}")
            print(f"デバイス: {model.device}")

            return True
        except Exception as e:
            print(f"⚠️ YOLO モデル読み込みエラー: {e}")
            return False

    except ImportError as e:
        print(f"❌ Ultralytics インポートエラー: {e}")
        return False


if __name__ == "__main__":
    print("Jetson PyTorch & Ultralytics 動作確認")
    print("=" * 50)

    pytorch_ok = test_pytorch_cuda()
    ultralytics_ok = test_ultralytics_import()

    print("\n" + "=" * 50)
    print("テスト結果:")
    print(f"PyTorch CUDA: {'✅ OK' if pytorch_ok else '❌ NG'}")
    print(f"Ultralytics: {'✅ OK' if ultralytics_ok else '❌ NG'}")

    if pytorch_ok and ultralytics_ok:
        print("\n🎉 すべてのテストが成功しました！")
        sys.exit(0)
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        sys.exit(1)
