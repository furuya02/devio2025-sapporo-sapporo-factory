import os
import shutil
import cv2


class DetectImages:
    def __init__(self, detect_img_path: str):
        # フォルダを再作成
        if os.path.exists(detect_img_path):
            shutil.rmtree(detect_img_path)
        os.makedirs(detect_img_path)
        self.detect_img_path = detect_img_path
        self.image_counter = 0

    def save_image(self, image):
        save_path = os.path.join(self.detect_img_path, f"{self.image_counter:010d}.jpg")
        self.image_counter += 1
        if self.image_counter > 10000000:
            self.image_counter = 0
        cv2.imwrite(save_path, image)
        return save_path
