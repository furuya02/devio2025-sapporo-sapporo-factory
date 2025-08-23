import cv2
import numpy as np


class BaseImage:
    def __init__(self, base_img_path: str):
        self.path = base_img_path
        self.image = cv2.imread(base_img_path)
        if self.image is None:
            raise FileNotFoundError(f"{base_img_path:} が見つかりません")

    def get_path(self):
        return self.path

    def create_blank_image(self):
        return np.zeros_like(self.image)

    def get_image(self):
        return self.image
