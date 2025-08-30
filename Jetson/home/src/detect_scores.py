import os
import shutil
import cv2


class DetectScores:
    def __init__(self, detect_score_path: str):
        # フォルダを再作成
        if os.path.exists(detect_score_path):
            shutil.rmtree(detect_score_path)
        os.makedirs(detect_score_path)
        self.detect_score_path = detect_score_path
        self.score_counter = 0

    def save_score(self, score, image, THRESHOLD):
        judge = "OK" if score >= THRESHOLD else "NG"
        save_path = os.path.join(
            self.detect_score_path, f"{self.score_counter:010d}_{judge}_{score:.3f}.jpg"
        )
        self.score_counter += 1
        if self.score_counter > 10000000:
            self.score_counter = 0
        cv2.imwrite(save_path, image)
        return save_path
