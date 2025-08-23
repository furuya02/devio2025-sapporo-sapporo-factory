from typing import List
import boto3
import base64
import json
from sklearn.metrics.pairwise import cosine_similarity


class Embedding:
    def __init__(
        self, base_img_path: str, dimensions: int = 1024, region: str = "us-east-1"
    ):
        self.dimensions = dimensions
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )
        self.base_embedding = self._create_embedding(base_img_path)
        print(f"base_embedding.len: {len(self.base_embedding)}")

    def _create_embedding(self, img_path) -> List[float]:
        with open(img_path, "rb") as image:
            body = image.read()
        response = self.bedrock.invoke_model(
            body=json.dumps(
                {
                    "inputImage": base64.b64encode(body).decode("utf8"),
                    "embeddingConfig": {"outputEmbeddingLength": self.dimensions},
                }
            ),
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")

    def compare(self, target_img_path) -> float:
        target_embedding = self._create_embedding(target_img_path)
        cosine = cosine_similarity([self.base_embedding], [target_embedding])
        return cosine[0][0]
