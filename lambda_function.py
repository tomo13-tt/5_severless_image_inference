import io
import json
import os
import time
import urllib.parse

import boto3
import numpy as np
import onnxruntime as ort
from PIL import Image

# =====================================================================
# ここから下（関数の外）は、コンテナが起動したとき「1回だけ」実行される。
# Lambdaは一度起動したコンテナをしばらく使い回すので、
# 2回目以降の呼び出しではこの処理がスキップされ、その分だけ速くなる。
# =====================================================================

# モデルはイメージに同梱されているので、S3からのダウンロードは不要
TASK_ROOT = os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(__file__))
MODEL_PATH = os.path.join(TASK_ROOT, "model.onnx")
LABELS_PATH = os.path.join(TASK_ROOT, "labels.json")

_t0 = time.time()
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
with open(LABELS_PATH) as f:
    labels = json.load(f)
print(f"[init] モデル読み込み完了: {time.time() - _t0:.2f} 秒 / クラス数 {len(labels)}")

s3 = boto3.client("s3")

# 結果の保存先バケット（環境変数で渡す。ローカルテスト時は未設定でもよい）
BUCKET = os.environ.get("BUCKET_NAME")
RESULT_PREFIX = os.environ.get("RESULT_PREFIX", "results/")

# ImageNetの標準的な正規化パラメータ
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bytes):
    """画像のバイト列を、モデルが受け取れる形の配列に変換する"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    left = (256 - 224) // 2          # 中央を224x224で切り出す
    img = img.crop((left, left, left + 224, left + 224))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]   # (H,W,C) -> (1,C,H,W)
    return arr.astype(np.float32)


def softmax(x):
    """スコアを合計1の確率に変換する"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(image_bytes, top_k=5):
    """画像を分類して、上位k件の結果を返す。S3に依存しないのでローカルでも試せる"""
    t0 = time.time()
    x = preprocess(image_bytes)
    logits = session.run(None, {"input": x})[0][0]
    probs = softmax(logits)
    top = np.argsort(probs)[::-1][:top_k]
    return {
        "predictions": [
            {"label": labels[i], "score": round(float(probs[i]), 4)} for i in top
        ],
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }


def lambda_handler(event, context):
    """
    Lambdaが呼び出す入口。
    event の中に「どのバケットのどのファイルが追加されたか」が入っている。
    """
    results = []

    for record in event["Records"]:
        src_bucket = record["s3"]["bucket"]["name"]
        # キーはURLエンコードされているのでデコードする（日本語ファイル名対策）
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        print(f"[start] 処理開始: s3://{src_bucket}/{key}")

        # --- 1. 画像をS3から読む ---
        obj = s3.get_object(Bucket=src_bucket, Key=key)
        image_bytes = obj["Body"].read()

        # --- 2. 推論する ---
        result = predict(image_bytes)
        result["source_key"] = key
        result["request_id"] = context.aws_request_id
        print(f"[result] {json.dumps(result, ensure_ascii=False)}")

        # --- 3. 結果をS3に書き戻す ---
        filename = key.split("/")[-1]
        out_key = f"{RESULT_PREFIX}{filename}.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=out_key,
            Body=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"[done] 保存しました: s3://{BUCKET}/{out_key}")
        results.append(result)

    return {"statusCode": 200, "body": json.dumps(results, ensure_ascii=False)}
