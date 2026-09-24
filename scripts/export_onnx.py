"""
学習済み GCViT-Tiny（犬猫37品種分類）を PyTorch から ONNX へ変換し、
変換前後で出力が一致することを検証するスクリプト。

ONNX変換は、失敗しても例外を出さず「結果だけが静かにずれる」ことがある。
そのため変換と検証を必ず1つの工程として扱い、数値一致を確認できない場合は
ONNXファイルを出力せずに異常終了させる。

使い方（ローカル / Colab 共通）:
    python scripts/export_onnx.py \
        --weights gcvit_tiny_best.pth \
        --outdir .

Colab のセル内から実行する場合:
    !python scripts/export_onnx.py --weights /content/gcvit_tiny_best.pth --outdir /content

出力:
    model.onnx   ... Lambda のコンテナイメージに同梱する推論モデル
    labels.json  ... 品種ラベル（クラスID順）
    meta.json    ... 前処理の設定値（推論側と学習側の齟齬を防ぐため）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

MODEL_NAME = "gcvit_tiny"
INPUT_SIZE = 224
OPSET = 17

# Oxford-IIIT Pet の37品種。ImageFolder のクラス順（辞書順）と一致させる。
# 学習時の dataset.classes をそのまま貼り付けたもの。
CLASS_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair",
    "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
    "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "Siamese", "Sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier",
]

# ImageNet の標準値。推論側（lambda_function.py）と必ず一致させること。
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 数値一致の許容誤差。
# ONNX Runtime と PyTorch では演算の順序や使用カーネルが異なるため、
# 完全一致は期待できない。分類タスクで順位が入れ替わらない水準として設定。
RTOL = 1e-3
ATOL = 1e-5


# ---------------------------------------------------------------------------
# モデルの復元
# ---------------------------------------------------------------------------

def build_model(weights_path: Path, num_classes: int) -> torch.nn.Module:
    """学習時と同じ構造のモデルを作り、保存した重みを読み込む。

    ONNX変換には「モデルの構造」と「学習した重み」の両方が必要になる。
    .pth に入っているのは重みだけなので、構造は timm で作り直す。
    """
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

    state = torch.load(weights_path, map_location="cpu")
    # {"model": ..., "epoch": ...} のような形で保存されている場合に対応
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    # DataParallel 経由で保存された場合に付く "module." を除去
    state = {k.removeprefix("module."): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"       missing 例: {missing[:5]}")
        if unexpected:
            print(f"       unexpected 例: {unexpected[:5]}")
        # 分類ヘッドだけが欠けている場合は学習結果が反映されないため致命的
        if any("head" in k or "fc" in k for k in missing):
            sys.exit("[error] 分類ヘッドの重みが読み込めていません。--weights を確認してください。")

    # BatchNorm や Dropout を推論モードに切り替える。
    # これを忘れると変換後のモデルの出力がずれる。
    model.eval()
    return model


# ---------------------------------------------------------------------------
# ONNX への変換
# ---------------------------------------------------------------------------

def export(model: torch.nn.Module, onnx_path: Path) -> None:
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        # バッチ次元だけ可変にしておく。複数画像をまとめて推論する拡張に備える。
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        # PyTorch 2.6 以降は dynamo ベースのエクスポータが既定になり、
        # onnxscript の追加インストールが必要になる。
        # 従来の TorchScript ベースで安定して変換するため明示的に無効化する。
        dynamo=False,
    )
    print(f"[ok] 変換しました: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------

def verify(model: torch.nn.Module, onnx_path: Path, trials: int = 3) -> None:
    """同じ入力を PyTorch と ONNX Runtime に与え、出力の数値一致を確認する。"""
    import onnx
    import onnxruntime as ort

    # グラフ構造そのものの妥当性チェック
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print("[ok] ONNXグラフの構造チェックを通過")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for i in range(trials):
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

        with torch.no_grad():
            expected = model(x).numpy()
        actual = session.run(None, {input_name: x.numpy()})[0]

        # 1. 数値そのものの一致
        np.testing.assert_allclose(expected, actual, rtol=RTOL, atol=ATOL)

        # 2. 予測クラスの一致（分類タスクとして本当に重要なのはこちら）
        assert expected.argmax(1) == actual.argmax(1), "予測クラスが一致しません"

        diff = np.abs(expected - actual).max()
        print(f"[ok] 検証 {i + 1}/{trials}: 最大絶対誤差 {diff:.3e} / 予測クラス一致")

    print(f"[ok] 数値一致を確認（rtol={RTOL}, atol={ATOL}）")


# ---------------------------------------------------------------------------
# 付随ファイルの出力
# ---------------------------------------------------------------------------

def write_metadata(outdir: Path, num_classes: int) -> None:
    (outdir / "labels.json").write_text(
        json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "input_size": INPUT_SIZE,
        "mean": MEAN,
        "std": STD,
        "opset": OPSET,
    }
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[ok] labels.json / meta.json を出力しました")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GCViT-Tiny を ONNX に変換して検証する")
    parser.add_argument("--weights", required=True, type=Path, help="学習済み重み（.pth）")
    parser.add_argument("--outdir", default=Path("."), type=Path, help="出力先ディレクトリ")
    parser.add_argument("--trials", default=3, type=int, help="検証に使うランダム入力の数")
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"[error] 重みファイルが見つかりません: {args.weights}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.outdir / "model.onnx"
    num_classes = len(CLASS_NAMES)

    print(f"[info] {MODEL_NAME} / {num_classes}クラス / 入力 {INPUT_SIZE}x{INPUT_SIZE}")

    model = build_model(args.weights, num_classes)
    export(model, onnx_path)

    try:
        verify(model, onnx_path, trials=args.trials)
    except AssertionError as e:
        # 一致しないモデルを残すと、そのままデプロイされる危険がある
        onnx_path.unlink(missing_ok=True)
        sys.exit(f"[error] 数値一致の検証に失敗したため model.onnx を削除しました\n{e}")

    write_metadata(args.outdir, num_classes)
    print("\n完了。model.onnx と labels.json を Dockerfile と同じ階層に配置してください。")


if __name__ == "__main__":
    main()
