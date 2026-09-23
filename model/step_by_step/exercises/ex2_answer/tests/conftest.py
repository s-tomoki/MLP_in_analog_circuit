"""pytest の共通設定。

tests/ 以下のテストを実行する前に pytest が自動で読み込む。
- src/ を import パスに加え、テストから `import nn_core` できるようにする。
- 段階ごとにテストを選んで実行できるよう、マーカー stage1〜stage4 を登録する
  (例: `pytest -m stage1`)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

STAGES = {
    "stage1": "段階1: シグモイド関数と学習(誤差逆伝播法・勾配降下法)",
    "stage2": "段階2: L2 正則化",
    "stage3": "段階3: ReLU 関数",
    "stage4": "段階4: ステップ関数(Straight-Through Estimator)",
}


def pytest_configure(config):
    for name, description in STAGES.items():
        config.addinivalue_line("markers", f"{name}: {description}")
