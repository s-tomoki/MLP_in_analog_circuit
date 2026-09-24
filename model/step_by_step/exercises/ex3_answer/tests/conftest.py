"""pytest の共通設定。

tests/ 以下のテストを実行する前に pytest が自動で読み込む。
- src/ を import パスに加え、テストから `import nn_core` などができるようにする。
- 実データを使うテストに付けるマーカー `smoke` を登録する
  (例: `pytest -m "not smoke"` でユニットテストだけを実行)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: 実際の MNIST を使うスモークテスト(データのダウンロードが必要)"
    )
