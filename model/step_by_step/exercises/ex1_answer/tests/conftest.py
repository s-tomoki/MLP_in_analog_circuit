"""pytest の共通設定。

tests/ 以下のテストを実行する前に pytest が自動で読み込む。
src/ を import パスに加え、テストから `import nn_core` できるようにする。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
