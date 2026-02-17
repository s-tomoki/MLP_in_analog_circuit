import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback

# ログ設定
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# ログファイルハンドラ追加
log_filename = f"nand_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)


class DebugCallback(Callback):
    """エポック毎に損失と重み・バイアスをログに記録"""

    def __init__(self, logger, activation_name):
        super().__init__()
        self.logger = logger
        self.activation_name = activation_name

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if epoch == 0 or epoch % 100 == 0 or epoch == self.params["epochs"] - 1:
            loss = logs.get("loss", "N/A")
            acc = logs.get("accuracy", "N/A")
            weights, biases = self.model.layers[0].get_weights()
            self.logger.info(
                f"[{self.activation_name}] Epoch {epoch+1}/{self.params['epochs']}: "
                f"loss={loss:.6f}, accuracy={acc:.4f}, "
                f"weights={weights.flatten()}, bias={biases.flatten()}"
            )


def sigmoid_ste_activation(x):
    # STE 実装（推奨パターン）:
    # - 順伝播は厳密なステップ関数 (0/1)
    # - 逆伝播はシグモイドの微分を用いるために `stop_gradient` トリックを使う
    # これにより順伝播は離散化される一方で、勾配は滑らかなシグモイド由来になる
    s = tf.sigmoid(x)
    step = tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))
    # forward = step, backward = sigmoid の勾配
    rt = step + s - tf.stop_gradient(s)
    return rt


def train(activation=sigmoid, activation_name="sigmoid"):
    # NANDゲートの学習データ
    X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_train = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    logger.info(f"=== Training with {activation_name} activation ===")

    # 単純なモデルを定義 (入力層1つ, 出力層1つ)
    model = Sequential()
    model.add(Dense(1, input_shape=(2,)))
    model.add(Lambda(lambda z: activation(z)))

    # モデルをコンパイル (損失関数: Binary Crossentropy, オプティマイザ: Adam)
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(f"Model summary: {model.count_params()} parameters")

    # 学習の実行
    # 単純パーセプトロンの場合、学習データが少ないためエポック数を多めに設定
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        verbose=0,
        callbacks=[TqdmCallback(verbose=1), DebugCallback(logger, activation_name)],
    )

    # 結果の確認
    predictions = model.predict(X_train)
    # しきい値 0.5 で2値化
    predicted_classes = (predictions > 0.5).astype(int)

    print("\n--- 学習結果 ---")
    logger.info("=== Final Training Results ===")
    for i in range(len(X_train)):
        result_str = (
            f"入力: {X_train[i]}, 予測出力: {predicted_classes[i][0]}"
            + f" (確率: {predictions[i][0]:.4f})"
        )
        print(result_str)
        logger.info(result_str)

    # 学習後の重みとバイアスを表示 (参考情報)
    weights, biases = model.layers[0].get_weights()
    print(f"\n学習後の重み: {weights.flatten()}")
    print(f"学習後のバイアス: {biases.flatten()}")
    logger.info(f"Final weights: {weights.flatten()}")
    logger.info(f"Final bias: {biases.flatten()}")

    # CSV ファイルに重みとバイアスを保存
    np.savetxt(
        f"nand_weights_{activation_name}.csv",
        weights,
        delimiter=",",
        header="weight_in0,weight_in1",
        comments="",
    )
    np.savetxt(
        f"nand_biases_{activation_name}.csv", biases, delimiter=",", header="bias", comments=""
    )

    print(f"\n重みをCSVファイルに保存しました: nand_weights_{activation_name}.csv")
    print(f"バイアスをCSVファイルに保存しました: nand_biases_{activation_name}.csv")
    logger.info(f"Weights saved to: nand_weights_{activation_name}.csv")
    logger.info(f"Bias saved to: nand_biases_{activation_name}.csv")
    return None


def main():
    logger.info("=== NAND Gate Neural Network Training Start ===")
    print("--- 学習開始: Sigmoid Activation ---")
    train(activation=sigmoid, activation_name="sigmoid")
    print("--- 学習開始: Sigmoid STE Activation (順伝搬：Step / 逆伝搬：Sigmoid) ---")
    train(activation=sigmoid_ste_activation, activation_name="sigmoid_ste")
    logger.info("=== All Training Complete ===")
    logger.info(f"Log file saved to: {log_filename}")


if __name__ == "__main__":
    main()
