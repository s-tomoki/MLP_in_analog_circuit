import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback


def sigmoid_ste_activation(x):
    # STE 実装（推奨パターン）:
    # - 順伝播は厳密なステップ関数 (0/1)
    # - 逆伝播はシグモイドの微分を用いるために `stop_gradient` トリックを使う
    # これにより順伝播は離散化される一方で、勾配は滑らかなシグモイド由来になる
    s = tf.sigmoid(x)
    step = tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))
    # forward = step, backward = sigmoid の勾配
    return step + s - tf.stop_gradient(s)


def train(activation=sigmoid, activation_name="sigmoid"):
    # NANDゲートの学習データ
    X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_train = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    # 単純なモデルを定義 (入力層1つ, 出力層1つ)
    model = Sequential()
    model.add(Dense(1, input_shape=(2,)))
    model.add(Lambda(lambda z: activation(z)))

    # モデルをコンパイル (損失関数: Binary Crossentropy, オプティマイザ: SGD)
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 学習の実行
    # 単純パーセプトロンの場合、学習データが少ないためエポック数を多めに設定
    model.fit(X_train, y_train, epochs=1_000, verbose=0, callbacks=[TqdmCallback(verbose=1)])

    # 結果の確認
    predictions = model.predict(X_train)
    # しきい値 0.5 で2値化
    predicted_classes = (predictions > 0.5).astype(int)

    print("\n--- 学習結果 ---")
    for i in range(len(X_train)):
        print(
            f"入力: {X_train[i]}, 予測出力: {predicted_classes[i][0]}"
            + f" (確率: {predictions[i][0]:.4f})"
        )

    # 学習後の重みとバイアスを表示 (参考情報)
    weights, biases = model.layers[0].get_weights()
    print(f"\n学習後の重み: {weights.flatten()}")
    print(f"学習後のバイアス: {biases.flatten()}")

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
    return None


def main():
    print("--- 学習開始: Sigmoid Activation ---")
    train(activation=sigmoid, activation_name="sigmoid")
    print("--- 学習開始: ReLU Activation ---")
    train(activation=relu, activation_name="relu")
    print("--- 学習開始: Sigmoid STE Activation (順伝搬：Step / 逆伝搬：Sigmoid) ---")
    train(activation=sigmoid_ste_activation, activation_name="sigmoid_ste")


if __name__ == "__main__":
    main()
