import tensorflow as tf
import numpy as np

# NANDゲートの学習データ
X_train = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
y_train = np.array([[1.], [1.], [1.], [0.]], dtype=np.float32)

# 単純なモデルを定義 (入力層1つ, 出力層1つ)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

# モデルをコンパイル (損失関数: Binary Crossentropy, オプティマイザ: SGD)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 学習の実行
# 単純パーセプトロンの場合、学習データが少ないためエポック数を多めに設定
model.fit(X_train, y_train, epochs=1000, verbose=0)

# 結果の確認
predictions = model.predict(X_train)
# しきい値 0.5 で2値化
predicted_classes = (predictions > 0.5).astype(int)

print("\n--- 学習結果 ---")
for i in range(len(X_train)):
    print(f"入力: {X_train[i]}, 予測出力: {predicted_classes[i][0]} (確率: {predictions[i][0]:.4f})")

# 学習後の重みとバイアスを表示 (参考情報)
weights, biases = model.layers[0].get_weights()
print(f"\n学習後の重み: {weights.flatten()}")
print(f"学習後のバイアス: {biases.flatten()}")

