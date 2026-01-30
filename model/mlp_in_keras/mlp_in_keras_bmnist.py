# Ref: 
# https://machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api

# Imports
import os
from PIL import Image
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def visualize_dataset(X, Y, dirname='visualizations'):
# Create output directory for visualizations
    os.makedirs(dirname, exist_ok=True)

    # Visualize and save training samples
    x_width = int(np.sqrt(X.shape[1]))
    for i in range(min(10, X.shape[0])):
        label_index = np.argmax(Y == i)
        print(f'Saving visualization for sample {label_index} with label {i}')
        # Reshape back to len(x_width) x len(x_width)
        img_array = X[label_index].reshape(x_width, x_width).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(f'{dirname}/train_sample_label={i}_index={label_index}.png')

    print(f'Visualizations saved to {dirname}/ directory')
    return None


def average_pool_4x4(image):
    pooled_image = image.reshape(7, 4, 7, 4).mean(axis=(1,3))
    return pooled_image

def learning(X_train, X_test, Y_train, Y_test, layers=(8,2,), epochs=100, batch_size=250, dirname='model_analysis'):
    # Convert into greyscale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert target classes to categorical ones
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    # Set the input shape
    input_shape = (feature_vector_length,)
    print(f'Feature shape: {input_shape}')

    # Create the model
    model = Sequential()
    model.add(Dense(layers[0], input_shape=input_shape, activation='relu'))
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

    # Test the model after training
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    save_training_history(model, history, dirname=dirname)

    return None

def save_model_weights(model, dirname='model_analysis'):
    # Create output directory
    os.makedirs(dirname, exist_ok=True)
    
    # Save all weights and biases
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            w, b = layer.get_weights()
            weights_dict[f'layer_{i}_weights'] = w
            weights_dict[f'layer_{i}_bias'] = b
    
    # Save to NPZ
    np.savez(f'{dirname}/model_weights.npz', **weights_dict)
    print(f'Weights saved to {dirname}/model_weights.npz')
    
    # Save to CSV
    for key, value in weights_dict.items():
        np.savetxt(f'{dirname}/{key}.csv', value.reshape(value.shape[0], -1), delimiter=',')
    print(f'Weights saved to {dirname}/*.csv files')
    
    return None

def save_training_history(model, history, dirname='model_analysis'):
    # Visualize model properties
    os.makedirs(dirname, exist_ok=True)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.tight_layout()
    plt.savefig(f'{dirname}/training_history.png', dpi=150)
    plt.close()

    # Print model summary
    model.summary()
    with open(f'{dirname}/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Visualize weights of first layer
    # weights_l1 = model.layers[0].get_weights()[0]
    # fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(weights_l1[:, i].reshape(7, 7), cmap='gray')
    #     ax.set_title(f'Neuron {i}')
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'{dirname}/layer1_weights.png', dpi=150)
    # plt.close()

    save_model_weights(model, dirname=dirname)
    
    print(f'Model analysis saved to {dirname}/ directory')
    return None

# Configuration options
feature_vector_length = 49
num_classes = 10

# Load the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(f'Original X_train shape: {X_train.shape}')
print(f'Original Y_train shape: {Y_train.shape}')

# X_train_sampled = X_train[:, ::4, ::4].reshape(X_train.shape[0], feature_vector_length)
# X_test_sampled = X_test[:, ::4, ::4].reshape(X_test.shape[0], feature_vector_length)
# visualize_dataset(X_train_sampled, Y_train, dirname='visualizations_sampled')
# learning(X_train_sampled, X_test_sampled, Y_train, Y_test, dirname='model_analysis_sampled')

X_train_pooled = np.array([average_pool_4x4(image) for image in X_train]).reshape(X_train.shape[0], feature_vector_length)
X_test_pooled = np.array([average_pool_4x4(image) for image in X_test]).reshape(X_test.shape[0], feature_vector_length)
visualize_dataset(X_train_pooled, Y_train, dirname='visualizations_pooled')
# learning(X_train_pooled, X_test_pooled, Y_train, Y_test, dirname='model_analysis_pooled')

mask_train_01 = np.isin(Y_train, [0,1])
mask_test_01 = np.isin(Y_test, [0,1])
X_train_pooled_01 = X_train[mask_train_01]
Y_train_pooled_01 = Y_train[mask_train_01]
X_test_01 = X_test[mask_test_01]
Y_test_01 = Y_test[mask_test_01]
learning(X_train_pooled_01, X_test_01, Y_train_pooled_01, Y_test_01, dirname='model_analysis_pooled_01')

