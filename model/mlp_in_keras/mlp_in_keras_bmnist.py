# Ref: 
# https://machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api

# Imports
import os
from PIL import Image
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

import converter

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

def learning(X_train, X_test, Y_train, Y_test, num_classes = 10, layers=(8,2,), epochs=100, batch_size=250, dirname='model_analysis'):
    input_shape = (49,)
    # Convert into greyscale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert target classes to categorical ones
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    # Create the model
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(layers[0], activation='relu', kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation='relu', kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
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
    save_model_weights(model, dirname=dirname)
    print(f'Model analysis saved to {dirname}/ directory')
    return None



def main():
    # Configuration options
    feature_vector_length = 49
    num_classes = 10

    # Load the data
    cvt = converter.Converter()
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(f'Original X_train shape: {X_train.shape}')
    print(f'Original Y_train shape: {Y_train.shape}')

    # Case 1: 
    #   Input: Pooling 4x4 (7x7 images, grayscale)
    #   Category: 10 classes (0-9)
    #   Layers: (10, 8, 2)
    (X_train_pooled, X_test_pooled) = cvt.pooling_4x4(X_train, X_test)
    visualize_dataset(X_train_pooled, Y_train, dirname='visualizations_pooled')
    learning(X_train_pooled, X_test_pooled, Y_train, Y_test, dirname='model_analysis_pooled')

    # Case 2: 
    #   Input: Pooling 4x4 and binarization (7x7 images, binary)
    #   Category: 10 classes (0-9)
    #   Layers: (10, 8, 2)
    (X_train_pooled_bin, X_test_pooled_bin) = cvt.binarize(X_train_pooled, X_test_pooled)
    visualize_dataset(X_train_pooled_bin, Y_train, dirname='visualizations_pooled_bin')
    learning(X_train_pooled_bin, X_test_pooled_bin, Y_train, Y_test, dirname='model_analysis_pooled_bin')

    # Case 3: 
    #   Input: Pooling 4x4 (7x7 images, grayscale) 
    #   Category: 2 classes (0 and 1)
    #   Layers: (2, 2)
    (X_train_pooled_01, Y_train_pooled_01), (X_test_pooled_01, Y_test_01) = cvt.extract_labels([0,1], X_train_pooled, Y_train, X_test_pooled, Y_test)
    print(f'Number of training samples for digits 0 and 1: {len(Y_train_pooled_01)}')
    print(f'Number of test samples for digits 0 and 1: {len(Y_test_01)}')
    learning(X_train_pooled_01, X_test_pooled_01, Y_train_pooled_01, Y_test_01, num_classes=2, layers=(2,),dirname='model_analysis_pooled_01')

    # Case 4: 
    #   Input: Pooling 4x4 and binarization (7x7 images, binary) 
    #   Category: 2 classes (0 and 1)
    #   Layers: (2, 2)
    (X_train_pooled_bin_01, X_test_pooled_bin_01) = cvt.binarize(X_train_pooled_01, X_test_pooled_01)
    learning(X_train_pooled_bin_01, X_test_pooled_bin_01, Y_train_pooled_01, Y_test_01, num_classes=2, layers=(2,),dirname='model_analysis_pooled_bin_01')

if __name__ == '__main__':
    main()