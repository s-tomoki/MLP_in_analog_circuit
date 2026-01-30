import numpy as np
from tensorflow.keras.datasets import mnist

class Converter:
    """
    A utility class for preprocessing and converting image data for neural network training.
    This class provides methods for sampling, pooling, binarization, and label extraction
    operations on image datasets, typically used in preparing data for machine learning models.
    """

    def __init__(self):
        """Initialize the Converter instance."""
        pass

    def sampling_4x4(self, X_train, X_test):
        """
        Downsample images by selecting every 4th pixel in both dimensions.
        Args:
            X_train: Training images array of shape (n_samples, height, width)
            X_test: Test images array of shape (n_samples, height, width)
        Returns:
            Tuple of (X_train_sampled, X_test_sampled) with shape (n_samples, 49)
        """
        feature_vector_length = 49
        X_train_sampled = X_train[:, ::4, ::4].reshape(X_train.shape[0], feature_vector_length)
        X_test_sampled = X_test[:, ::4, ::4].reshape(X_test.shape[0], feature_vector_length)
        return (X_train_sampled, X_test_sampled)

    def _average_pool_4x4(self,image):    
        """
        Apply 4x4 average pooling to a single image.
        Args:
            image: Input image array of shape (28, 28)
        Returns:
            Pooled image array of shape (7, 7)
        """
        pooled_image = image.reshape(7, 4, 7, 4).mean(axis=(1,3))
        return pooled_image

    def pooling_4x4(self, X_train, X_test):
        """
        Apply 4x4 average pooling to downsample images.
        Args:
            X_train: Training images array of shape (n_samples, height, width)
            X_test: Test images array of shape (n_samples, height, width)
        Returns:
            Tuple of (X_train_pooled, X_test_pooled) with shape (n_samples, 49)
        """
        feature_vector_length = 49
        X_train_pooled = np.array([self._average_pool_4x4(image) for image in X_train]).reshape(X_train.shape[0], feature_vector_length)
        X_test_pooled = np.array([self._average_pool_4x4(image) for image in X_test]).reshape(X_test.shape[0], feature_vector_length)
        return (X_train_pooled, X_test_pooled)
    
    def binarize(self, X_train, X_test, threshold=127):
        """
        Convert images to binary values based on a threshold.
        Args:
            X_train: Training images array
            X_test: Test images array
            threshold: Threshold value for binarization (default: 127)
        Returns:
            Tuple of (X_train_bin, X_test_bin) with binary values (0 or 255)
        """
        X_train_bin = ((X_train > threshold)*255).astype(np.uint8)
        X_test_bin = ((X_test > threshold)*255).astype(np.uint8)
        return (X_train_bin, X_test_bin)
    
    def extract_labels(self, labels_to_extract, X_train, Y_train, X_test, Y_test):
        """
        Extract samples corresponding to specific labels from the dataset.
        Args:
            labels_to_extract: List or array of label values to extract
            X_train: Training images array
            Y_train: Training labels array
            X_test: Test images array
            Y_test: Test labels array
        Returns:
            Tuple of ((X_train_extracted, Y_train_extracted), (X_test_extracted, Y_test_extracted))
        """
        mask_train = np.isin(Y_train, labels_to_extract)
        mask_test = np.isin(Y_test, labels_to_extract)
        X_train_extracted = X_train[mask_train]
        Y_train_extracted = Y_train[mask_train]
        X_test_extracted = X_test[mask_test]
        Y_test_extracted = Y_test[mask_test]
        return (X_train_extracted, Y_train_extracted), (X_test_extracted, Y_test_extracted)