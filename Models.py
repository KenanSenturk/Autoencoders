# Stacked Autoencoder vs Single Autoencoder Feature Extraction Comparison
# YZM304 Deep Learning Course - Assignment 4

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SingleAutoencoder:
    """Single Autoencoder for feature extraction"""

    def __init__(self, input_dim, encoding_dims=[128, 64]):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.model = None
        self.encoder = None
        self.decoder = None

    def build_model(self):
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,))

        # Encoder
        encoded = layers.Dense(self.encoding_dims[0], activation='relu')(input_layer)
        encoded = layers.Dense(self.encoding_dims[1], activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(self.encoding_dims[0], activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)

        # Complete autoencoder
        self.model = keras.Model(input_layer, decoded)

        # Encoder model for feature extraction
        self.encoder = keras.Model(input_layer, encoded)

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X_train, X_val, epochs=100, batch_size=256):
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        return history

    def extract_features(self, X):
        return self.encoder.predict(X)

    def reconstruct(self, X):
        return self.model.predict(X)


class StackedAutoencoder:
    """Stacked Autoencoder with multiple encoding layers"""

    def __init__(self, input_dim, encoding_dims=[256, 128, 64]):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.autoencoders = []
        self.encoders = []
        self.complete_model = None
        self.complete_encoder = None

    def build_individual_autoencoders(self):
        """Build individual autoencoders for layer-wise pre-training"""
        prev_dim = self.input_dim

        for i, dim in enumerate(self.encoding_dims):
            # Input
            input_layer = keras.Input(shape=(prev_dim,))

            # Encoder
            encoded = layers.Dense(dim, activation='relu')(input_layer)

            # Decoder
            decoded = layers.Dense(prev_dim, activation='sigmoid' if i == 0 else 'relu')(encoded)

            # Autoencoder
            autoencoder = keras.Model(input_layer, decoded)
            encoder = keras.Model(input_layer, encoded)

            autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

            self.autoencoders.append(autoencoder)
            self.encoders.append(encoder)

            prev_dim = dim

    def pretrain_layers(self, X_train, X_val, epochs=50, batch_size=256):
        """Pre-train each autoencoder layer individually"""
        training_histories = []
        current_input = X_train.copy()
        current_val = X_val.copy()

        for i, autoencoder in enumerate(self.autoencoders):
            print(f"\nPre-training layer {i + 1}/{len(self.autoencoders)}")

            history = autoencoder.fit(
                current_input, current_input,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(current_val, current_val),
                verbose=1,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )

            training_histories.append(history)

            # Prepare input for next layer
            current_input = self.encoders[i].predict(current_input)
            current_val = self.encoders[i].predict(current_val)

        return training_histories

    def build_complete_model(self):
        """Build the complete stacked autoencoder"""
        input_layer = keras.Input(shape=(self.input_dim,))

        # Encoder path
        x = input_layer
        for i, dim in enumerate(self.encoding_dims):
            x = layers.Dense(dim, activation='relu', name=f'encoder_{i + 1}')(x)

        encoded = x

        # Decoder path
        for i, dim in enumerate(reversed(self.encoding_dims[:-1])):
            x = layers.Dense(dim, activation='relu', name=f'decoder_{len(self.encoding_dims) - i}')(x)

        decoded = layers.Dense(self.input_dim, activation='sigmoid', name='output')(x)

        self.complete_model = keras.Model(input_layer, decoded)
        self.complete_encoder = keras.Model(input_layer, encoded)

        self.complete_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def fine_tune(self, X_train, X_val, epochs=50, batch_size=256):
        """Fine-tune the complete stacked autoencoder"""
        # Initialize weights from pre-trained layers
        for i, autoencoder in enumerate(self.autoencoders):
            # Get weights from pre-trained encoder
            encoder_weights = autoencoder.layers[1].get_weights()
            self.complete_model.get_layer(f'encoder_{i + 1}').set_weights(encoder_weights)

        history = self.complete_model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5)
            ]
        )
        return history

    def extract_features(self, X):
        return self.complete_encoder.predict(X)

    def reconstruct(self, X):
        return self.complete_model.predict(X)


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape to flatten
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    return (X_train, X_val, X_test), (y_train, y_val, y_test)


def evaluate_classification_performance(X_train_features, X_test_features, y_train, y_test, method_name):
    """Evaluate classification performance using different classifiers"""
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}

    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name} with {method_name} features...")
        clf.fit(X_train_features, y_train)
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        results[clf_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classifier': clf
        }
        print(f"{clf_name} Accuracy: {accuracy:.4f}")

    return results


def plot_training_history(history, title):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title(f'{title} - MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(X_original, X_single_recon, X_stacked_recon, n_samples=10):
    """Plot reconstruction comparison"""
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 6))

    for i in range(n_samples):
        # Original
        axes[0, i].imshow(X_original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', rotation=90, size='large')

        # Single Autoencoder
        axes[1, i].imshow(X_single_recon[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Single AE', rotation=90, size='large')

        # Stacked Autoencoder
        axes[2, i].imshow(X_stacked_recon[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Stacked AE', rotation=90, size='large')

    plt.suptitle('Reconstruction Comparison', size=16)
    plt.tight_layout()
    plt.show()


def plot_feature_visualization(X_single_features, X_stacked_features, y_test):
    """Visualize extracted features using PCA"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Single Autoencoder features
    pca_single = PCA(n_components=2)
    features_2d_single = pca_single.fit_transform(X_single_features)

    scatter1 = ax1.scatter(features_2d_single[:, 0], features_2d_single[:, 1],
                           c=y_test, cmap='tab10', alpha=0.6, s=1)
    ax1.set_title('Single Autoencoder Features (PCA)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    plt.colorbar(scatter1, ax=ax1)

    # Stacked Autoencoder features
    pca_stacked = PCA(n_components=2)
    features_2d_stacked = pca_stacked.fit_transform(X_stacked_features)

    scatter2 = ax2.scatter(features_2d_stacked[:, 0], features_2d_stacked[:, 1],
                           c=y_test, cmap='tab10', alpha=0.6, s=1)
    ax2.set_title('Stacked Autoencoder Features (PCA)')
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    plt.colorbar(scatter2, ax=ax2)

    plt.tight_layout()
    plt.show()


def create_results_comparison_table(single_results, stacked_results):
    """Create comparison table of results"""
    comparison_data = []

    for classifier in single_results.keys():
        comparison_data.append({
            'Classifier': classifier,
            'Single Autoencoder Accuracy': single_results[classifier]['accuracy'],
            'Stacked Autoencoder Accuracy': stacked_results[classifier]['accuracy'],
            'Improvement': stacked_results[classifier]['accuracy'] - single_results[classifier]['accuracy']
        })

    df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 70)
    print("CLASSIFICATION PERFORMANCE COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False, float_format='%.4f'))
    print("=" * 70)

    return df


def plot_accuracy_comparison(comparison_df):
    """Plot accuracy comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, comparison_df['Single Autoencoder Accuracy'],
                   width, label='Single Autoencoder', alpha=0.8)
    bars2 = ax.bar(x + width / 2, comparison_df['Stacked Autoencoder Accuracy'],
                   width, label='Stacked Autoencoder', alpha=0.8)

    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Classifier'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    print("=" * 70)
    print("STACKED VS SINGLE AUTOENCODER COMPARISON PROJECT")
    print("YZM304 Deep Learning Course - Assignment 4")
    print("=" * 70)

    # Load and preprocess data
    (X_train, X_val, X_test), (y_train, y_val, y_test) = load_and_preprocess_data()

    input_dim = X_train.shape[1]  # 784 for MNIST

    print("\n" + "=" * 50)
    print("1. TRAINING SINGLE AUTOENCODER")
    print("=" * 50)

    # Single Autoencoder
    single_ae = SingleAutoencoder(input_dim, encoding_dims=[128, 64])
    single_ae.build_model()

    print("Single Autoencoder Architecture:")
    single_ae.model.summary()

    single_history = single_ae.train(X_train, X_val, epochs=100)
    plot_training_history(single_history, "Single Autoencoder")

    print("\n" + "=" * 50)
    print("2. TRAINING STACKED AUTOENCODER")
    print("=" * 50)

    # Stacked Autoencoder
    stacked_ae = StackedAutoencoder(input_dim, encoding_dims=[256, 128, 64])
    stacked_ae.build_individual_autoencoders()

    # Pre-train layers
    print("Pre-training individual layers...")
    pretrain_histories = stacked_ae.pretrain_layers(X_train, X_val, epochs=50)

    # Build and fine-tune complete model
    stacked_ae.build_complete_model()
    print("\nStacked Autoencoder Architecture:")
    stacked_ae.complete_model.summary()

    print("Fine-tuning complete stacked autoencoder...")
    stacked_history = stacked_ae.fine_tune(X_train, X_val, epochs=100)
    plot_training_history(stacked_history, "Stacked Autoencoder (Fine-tuning)")

    print("\n" + "=" * 50)
    print("3. FEATURE EXTRACTION AND EVALUATION")
    print("=" * 50)

    # Extract features
    print("Extracting features...")
    X_train_single = single_ae.extract_features(X_train)
    X_test_single = single_ae.extract_features(X_test)

    X_train_stacked = stacked_ae.extract_features(X_train)
    X_test_stacked = stacked_ae.extract_features(X_test)

    print(f"Single Autoencoder feature dimensions: {X_train_single.shape[1]}")
    print(f"Stacked Autoencoder feature dimensions: {X_train_stacked.shape[1]}")

    # Evaluate classification performance
    print("\nEvaluating classification performance...")
    single_results = evaluate_classification_performance(
        X_train_single, X_test_single, y_train, y_test, "Single Autoencoder"
    )

    stacked_results = evaluate_classification_performance(
        X_train_stacked, X_test_stacked, y_train, y_test, "Stacked Autoencoder"
    )

    print("\n" + "=" * 50)
    print("4. RESULTS COMPARISON")
    print("=" * 50)

    # Create comparison table
    comparison_df = create_results_comparison_table(single_results, stacked_results)

    # Plot accuracy comparison
    plot_accuracy_comparison(comparison_df)

    # Feature visualization
    print("\nVisualizing extracted features...")
    plot_feature_visualization(X_test_single, X_test_stacked, y_test)

    # Reconstruction comparison
    print("\nComparing reconstructions...")
    X_single_recon = single_ae.reconstruct(X_test[:10])
    X_stacked_recon = stacked_ae.reconstruct(X_test[:10])
    plot_reconstruction_comparison(X_test[:10], X_single_recon, X_stacked_recon)

    print("\n" + "=" * 50)
    print("5. ANALYSIS SUMMARY")
    print("=" * 50)

    # Calculate average improvements
    avg_improvement = comparison_df['Improvement'].mean()
    best_improvement = comparison_df['Improvement'].max()
    best_classifier = comparison_df.loc[comparison_df['Improvement'].idxmax(), 'Classifier']

    print(f"Average accuracy improvement with Stacked Autoencoder: {avg_improvement:.4f}")
    print(f"Best improvement: {best_improvement:.4f} (with {best_classifier})")

    # Reconstruction quality analysis - FIX: Use same number of samples for MSE calculation
    print("\nAnalyzing reconstruction quality...")
    n_recon_samples = 1000  # Number of samples to use for reconstruction quality analysis

    # Generate reconstructions for the analysis
    X_single_recon_analysis = single_ae.reconstruct(X_test[:n_recon_samples])
    X_stacked_recon_analysis = stacked_ae.reconstruct(X_test[:n_recon_samples])

    # Calculate MSE
    single_mse = np.mean((X_test[:n_recon_samples] - X_single_recon_analysis) ** 2)
    stacked_mse = np.mean((X_test[:n_recon_samples] - X_stacked_recon_analysis) ** 2)

    print(f"\nReconstruction Quality (MSE on {n_recon_samples} samples):")
    print(f"Single Autoencoder: {single_mse:.6f}")
    print(f"Stacked Autoencoder: {stacked_mse:.6f}")

    if single_mse > stacked_mse:
        improvement_pct = ((single_mse - stacked_mse) / single_mse * 100)
        print(f"Stacked AE improvement: {improvement_pct:.2f}%")
    elif stacked_mse > single_mse:
        degradation_pct = ((stacked_mse - single_mse) / single_mse * 100)
        print(f"Single AE better by: {degradation_pct:.2f}%")
    else:
        print("Both models have similar reconstruction quality")

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()