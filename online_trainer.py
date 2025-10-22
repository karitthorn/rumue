"""
Online Learning Trainer for Sign Language Model
ใช้สำหรับ fine-tune โมเดลด้วยข้อมูล feedback จาก webcam แบบ real-time
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from PIL import Image
import json
from datetime import datetime

class OnlineTrainer:
    def __init__(self, model_path='sign_language_cnn_final.keras',
                 feedback_dir='feedback_data',
                 replay_buffer_size=100):
        """
        Initialize Online Trainer

        Args:
            model_path: path to the trained model
            feedback_dir: directory to store feedback data
            replay_buffer_size: number of old samples to mix with new data
        """
        self.model_path = model_path
        self.feedback_dir = Path(feedback_dir)
        self.replay_buffer_size = replay_buffer_size

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        # Label mapping (A-Y, 24 classes)
        self.label_map = {i: chr(65 + i) for i in range(24)}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Create feedback directory structure
        self.setup_directories()

        # Load original training data for replay buffer (optional)
        self.original_data = None

        print("OnlineTrainer initialized successfully!")

    def setup_directories(self):
        """Create directory structure for feedback data"""
        self.feedback_dir.mkdir(exist_ok=True)

        # Create subdirectories for each letter
        for letter in self.label_map.values():
            (self.feedback_dir / letter).mkdir(exist_ok=True)

        # Create log file
        self.log_file = self.feedback_dir / 'feedback_log.csv'
        if not self.log_file.exists():
            pd.DataFrame(columns=['timestamp', 'image_path', 'predicted_label',
                                'correct_label', 'confidence']).to_csv(self.log_file, index=False)

    def save_feedback(self, image, predicted_label, correct_label, confidence):
        """
        Save feedback data (image + metadata)

        Args:
            image: numpy array (28, 28, 1) or (28, 28)
            predicted_label: str (A-Y)
            correct_label: str (A-Y)
            confidence: float
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"feedback_{timestamp}.png"

        # Save image
        image_path = self.feedback_dir / correct_label / filename

        # Convert to PIL Image and save
        if image.ndim == 3:
            image = image.squeeze()
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil.save(image_path)

        # Log to CSV
        log_entry = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'image_path': str(image_path),
            'predicted_label': predicted_label,
            'correct_label': correct_label,
            'confidence': confidence
        }])
        log_entry.to_csv(self.log_file, mode='a', header=False, index=False)

        print(f"Feedback saved: {predicted_label} -> {correct_label} ({confidence:.2%})")
        return image_path

    def load_feedback_data(self):
        """
        Load all feedback data from directory

        Returns:
            X: numpy array of images
            y: numpy array of labels (integers)
        """
        images = []
        labels = []

        for letter in self.label_map.values():
            letter_dir = self.feedback_dir / letter
            if not letter_dir.exists():
                continue

            for img_path in letter_dir.glob('*.png'):
                # Load image
                img = Image.open(img_path).convert('L')
                img_array = np.array(img) / 255.0
                images.append(img_array)

                # Get label
                label_idx = self.reverse_label_map[letter]
                labels.append(label_idx)

        if len(images) == 0:
            return None, None

        X = np.array(images).reshape(-1, 28, 28, 1)
        y = np.array(labels)

        return X, y

    def load_replay_buffer(self, original_train_csv='sign_mnist_train.csv'):
        """
        Load sample of original training data to prevent catastrophic forgetting

        Args:
            original_train_csv: path to original training CSV
        """
        try:
            df = pd.read_csv(original_train_csv)
            # Sample random subset
            df_sample = df.sample(n=min(self.replay_buffer_size, len(df)))

            X = df_sample.drop('label', axis=1).values
            y = df_sample['label'].values

            X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0

            self.original_data = (X, y)
            print(f"Loaded {len(X)} samples for replay buffer")
            return True
        except FileNotFoundError:
            print(f"Warning: {original_train_csv} not found. Training without replay buffer.")
            return False

    def incremental_train(self, epochs=2, learning_rate=0.0001,
                         use_replay_buffer=True, batch_size=8):
        """
        Perform incremental training with feedback data

        Args:
            epochs: number of epochs (keep small to prevent forgetting)
            learning_rate: small learning rate for fine-tuning
            use_replay_buffer: whether to mix old data with new data
            batch_size: batch size for training
        """
        # Load feedback data
        X_new, y_new = self.load_feedback_data()

        if X_new is None:
            print("No feedback data available for training!")
            return False

        print(f"\nLoaded {len(X_new)} feedback samples")

        # Mix with replay buffer if available
        if use_replay_buffer and self.original_data is not None:
            X_old, y_old = self.original_data
            X_train = np.concatenate([X_new, X_old])
            y_train = np.concatenate([y_new, y_old])
            print(f"Mixed with {len(X_old)} replay buffer samples")
        else:
            X_train = X_new
            y_train = y_new

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, 24)

        # Compile with small learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\nStarting incremental training...")
        print(f"Epochs: {epochs}, LR: {learning_rate}, Batch size: {batch_size}")

        # Train
        history = self.model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )

        # Save updated model
        updated_model_path = self.model_path.replace('.keras', '_updated.keras')
        self.model.save(updated_model_path)

        # Save training log
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'feedback_samples': len(X_new),
            'replay_samples': len(self.original_data[0]) if self.original_data else 0,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'model_path': updated_model_path
        }

        training_log_file = self.feedback_dir / 'training_log.json'

        if training_log_file.exists():
            with open(training_log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_data)

        with open(training_log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        print(f"\nTraining completed!")
        print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Model saved to: {updated_model_path}")

        return True

    def get_feedback_stats(self):
        """Get statistics about collected feedback"""
        stats = {}
        total_count = 0

        for letter in self.label_map.values():
            letter_dir = self.feedback_dir / letter
            if letter_dir.exists():
                count = len(list(letter_dir.glob('*.png')))
                stats[letter] = count
                total_count += count

        return stats, total_count

    def clear_feedback_data(self, archive=True):
        """
        Clear feedback data (optionally archive first)

        Args:
            archive: if True, move data to archive folder before clearing
        """
        if archive:
            archive_dir = self.feedback_dir / f'archive_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            archive_dir.mkdir(exist_ok=True)

            for letter in self.label_map.values():
                letter_dir = self.feedback_dir / letter
                if letter_dir.exists():
                    for img_path in letter_dir.glob('*.png'):
                        img_path.rename(archive_dir / letter / img_path.name)

            print(f"Feedback data archived to: {archive_dir}")
        else:
            for letter in self.label_map.values():
                letter_dir = self.feedback_dir / letter
                if letter_dir.exists():
                    for img_path in letter_dir.glob('*.png'):
                        img_path.unlink()

            print("Feedback data cleared")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Online Learning Trainer - Example Usage")
    print("=" * 60)

    # Initialize trainer
    trainer = OnlineTrainer(
        model_path='sign_language_cnn_final.keras',
        feedback_dir='feedback_data',
        replay_buffer_size=100
    )

    # Get current stats
    stats, total = trainer.get_feedback_stats()
    print(f"\nCurrent feedback data:")
    print(f"Total samples: {total}")
    for letter, count in sorted(stats.items()):
        if count > 0:
            print(f"  {letter}: {count} samples")

    # Check if we have enough data to train
    if total >= 10:
        print(f"\nFound {total} feedback samples. Ready to train!")

        # Load replay buffer (optional but recommended)
        response = input("Load replay buffer from original data? (y/n): ")
        if response.lower() == 'y':
            trainer.load_replay_buffer()

        # Train
        response = input("\nStart incremental training? (y/n): ")
        if response.lower() == 'y':
            trainer.incremental_train(
                epochs=2,
                learning_rate=0.0001,
                use_replay_buffer=True,
                batch_size=8
            )
    else:
        print(f"\nOnly {total} feedback samples collected.")
        print("Collect at least 10 samples before training.")
        print("\nUse interactive_webcam.py to collect feedback data!")
