"""
Interactive Webcam Sign Language Recognition with Online Learning
ระบบ webcam ที่สามารถเรียนรู้ต่อเนื่องจาก user feedback
"""

import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import time
from pathlib import Path
from online_trainer import OnlineTrainer

class InteractiveWebcam:
    def __init__(self, model_path='sign_language_cnn_final.keras',
                 feedback_dir='feedback_data',
                 model_type='cnn'):
        """
        Initialize Interactive Webcam System

        Args:
            model_path: path to trained model
            feedback_dir: directory to save feedback data
            model_type: 'cnn' or 'transfer' (affects input size)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.feedback_dir = Path(feedback_dir)

        # Load model
        print(f"Loading model: {model_path}")
        self.model = keras.models.load_model(model_path)

        # Initialize online trainer
        self.trainer = OnlineTrainer(
            model_path=model_path,
            feedback_dir=feedback_dir,
            replay_buffer_size=100
        )

        # Label mapping
        self.label_map = {i: chr(65 + i) for i in range(24)}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Mode settings
        self.capture_mode = False
        self.last_prediction = None
        self.last_image = None
        self.last_confidence = 0.0

        # Stats
        self.prediction_count = 0
        self.feedback_count = 0

        # Input size based on model type
        self.input_size = (96, 96) if model_type == 'transfer' else (28, 28)
        self.input_channels = 3 if model_type == 'transfer' else 1

    def preprocess_image(self, frame):
        """
        Preprocess frame for model prediction

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            processed: numpy array ready for model
            display: frame for visualization
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Resize to model input size
        resized = cv2.resize(blurred, self.input_size)

        # Normalize
        normalized = resized.astype('float32') / 255.0

        # Reshape based on model type
        if self.input_channels == 1:
            processed = normalized.reshape(1, self.input_size[0], self.input_size[1], 1)
        else:
            # Convert grayscale to RGB for transfer learning model
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            processed = rgb.reshape(1, self.input_size[0], self.input_size[1], 3)

        return processed, gray

    def predict(self, processed_image):
        """Make prediction"""
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        predicted_letter = self.label_map[predicted_class]

        return predicted_letter, confidence, predictions[0]

    def draw_ui(self, frame, prediction, confidence, mode="normal"):
        """
        Draw UI elements on frame

        Args:
            frame: OpenCV frame
            prediction: predicted letter
            confidence: confidence score
            mode: "normal", "feedback", "training"
        """
        height, width = frame.shape[:2]

        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height - 250), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Title
        cv2.putText(frame, "Interactive Sign Language Recognition",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Prediction
        if prediction:
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"Prediction: {prediction}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.1%}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mode indicator
        if mode == "feedback":
            cv2.putText(frame, "FEEDBACK MODE - Select correct letter",
                       (10, height - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif mode == "training":
            cv2.putText(frame, "TRAINING...",
                       (10, height - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Stats
        stats, total = self.trainer.get_feedback_stats()
        cv2.putText(frame, f"Predictions: {self.prediction_count} | Feedback: {total}",
                   (10, height - 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Controls
        controls = [
            "CONTROLS:",
            "C - Correct (save as positive example)",
            "W - Wrong (select correct label)",
            "T - Train model with feedback",
            "S - Show feedback statistics",
            "R - Reload model",
            "Q - Quit"
        ]

        y_pos = height - 160
        for text in controls:
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 25

        return frame

    def get_letter_input(self):
        """
        Show letter selection menu and get user input

        Returns:
            selected letter (str) or None
        """
        print("\n" + "="*50)
        print("SELECT CORRECT LETTER")
        print("="*50)
        letters = list(self.label_map.values())

        # Display in grid
        for i in range(0, len(letters), 8):
            row = letters[i:i+8]
            print(" | ".join(f"{j+i+1:2d}. {letter}" for j, letter in enumerate(row)))

        print("\n0. Cancel")
        print("="*50)

        try:
            choice = input("Enter number (1-24) or 0 to cancel: ")
            choice = int(choice)

            if choice == 0:
                return None
            elif 1 <= choice <= 24:
                return letters[choice - 1]
            else:
                print("Invalid choice!")
                return None
        except ValueError:
            print("Invalid input!")
            return None

    def show_statistics(self):
        """Display detailed feedback statistics"""
        print("\n" + "="*60)
        print("FEEDBACK STATISTICS")
        print("="*60)

        stats, total = self.trainer.get_feedback_stats()

        if total == 0:
            print("No feedback data collected yet.")
        else:
            print(f"Total feedback samples: {total}")
            print("\nPer-letter breakdown:")

            # Show in columns
            letters = sorted(stats.items())
            for i in range(0, len(letters), 4):
                row = letters[i:i+4]
                print("  ".join(f"{letter}: {count:3d}" for letter, count in row))

            # Show which letters need more data
            min_count = min(stats.values()) if stats else 0
            if min_count < 5:
                needs_more = [letter for letter, count in stats.items() if count < 5]
                print(f"\nLetters needing more samples (< 5): {', '.join(needs_more)}")

        print("="*60)

    def reload_model(self):
        """Reload the model (useful after training)"""
        print("\n" + "="*60)
        print("RELOAD MODEL")
        print("="*60)

        # Check for updated model
        updated_path = self.model_path.replace('.keras', '_updated.keras')

        if Path(updated_path).exists():
            print(f"Found updated model: {updated_path}")
            response = input("Load updated model? (y/n): ")

            if response.lower() == 'y':
                self.model = keras.models.load_model(updated_path)
                self.model_path = updated_path
                print("Updated model loaded successfully!")
                return

        # Reload original model
        print(f"Reloading original model: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("Model reloaded successfully!")

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("INTERACTIVE WEBCAM SIGN LANGUAGE RECOGNITION")
        print("="*60)
        print("\nStarting webcam...")

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open webcam!")
            print("Try running camera_diagnostic.py to troubleshoot.")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Webcam opened successfully!")
        print("\nPress keys to interact (see controls on screen)")

        mode = "normal"
        fps_start_time = time.time()
        fps_counter = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)

                # Define ROI (Region of Interest) for hand
                roi_x, roi_y, roi_w, roi_h = 50, 100, 300, 300
                cv2.rectangle(frame, (roi_x, roi_y),
                            (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

                # Extract ROI
                roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                # Preprocess and predict
                processed, gray = self.preprocess_image(roi)
                prediction, confidence, _ = self.predict(processed)

                # Store for feedback
                self.last_prediction = prediction
                self.last_confidence = confidence
                self.last_image = gray
                self.prediction_count += 1

                # Draw UI
                frame = self.draw_ui(frame, prediction, confidence, mode)

                # Show preview of processed image
                preview = cv2.resize(gray, (150, 150))
                frame[10:160, frame.shape[1]-160:frame.shape[1]-10] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

                # Calculate and show FPS
                fps_counter += 1
                if fps_counter >= 10:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_counter = 0
                    cv2.putText(frame, f"FPS: {fps:.1f}",
                              (frame.shape[1] - 150, frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display
                cv2.imshow('Interactive Webcam', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break

                elif key == ord('c'):
                    # Correct - save as positive example
                    if self.last_image is not None:
                        self.trainer.save_feedback(
                            self.last_image / 255.0,  # Normalize
                            self.last_prediction,
                            self.last_prediction,  # Same label
                            self.last_confidence
                        )
                        print(f"Saved positive example: {self.last_prediction}")

                elif key == ord('w'):
                    # Wrong - get correct label
                    if self.last_image is not None:
                        mode = "feedback"
                        correct_label = self.get_letter_input()

                        if correct_label:
                            self.trainer.save_feedback(
                                self.last_image / 255.0,
                                self.last_prediction,
                                correct_label,
                                self.last_confidence
                            )
                            print(f"Feedback saved: {self.last_prediction} -> {correct_label}")

                        mode = "normal"

                elif key == ord('t'):
                    # Train model
                    mode = "training"
                    cv2.imshow('Interactive Webcam', frame)

                    print("\n" + "="*60)
                    print("TRAINING MODE")
                    print("="*60)

                    stats, total = self.trainer.get_feedback_stats()

                    if total < 10:
                        print(f"Not enough feedback data: {total} samples")
                        print("Collect at least 10 samples before training.")
                        mode = "normal"
                        continue

                    print(f"Found {total} feedback samples")

                    # Ask about replay buffer
                    response = input("Load replay buffer from original data? (y/n): ")
                    if response.lower() == 'y':
                        self.trainer.load_replay_buffer()

                    # Train
                    print("\nStarting training...")
                    success = self.trainer.incremental_train(
                        epochs=2,
                        learning_rate=0.0001,
                        use_replay_buffer=True,
                        batch_size=8
                    )

                    if success:
                        print("\nTraining completed!")
                        response = input("Load updated model now? (y/n): ")
                        if response.lower() == 'y':
                            self.reload_model()

                    mode = "normal"

                elif key == ord('s'):
                    # Show statistics
                    self.show_statistics()

                elif key == ord('r'):
                    # Reload model
                    self.reload_model()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            print("\nCleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam closed")

            # Show final stats
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total predictions made: {self.prediction_count}")
            stats, total = self.trainer.get_feedback_stats()
            print(f"Total feedback collected: {total}")
            print("="*60)


if __name__ == "__main__":
    import sys

    # Default values
    model_path = 'sign_language_cnn_final.keras'
    model_type = 'cnn'

    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

        # Detect model type from filename
        if 'transfer' in model_path.lower():
            model_type = 'transfer'

    print("\n" + "="*60)
    print("INTERACTIVE WEBCAM - ONLINE LEARNING")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Type: {model_type}")
    print("="*60)

    # Initialize and run
    webcam = InteractiveWebcam(
        model_path=model_path,
        feedback_dir='feedback_data',
        model_type=model_type
    )

    webcam.run()
