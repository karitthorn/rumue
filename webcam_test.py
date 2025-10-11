"""
Real-time Sign Language Recognition with Webcam
ทดสอบโมเดลแบบ Real-time ด้วย Webcam

วิธีใช้:
    python webcam_test.py --model cnn
    python webcam_test.py --model transfer

คีย์บอร์ด:
    - SPACE: ถ่ายภาพและทำนาย
    - 'e': สลับโหมด Edge Detection (Normal/Sobel/Laplacian/Canny)
    - 'c': ล้างหน้าจอ
    - 's': บันทึกภาพ
    - 'q' หรือ ESC: ออกจากโปรแกรม
"""

import numpy as np
import cv2
import argparse
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Label mapping
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}


class SignLanguagePredictor:
    """Class สำหรับทำนาย Sign Language แบบ Real-time"""

    def __init__(self, model_path, model_type='cnn'):
        """
        Args:
            model_path: path ของโมเดล
            model_type: 'cnn' หรือ 'transfer'
        """
        self.model_type = model_type
        print(f"🔄 กำลังโหลดโมเดล {model_type}...")
        self.model = keras.models.load_model(model_path)
        print("✅ โหลดโมเดลสำเร็จ!")

        self.prediction_history = []
        self.max_history = 10

    def preprocess_image(self, image):
        """
        เตรียมภาพสำหรับการทำนาย

        Args:
            image: numpy array (BGR image from webcam)

        Returns:
            processed_image: ภาพที่พร้อมสำหรับทำนาย
        """
        # แปลงเป็น grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize เป็น 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize
        normalized = resized.astype('float32') / 255.0

        # เตรียมตามประเภทโมเดล
        if self.model_type == 'cnn':
            # CNN: (1, 28, 28, 1)
            processed = normalized.reshape(1, 28, 28, 1)
        else:
            # Transfer Learning: (1, 96, 96, 3)
            normalized = normalized.reshape(28, 28, 1)
            rgb = np.repeat(normalized, 3, axis=-1)
            resized_96 = tf.image.resize(rgb, [96, 96]).numpy()
            processed = resized_96.reshape(1, 96, 96, 3)

        return processed

    def predict(self, image):
        """
        ทำนาย Sign Language

        Args:
            image: numpy array (BGR image from webcam)

        Returns:
            predicted_class: คลาสที่ทำนาย
            confidence: ความมั่นใจ (0-1)
            all_probs: ความน่าจะเป็นทั้งหมด
        """
        # เตรียมภาพ
        processed = self.preprocess_image(image)

        # ทำนาย
        predictions = self.model.predict(processed, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # เก็บประวัติ
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence,
            'letter': LABELS[predicted_class]
        })

        # จำกัดจำนวนประวัติ
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)

        return predicted_class, confidence, predictions[0]

    def get_averaged_prediction(self):
        """
        คำนวณการทำนายเฉลี่ยจากประวัติ (ช่วยลด noise)

        Returns:
            most_common_letter: ตัวอักษรที่ปรากฏบ่อยที่สุด
            avg_confidence: ความมั่นใจเฉลี่ย
        """
        if not self.prediction_history:
            return None, 0.0

        # นับความถี่ของแต่ละ class
        class_counts = {}
        for pred in self.prediction_history:
            letter = pred['letter']
            if letter in class_counts:
                class_counts[letter] += 1
            else:
                class_counts[letter] = 1

        # หา class ที่ปรากฏบ่อยที่สุด
        most_common = max(class_counts.items(), key=lambda x: x[1])
        most_common_letter = most_common[0]

        # คำนวณความมั่นใจเฉลี่ย
        confidences = [p['confidence'] for p in self.prediction_history if p['letter'] == most_common_letter]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return most_common_letter, avg_confidence


def apply_edge_detection(frame, mode='normal'):
    """
    ใช้ Convolution filters เพื่อทำ Edge Detection

    Args:
        frame: เฟรมจากกล้อง (BGR)
        mode: โหมดการประมวลผล
            - 'normal': ไม่แก้ไข
            - 'sobel': Sobel edge detection (รวม x และ y)
            - 'laplacian': Laplacian edge detection
            - 'canny': Canny edge detection
            - 'sobel_x': Sobel แนวนอน
            - 'sobel_y': Sobel แนวตั้ง

    Returns:
        processed_frame: เฟรมที่ผ่านการประมวลผล
    """
    if mode == 'normal':
        return frame

    # แปลงเป็น grayscale สำหรับการหาขอบ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == 'sobel':
        # Sobel filter - หาขอบทั้งแนวนอนและแนวตั้ง
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # รวมผลลัพธ์
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(np.clip(sobel, 0, 255))

        # แปลงกลับเป็น BGR
        result = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    elif mode == 'sobel_x':
        # Sobel แนวนอน (หาขอบแนวตั้ง)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.uint8(np.absolute(sobelx))
        result = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)

    elif mode == 'sobel_y':
        # Sobel แนวตั้ง (หาขอบแนวนอน)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobely = np.uint8(np.absolute(sobely))
        result = cv2.cvtColor(sobely, cv2.COLOR_GRAY2BGR)

    elif mode == 'laplacian':
        # Laplacian filter - หาขอบทุกทิศทาง
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))
        result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    elif mode == 'canny':
        # Canny edge detection - วิธีที่ดีที่สุด
        # ใช้ Gaussian blur ก่อนเพื่อลด noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, 50, 150)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    else:
        result = frame

    return result


def draw_ui(frame, predictor, prediction_result=None, edge_mode='normal'):
    """
    วาด UI บนเฟรม

    Args:
        frame: เฟรมจากกล้อง
        predictor: SignLanguagePredictor instance
        prediction_result: ผลการทำนาย (tuple: class, confidence, probs)
        edge_mode: โหมด Edge Detection ที่ใช้
    """
    height, width = frame.shape[:2]

    # วาดกล่องสำหรับวางมือ
    box_size = 300
    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # แสดงชื่อโมเดลและ Edge Detection mode
    model_name = "CNN Model" if predictor.model_type == 'cnn' else "Transfer Learning"
    cv2.putText(frame, f"Model: {model_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # แสดงโหมด Edge Detection
    edge_color = (0, 255, 255) if edge_mode != 'normal' else (255, 255, 255)
    cv2.putText(frame, f"Edge Mode: {edge_mode.upper()}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, edge_color, 2)

    # แสดงคำแนะนำ
    instructions = [
        "Controls:",
        "SPACE - Capture & Predict",
        "E - Toggle Edge Mode",
        "C - Clear history",
        "S - Save image",
        "Q/ESC - Quit"
    ]

    y_pos = 90
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25

    # แสดงผลการทำนาย
    if prediction_result is not None:
        pred_class, confidence, all_probs = prediction_result
        letter = LABELS[pred_class]

        # แสดงผลการทำนายปัจจุบัน
        pred_text = f"Prediction: {letter}"
        conf_text = f"Confidence: {confidence*100:.1f}%"

        # เลือกสี (เขียว = มั่นใจสูง, แดง = มั่นใจต่ำ)
        color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)

        cv2.putText(frame, pred_text, (width - 300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, conf_text, (width - 300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # แสดง Top-3 predictions
        top_3_idx = np.argsort(all_probs)[-3:][::-1]
        y_top = 120
        cv2.putText(frame, "Top 3:", (width - 300, y_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i, idx in enumerate(top_3_idx):
            prob = all_probs[idx]
            text = f"{i+1}. {LABELS[idx]}: {prob*100:.1f}%"
            cv2.putText(frame, text, (width - 280, y_top + 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # แสดงการทำนายเฉลี่ย (จากประวัติ)
    if len(predictor.prediction_history) > 0:
        avg_letter, avg_conf = predictor.get_averaged_prediction()
        avg_text = f"Averaged: {avg_letter} ({avg_conf*100:.1f}%)"
        cv2.putText(frame, avg_text, (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # แสดงประวัติการทำนาย
        history_text = "History: "
        for pred in predictor.prediction_history[-5:]:
            history_text += f"{pred['letter']} "

        cv2.putText(frame, history_text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description='Real-time Sign Language Recognition')
    parser.add_argument('--model', type=str, choices=['cnn', 'transfer'], default='cnn',
                        help='ชนิดของโมเดล')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--save_dir', type=str, default='webcam_captures',
                        help='โฟลเดอร์สำหรับบันทึกภาพ')

    args = parser.parse_args()

    # เตรียมโฟลเดอร์สำหรับบันทึกภาพ
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # โหลดโมเดล
    model_path = f"sign_language_{args.model}_final.keras" if args.model == 'cnn' else "sign_language_transfer_final.keras"

    if not Path(model_path).exists():
        print(f"❌ ไม่พบโมเดล: {model_path}")
        return

    predictor = SignLanguagePredictor(model_path, args.model)

    # เปิดกล้อง
    print(f"\n🎥 กำลังเปิดกล้อง (index: {args.camera})...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้อง!")
        print("💡 ลองเปลี่ยน camera index ด้วย --camera <number>")
        return

    print("✅ เปิดกล้อง สำเร็จ!")
    print("\n" + "="*60)
    print("🚀 REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("="*60)
    print("\nControls:")
    print("  SPACE - Capture & Predict")
    print("  E     - Toggle Edge Detection Mode")
    print("  C     - Clear prediction history")
    print("  S     - Save current frame")
    print("  Q/ESC - Quit")
    print("="*60 + "\n")

    current_prediction = None
    frame_count = 0
    saved_count = 0

    # Edge detection modes
    edge_modes = ['normal', 'sobel', 'laplacian', 'canny', 'sobel_x', 'sobel_y']
    current_edge_mode_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ ไม่สามารถอ่านเฟรมจากกล้อง!")
            break

        # Flip เฟรม (mirror effect)
        frame = cv2.flip(frame, 1)

        # ใช้ Edge Detection filter
        current_edge_mode = edge_modes[current_edge_mode_idx]
        processed_frame = apply_edge_detection(frame, current_edge_mode)

        # วาด UI
        frame_with_ui = draw_ui(processed_frame.copy(), predictor, current_prediction, current_edge_mode)

        # แสดงผล
        cv2.imshow('Sign Language Recognition', frame_with_ui)

        # จัดการ keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n👋 ปิดโปรแกรม...")
            break

        elif key == ord(' '):  # SPACE - Capture & Predict
            # ครอบตัดพื้นที่กล่อง
            height, width = frame.shape[:2]
            box_size = 300
            x1 = (width - box_size) // 2
            y1 = (height - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            hand_region = frame[y1:y2, x1:x2]

            # ทำนาย
            print("\n📸 Capturing and predicting...")
            pred_class, confidence, all_probs = predictor.predict(hand_region)
            current_prediction = (pred_class, confidence, all_probs)

            letter = LABELS[pred_class]
            print(f"   Prediction: {letter} (Confidence: {confidence*100:.2f}%)")

            # แสดง Top-3
            top_3_idx = np.argsort(all_probs)[-3:][::-1]
            print("   Top 3:")
            for i, idx in enumerate(top_3_idx, 1):
                print(f"      {i}. {LABELS[idx]}: {all_probs[idx]*100:.2f}%")

        elif key == ord('e'):  # Toggle Edge Detection Mode
            current_edge_mode_idx = (current_edge_mode_idx + 1) % len(edge_modes)
            new_mode = edge_modes[current_edge_mode_idx]
            print(f"\n🔄 เปลี่ยนโหมดเป็น: {new_mode.upper()}")

            # แสดงคำอธิบาย
            mode_descriptions = {
                'normal': 'ไม่มีการประมวลผล',
                'sobel': 'Sobel - หาขอบทั้งแนวนอนและแนวตั้ง',
                'laplacian': 'Laplacian - หาขอบทุกทิศทาง',
                'canny': 'Canny - Edge Detection ที่ดีที่สุด',
                'sobel_x': 'Sobel X - หาขอบแนวตั้ง',
                'sobel_y': 'Sobel Y - หาขอบแนวนอน'
            }
            print(f"   {mode_descriptions[new_mode]}")

        elif key == ord('c'):  # Clear history
            predictor.prediction_history = []
            current_prediction = None
            print("\n🗑️  ล้างประวัติการทำนาย")

        elif key == ord('s'):  # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1
            print(f"\n💾 บันทึกภาพ: {filename}")

        frame_count += 1

    # ปิดกล้อง
    cap.release()
    cv2.destroyAllWindows()

    # สรุปผล
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total predictions made: {len(predictor.prediction_history)}")
    print(f"Images saved: {saved_count}")

    if predictor.prediction_history:
        print(f"\nMost recent predictions:")
        for i, pred in enumerate(predictor.prediction_history[-5:], 1):
            print(f"  {i}. {pred['letter']} ({pred['confidence']*100:.1f}%)")

    print("\n✅ Session completed!")


if __name__ == '__main__':
    main()
