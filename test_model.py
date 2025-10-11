"""
Sign Language Recognition - Model Testing Script
ทดสอบโมเดล AI สำหรับการจำแนกภาษามือ

วิธีใช้งาน:
    python test_model.py --mode [test/predict/interactive]

    --mode test: ทดสอบกับ test dataset
    --mode predict: ทำนายภาพเดียว
    --mode interactive: โหมดทดสอบแบบโต้ตอบ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Image processing
from PIL import Image
import cv2

# ===== Configuration =====
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

MODEL_PATHS = {
    'cnn': 'sign_language_cnn_final.keras',
    'transfer': 'sign_language_transfer_final.keras'
}

# ===== Helper Functions =====
def load_model(model_type='cnn'):
    """โหลดโมเดลที่เทรนแล้ว"""
    model_path = MODEL_PATHS.get(model_type)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path}")

    print(f"กำลังโหลดโมเดล {model_type}...")
    model = keras.models.load_model(model_path)
    print(f"✅ โหลดโมเดล {model_type} สำเร็จ!")
    return model


def preprocess_image_from_file(image_path, model_type='cnn'):
    """
    เตรียมภาพจากไฟล์สำหรับการทำนาย

    Args:
        image_path: path ไปยังไฟล์ภาพ
        model_type: 'cnn' หรือ 'transfer'

    Returns:
        processed_image: numpy array ที่พร้อมสำหรับการทำนาย
        original_image: ภาพต้นฉบับสำหรับแสดงผล
    """
    # อ่านภาพ
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    original = np.array(img)

    # Resize เป็น 28x28
    img_resized = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img_resized)

    # Normalize
    img_normalized = img_array.astype('float32') / 255.0

    # เตรียมตามประเภทโมเดล
    if model_type == 'cnn':
        # CNN: (28, 28, 1)
        processed = img_normalized.reshape(1, 28, 28, 1)
    else:
        # Transfer Learning: (96, 96, 3)
        img_normalized = img_normalized.reshape(28, 28, 1)
        img_rgb = np.repeat(img_normalized, 3, axis=-1)
        img_resized = tf.image.resize(img_rgb, [96, 96]).numpy()
        processed = img_resized.reshape(1, 96, 96, 3)

    return processed, original


def predict_sign(model, image, model_type='cnn'):
    """
    ทำนาย Sign Language จากภาพ

    Args:
        model: โมเดลที่เทรนแล้ว
        image: numpy array ของภาพ
        model_type: 'cnn' หรือ 'transfer'

    Returns:
        predicted_class: คลาสที่ทำนาย
        confidence: ความมั่นใจ
        all_probabilities: ความน่าจะเป็นทั้งหมด
    """
    # ทำนาย
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence, predictions[0]


def visualize_prediction(image, true_label, pred_label, confidence, top_k_probs=None, save_path=None):
    """แสดงผลการทำนาย"""
    fig, axes = plt.subplots(1, 2 if top_k_probs is not None else 1, figsize=(12, 4))

    if top_k_probs is None:
        axes = [axes]

    # แสดงภาพ
    if len(image.shape) == 4:
        img_display = image[0, :, :, 0] if image.shape[-1] == 1 else image[0, :, :, :]
    else:
        img_display = image

    axes[0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)

    title = f"Prediction: {LABEL_MAP[pred_label]} ({confidence*100:.1f}%)"
    if true_label is not None:
        title = f"True: {LABEL_MAP[true_label]}\n" + title
        color = 'green' if pred_label == true_label else 'red'
    else:
        color = 'blue'

    axes[0].set_title(title, fontsize=14, color=color, weight='bold')
    axes[0].axis('off')

    # แสดง Top-K predictions
    if top_k_probs is not None:
        top_k = 5
        top_indices = np.argsort(top_k_probs)[-top_k:][::-1]
        top_probs = top_k_probs[top_indices]
        labels = [LABEL_MAP[i] for i in top_indices]

        colors = ['green' if i == 0 else 'lightblue' for i in range(top_k)]
        axes[1].barh(labels, top_probs, color=colors)
        axes[1].set_xlabel('Confidence (%)', fontsize=12)
        axes[1].set_title(f'Top {top_k} Predictions', fontsize=14)
        axes[1].set_xlim([0, 1])

        for i, (label, prob) in enumerate(zip(labels, top_probs)):
            axes[1].text(prob + 0.01, i, f'{prob*100:.1f}%', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ บันทึกผลลัพธ์ที่: {save_path}")

    plt.show()


# ===== Testing Functions =====
def test_with_dataset(model_type='cnn'):
    """ทดสอบโมเดลกับ test dataset"""
    print("\n" + "="*60)
    print(f"TESTING MODEL: {model_type.upper()}")
    print("="*60)

    # โหลดโมเดล
    model = load_model(model_type)

    # โหลด test data
    print("\nกำลังโหลด test dataset...")
    cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "datamunge" / "sign-language-mnist" / "versions" / "1"
    test_csv = cache_dir / "sign_mnist_test.csv"

    if not test_csv.exists():
        print(f"❌ ไม่พบไฟล์: {test_csv}")
        print("กรุณารันโค้ดใน notebook ก่อนเพื่อดาวน์โหลด dataset")
        return

    test_df = pd.read_csv(test_csv)

    # เตรียมข้อมูล
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # เตรียมข้อมูลตามประเภทโมเดล
    if model_type == 'transfer':
        X_test_rgb = np.repeat(X_test, 3, axis=-1)
        X_test = tf.image.resize(X_test_rgb, [96, 96]).numpy()

    print(f"Test samples: {len(X_test)}")

    # ทำนาย
    print("\nกำลังทำนาย...")
    predictions = model.predict(X_test, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)

    # ประเมินผล
    accuracy = accuracy_score(y_test, pred_classes)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*60}")

    # Classification Report
    available_labels = sorted(np.unique(y_test))
    target_names = [LABEL_MAP[i] for i in available_labels]

    print(f"\nClassification Report:")
    print(classification_report(y_test, pred_classes, labels=available_labels, target_names=target_names))

    # Confusion Matrix
    print("\nกำลังสร้าง Confusion Matrix...")
    cm = confusion_matrix(y_test, pred_classes, labels=available_labels)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_type.upper()} Model - Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    save_path = f'confusion_matrix_{model_type}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ บันทึก Confusion Matrix ที่: {save_path}")
    plt.show()

    # แสดงตัวอย่างการทำนาย
    print("\nแสดงตัวอย่างการทำนาย (สุ่ม 10 ภาพ)...")
    show_random_predictions(model, X_test, y_test, model_type, n_samples=10)

    return accuracy, predictions


def show_random_predictions(model, X_test, y_test, model_type='cnn', n_samples=10):
    """แสดงตัวอย่างการทำนายแบบสุ่ม"""
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image = X_test[idx]
        true_label = y_test[idx]

        # ทำนาย
        pred = model.predict(image.reshape(1, *image.shape), verbose=0)
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class]

        # แสดงภาพ
        if model_type == 'transfer':
            img_display = image[:, :, 0]  # Show first channel
        else:
            img_display = image.reshape(28, 28)

        axes[i].imshow(img_display, cmap='gray')

        title = f"True: {LABEL_MAP[true_label]}\n"
        title += f"Pred: {LABEL_MAP[pred_class]} ({confidence*100:.1f}%)"

        color = 'green' if pred_class == true_label else 'red'
        axes[i].set_title(title, fontsize=10, color=color, weight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(f'{model_type.upper()} - Random Predictions', y=1.02, fontsize=16)

    save_path = f'random_predictions_{model_type}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ บันทึกผลลัพธ์ที่: {save_path}")
    plt.show()


def predict_single_image(image_path, model_type='cnn', show_top_k=True):
    """ทำนายภาพเดียว"""
    print("\n" + "="*60)
    print(f"SINGLE IMAGE PREDICTION - {model_type.upper()}")
    print("="*60)

    # โหลดโมเดล
    model = load_model(model_type)

    # เตรียมภาพ
    print(f"\nกำลังเตรียมภาพ: {image_path}")
    processed_img, original_img = preprocess_image_from_file(image_path, model_type)

    # ทำนาย
    print("กำลังทำนาย...")
    pred_class, confidence, all_probs = predict_sign(model, processed_img, model_type)

    print(f"\n{'='*60}")
    print(f"Prediction: {LABEL_MAP[pred_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"{'='*60}")

    # แสดง Top-5
    if show_top_k:
        print("\nTop 5 Predictions:")
        top_indices = np.argsort(all_probs)[-5:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"{i}. {LABEL_MAP[idx]}: {all_probs[idx]*100:.2f}%")

    # Visualize
    visualize_prediction(
        processed_img,
        None,
        pred_class,
        confidence,
        all_probs if show_top_k else None,
        save_path=f'prediction_{Path(image_path).stem}_{model_type}.png'
    )

    return pred_class, confidence


def interactive_mode():
    """โหมดทดสอบแบบโต้ตอบ"""
    print("\n" + "="*60)
    print("INTERACTIVE TESTING MODE")
    print("="*60)
    print("\nโหลดทั้งสองโมเดล...")

    cnn_model = load_model('cnn')
    transfer_model = load_model('transfer')

    print("\n✅ พร้อมใช้งาน!")
    print("\nคำแนะนำ:")
    print("  - ใส่ path ของภาพที่ต้องการทดสอบ")
    print("  - พิมพ์ 'quit' หรือ 'exit' เพื่อออก")
    print("  - พิมพ์ 'random' เพื่อทดสอบภาพสุ่มจาก dataset")

    while True:
        print("\n" + "-"*60)
        user_input = input("📷 ใส่ path ของภาพ (หรือคำสั่ง): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 ขอบคุณที่ใช้งาน!")
            break

        if user_input.lower() == 'random':
            # ทดสอบภาพสุ่มจาก dataset
            print("\nกำลังโหลดภาพสุ่มจาก test dataset...")
            cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "datamunge" / "sign-language-mnist" / "versions" / "1"
            test_csv = cache_dir / "sign_mnist_test.csv"

            if test_csv.exists():
                test_df = pd.read_csv(test_csv)
                idx = np.random.randint(0, len(test_df))

                # เตรียมข้อมูล
                X_test = test_df.drop('label', axis=1).values
                y_test = test_df['label'].values
                X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

                image = X_test[idx]
                true_label = y_test[idx]

                print(f"\nภาพที่สุ่มได้: Index {idx}")
                print(f"True Label: {LABEL_MAP[true_label]}")

                # ทำนายด้วย CNN
                pred_cnn = cnn_model.predict(image.reshape(1, 28, 28, 1), verbose=0)
                class_cnn = np.argmax(pred_cnn[0])
                conf_cnn = pred_cnn[0][class_cnn]

                # ทำนายด้วย Transfer Learning
                img_transfer = np.repeat(image, 3, axis=-1)
                img_transfer = tf.image.resize(img_transfer, [96, 96]).numpy()
                pred_tl = transfer_model.predict(img_transfer.reshape(1, 96, 96, 3), verbose=0)
                class_tl = np.argmax(pred_tl[0])
                conf_tl = pred_tl[0][class_tl]

                # แสดงผล
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                # ภาพต้นฉบับ
                axes[0].imshow(image.reshape(28, 28), cmap='gray')
                axes[0].set_title(f'Original\nTrue: {LABEL_MAP[true_label]}',
                                fontsize=12, weight='bold')
                axes[0].axis('off')

                # CNN prediction
                axes[1].imshow(image.reshape(28, 28), cmap='gray')
                color_cnn = 'green' if class_cnn == true_label else 'red'
                axes[1].set_title(f'CNN Model\nPred: {LABEL_MAP[class_cnn]} ({conf_cnn*100:.1f}%)',
                                fontsize=12, color=color_cnn, weight='bold')
                axes[1].axis('off')

                # Transfer Learning prediction
                axes[2].imshow(image.reshape(28, 28), cmap='gray')
                color_tl = 'green' if class_tl == true_label else 'red'
                axes[2].set_title(f'Transfer Learning\nPred: {LABEL_MAP[class_tl]} ({conf_tl*100:.1f}%)',
                                fontsize=12, color=color_tl, weight='bold')
                axes[2].axis('off')

                plt.tight_layout()
                plt.show()

                print(f"\n{'='*60}")
                print(f"True Label: {LABEL_MAP[true_label]}")
                print(f"CNN Prediction: {LABEL_MAP[class_cnn]} ({conf_cnn*100:.2f}%) - {'✅' if class_cnn == true_label else '❌'}")
                print(f"Transfer Prediction: {LABEL_MAP[class_tl]} ({conf_tl*100:.2f}%) - {'✅' if class_tl == true_label else '❌'}")
                print(f"{'='*60}")
            else:
                print("❌ ไม่พบ test dataset")

            continue

        # ทดสอบภาพจากไฟล์
        if not os.path.exists(user_input):
            print(f"❌ ไม่พบไฟล์: {user_input}")
            continue

        try:
            print("\n🔍 ทดสอบด้วยทั้งสองโมเดล...")

            # เตรียมภาพสำหรับแต่ละโมเดล
            img_cnn, original = preprocess_image_from_file(user_input, 'cnn')
            img_transfer, _ = preprocess_image_from_file(user_input, 'transfer')

            # ทำนาย
            class_cnn, conf_cnn, probs_cnn = predict_sign(cnn_model, img_cnn, 'cnn')
            class_tl, conf_tl, probs_tl = predict_sign(transfer_model, img_transfer, 'transfer')

            # แสดงผล
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Original image
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original Image', fontsize=12, weight='bold')
            axes[0].axis('off')

            # CNN prediction
            axes[1].imshow(original, cmap='gray')
            axes[1].set_title(f'CNN Model\n{LABEL_MAP[class_cnn]} ({conf_cnn*100:.1f}%)',
                            fontsize=12, color='blue', weight='bold')
            axes[1].axis('off')

            # Transfer Learning prediction
            axes[2].imshow(original, cmap='gray')
            axes[2].set_title(f'Transfer Learning\n{LABEL_MAP[class_tl]} ({conf_tl*100:.1f}%)',
                            fontsize=12, color='green', weight='bold')
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

            # แสดงผลแบบละเอียด
            print(f"\n{'='*60}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"\nCNN Model:")
            print(f"  Prediction: {LABEL_MAP[class_cnn]}")
            print(f"  Confidence: {conf_cnn*100:.2f}%")

            print(f"\nTransfer Learning Model:")
            print(f"  Prediction: {LABEL_MAP[class_tl]}")
            print(f"  Confidence: {conf_tl*100:.2f}%")

            # แสดง Top-3
            print(f"\nTop 3 Predictions (CNN):")
            top_idx = np.argsort(probs_cnn)[-3:][::-1]
            for i, idx in enumerate(top_idx, 1):
                print(f"  {i}. {LABEL_MAP[idx]}: {probs_cnn[idx]*100:.2f}%")

            print(f"\nTop 3 Predictions (Transfer):")
            top_idx = np.argsort(probs_tl)[-3:][::-1]
            for i, idx in enumerate(top_idx, 1):
                print(f"  {i}. {LABEL_MAP[idx]}: {probs_tl[idx]*100:.2f}%")

            print(f"{'='*60}")

        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {str(e)}")


# ===== Main Function =====
def main():
    parser = argparse.ArgumentParser(
        description='Sign Language Recognition - Model Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ทดสอบ CNN model กับ test dataset
  python test_model.py --mode test --model cnn

  # ทำนายภาพเดียว
  python test_model.py --mode predict --model transfer --image path/to/image.jpg

  # โหมดแบบโต้ตอบ
  python test_model.py --mode interactive

  # เปรียบเทียบทั้งสองโมเดล
  python test_model.py --mode compare
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'predict', 'interactive', 'compare'],
        default='interactive',
        help='โหมดการทำงาน'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['cnn', 'transfer', 'both'],
        default='cnn',
        help='ชื่อโมเดล'
    )

    parser.add_argument(
        '--image',
        type=str,
        help='path ของภาพสำหรับโหมด predict'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SIGN LANGUAGE RECOGNITION - MODEL TESTING")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    try:
        if args.mode == 'test':
            if args.model == 'both':
                print("\n📊 ทดสอบทั้งสองโมเดล...")
                acc_cnn, _ = test_with_dataset('cnn')
                acc_tl, _ = test_with_dataset('transfer')

                print(f"\n{'='*60}")
                print("FINAL COMPARISON")
                print(f"{'='*60}")
                print(f"CNN Accuracy: {acc_cnn*100:.2f}%")
                print(f"Transfer Learning Accuracy: {acc_tl*100:.2f}%")
                print(f"Best Model: {'Transfer Learning' if acc_tl > acc_cnn else 'CNN'}")
                print(f"{'='*60}")
            else:
                test_with_dataset(args.model)

        elif args.mode == 'predict':
            if not args.image:
                print("❌ กรุณาระบุ --image สำหรับโหมด predict")
                return

            if args.model == 'both':
                print("\n🔍 ทำนายด้วยทั้งสองโมเดล...\n")
                pred_cnn, conf_cnn = predict_single_image(args.image, 'cnn')
                pred_tl, conf_tl = predict_single_image(args.image, 'transfer')

                print(f"\n{'='*60}")
                print("COMPARISON")
                print(f"{'='*60}")
                print(f"CNN: {LABEL_MAP[pred_cnn]} ({conf_cnn*100:.2f}%)")
                print(f"Transfer: {LABEL_MAP[pred_tl]} ({conf_tl*100:.2f}%)")
                print(f"Agreement: {'✅ Yes' if pred_cnn == pred_tl else '❌ No'}")
                print(f"{'='*60}")
            else:
                predict_single_image(args.image, args.model)

        elif args.mode == 'compare':
            print("\n📊 เปรียบเทียบประสิทธิภาพทั้งสองโมเดล...")
            acc_cnn, _ = test_with_dataset('cnn')
            acc_tl, _ = test_with_dataset('transfer')

            print(f"\n{'='*60}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"{'Model':<30} {'Test Accuracy':<15}")
            print("-"*60)
            print(f"{'CNN (Custom)':<30} {acc_cnn*100:>6.2f}%")
            print(f"{'Transfer Learning (MobileNetV2)':<30} {acc_tl*100:>6.2f}%")
            print(f"{'-'*60}")
            print(f"{'Winner:':<30} {'Transfer Learning' if acc_tl > acc_cnn else 'CNN'}")
            print(f"{'Difference:':<30} {abs(acc_tl - acc_cnn)*100:>6.2f}%")
            print(f"{'='*60}")

        else:  # interactive
            interactive_mode()

    except KeyboardInterrupt:
        print("\n\n👋 ยกเลิกการทำงาน")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
