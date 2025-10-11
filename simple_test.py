"""
Simple Sign Language Model Tester
โค้ดทดสอบอย่างง่ายสำหรับโมเดล Sign Language Recognition

วิธีใช้:
    python simple_test.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Label mapping
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}


def load_test_data():
    """โหลด test dataset"""
    print("📂 กำลังโหลด test data...")
    cache_dir = Path.home() / ".cache/kagglehub/datasets/datamunge/sign-language-mnist/versions/1"
    test_csv = cache_dir / "sign_mnist_test.csv"

    if not test_csv.exists():
        print(f"❌ ไม่พบไฟล์ test data ที่: {test_csv}")
        print("💡 กรุณารันโค้ดใน notebook ก่อนเพื่อดาวน์โหลด dataset")
        return None, None

    df = pd.read_csv(test_csv)
    X = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y = df['label'].values

    print(f"✅ โหลดข้อมูลสำเร็จ: {len(X)} ภาพ")
    return X, y


def test_model_quickly(model_path='sign_language_cnn_final.keras', n_samples=20):
    """ทดสอบโมเดลอย่างรวดเร็ว"""
    print("\n" + "="*70)
    print("🚀 QUICK MODEL TEST")
    print("="*70)

    # 1. โหลดโมเดล
    print(f"\n1️⃣ กำลังโหลดโมเดล: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print("   ✅ โหลดโมเดลสำเร็จ!")
    except Exception as e:
        print(f"   ❌ เกิดข้อผิดพลาด: {e}")
        return

    # 2. โหลด test data
    print("\n2️⃣ กำลังโหลดข้อมูลทดสอบ...")
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    # 3. สุ่มตัวอย่าง
    print(f"\n3️⃣ สุ่ม {n_samples} ภาพจาก test set...")
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    # 4. ทำนายและแสดงผล
    print("\n4️⃣ กำลังทำนาย...\n")

    correct = 0
    total = n_samples

    results = []

    for i, idx in enumerate(indices, 1):
        image = X_test[idx]
        true_label = y_test[idx]

        # ทำนาย
        pred = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class] * 100

        is_correct = pred_class == true_label
        if is_correct:
            correct += 1

        # แสดงผล
        status = "✅" if is_correct else "❌"
        print(f"   {i:2d}. True: {LABELS[true_label]:2s} | Pred: {LABELS[pred_class]:2s} ({confidence:5.1f}%) {status}")

        results.append({
            'idx': idx,
            'image': image,
            'true': true_label,
            'pred': pred_class,
            'conf': confidence,
            'correct': is_correct
        })

    # 5. สรุปผล
    accuracy = (correct / total) * 100
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"   Total Samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Incorrect: {total - correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")

    # 6. แสดงภาพตัวอย่าง
    print("📸 แสดงผลการทำนาย...\n")
    visualize_results(results[:min(12, len(results))])

    return results


def visualize_results(results):
    """แสดงผลการทำนายแบบกราฟ"""
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, result in enumerate(results):
        img = result['image'].reshape(28, 28)
        true_label = LABELS[result['true']]
        pred_label = LABELS[result['pred']]
        confidence = result['conf']
        is_correct = result['correct']

        axes[i].imshow(img, cmap='gray')

        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
        color = 'green' if is_correct else 'red'
        axes[i].set_title(title, fontsize=11, color=color, weight='bold')
        axes[i].axis('off')

    # ปิด axes ที่ไม่ได้ใช้
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=120, bbox_inches='tight')
    print("   ✅ บันทึกผลลัพธ์ที่: quick_test_results.png")
    plt.show()


def compare_models():
    """เปรียบเทียบทั้งสองโมเดล"""
    print("\n" + "="*70)
    print("⚖️  MODEL COMPARISON")
    print("="*70)

    # โหลดข้อมูล
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    # สุ่ม 100 ตัวอย่าง
    indices = np.random.choice(len(X_test), 100, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    models = {
        'CNN': 'sign_language_cnn_final.keras',
        'Transfer Learning': 'sign_language_transfer_final.keras'
    }

    results = {}

    for name, path in models.items():
        print(f"\n🔍 Testing {name}...")

        try:
            model = keras.models.load_model(path)

            # เตรียมข้อมูลสำหรับ Transfer Learning
            if 'transfer' in path.lower():
                X_input = np.repeat(X_sample, 3, axis=-1)
                X_input = tf.image.resize(X_input, [96, 96]).numpy()
            else:
                X_input = X_sample

            # ทำนาย
            predictions = model.predict(X_input, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)

            # คำนวณ accuracy
            correct = np.sum(pred_classes == y_sample)
            accuracy = (correct / len(y_sample)) * 100

            results[name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(y_sample)
            }

            print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct}/{len(y_sample)})")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = None

    # แสดงผลเปรียบเทียบ
    print(f"\n{'='*70}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*70}")

    for name, result in results.items():
        if result:
            print(f"   {name:20s}: {result['accuracy']:6.2f}%")

    if all(results.values()):
        best = max(results.items(), key=lambda x: x[1]['accuracy'] if x[1] else 0)
        print(f"\n   🏆 Winner: {best[0]}")

    print(f"{'='*70}\n")


def test_specific_letter(letter='A', n_samples=9):
    """ทดสอบกับตัวอักษรเฉพาะ"""
    print(f"\n{'='*70}")
    print(f"🔤 TESTING SPECIFIC LETTER: {letter}")
    print(f"{'='*70}")

    # แปลง letter เป็น label
    label = None
    for k, v in LABELS.items():
        if v == letter.upper():
            label = k
            break

    if label is None:
        print(f"❌ Invalid letter: {letter}")
        return

    # โหลดข้อมูล
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    # กรองเฉพาะตัวอักษรที่ต้องการ
    mask = y_test == label
    X_letter = X_test[mask]
    y_letter = y_test[mask]

    print(f"\n📊 พบตัวอักษร '{letter}' จำนวน {len(X_letter)} ภาพ")

    if len(X_letter) == 0:
        print(f"❌ ไม่พบตัวอักษร '{letter}' ใน test set")
        return

    # สุ่มตัวอย่าง
    n = min(n_samples, len(X_letter))
    indices = np.random.choice(len(X_letter), n, replace=False)

    # โหลดโมเดล
    print(f"\n🔍 กำลังทดสอบด้วย CNN model...")
    model = keras.models.load_model('sign_language_cnn_final.keras')

    # ทำนาย
    results = []
    correct = 0

    for idx in indices:
        image = X_letter[idx]
        pred = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class] * 100

        is_correct = pred_class == label
        if is_correct:
            correct += 1

        results.append({
            'image': image,
            'pred': pred_class,
            'conf': confidence,
            'correct': is_correct
        })

    accuracy = (correct / n) * 100
    print(f"\n   Accuracy for letter '{letter}': {accuracy:.2f}% ({correct}/{n})")

    # แสดงภาพ
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, result in enumerate(results[:9]):
        img = result['image'].reshape(28, 28)
        pred_label = LABELS[result['pred']]
        confidence = result['conf']
        is_correct = result['correct']

        axes[i].imshow(img, cmap='gray')
        title = f"{pred_label} ({confidence:.1f}%)"
        color = 'green' if is_correct else 'red'
        axes[i].set_title(title, fontsize=12, color=color, weight='bold')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Testing Letter '{letter}' - Accuracy: {accuracy:.1f}%",
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'test_letter_{letter}.png', dpi=120, bbox_inches='tight')
    print(f"\n   ✅ บันทึกผลลัพธ์ที่: test_letter_{letter}.png")
    plt.show()


def main_menu():
    """เมนูหลัก"""
    print("\n" + "="*70)
    print("🤖 SIGN LANGUAGE MODEL TESTER")
    print("="*70)
    print("\nเลือกโหมดทดสอบ:")
    print("  1. Quick Test (ทดสอบรวดเร็ว)")
    print("  2. Compare Models (เปรียบเทียบโมเดล)")
    print("  3. Test Specific Letter (ทดสอบตัวอักษรเฉพาะ)")
    print("  4. Exit")
    print("="*70)

    while True:
        choice = input("\n🔢 เลือกตัวเลือก (1-4): ").strip()

        if choice == '1':
            test_model_quickly()

        elif choice == '2':
            compare_models()

        elif choice == '3':
            letter = input("   พิมพ์ตัวอักษรที่ต้องการทดสอบ (A-Y, ยกเว้น J): ").strip().upper()
            if letter in LABELS.values():
                test_specific_letter(letter)
            else:
                print(f"   ❌ ตัวอักษร '{letter}' ไม่ถูกต้อง")

        elif choice == '4':
            print("\n👋 ขอบคุณที่ใช้งาน!\n")
            break

        else:
            print("   ❌ กรุณาเลือก 1-4 เท่านั้น")


if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}\n")

    # ตรวจสอบไฟล์โมเดล
    cnn_exists = Path('sign_language_cnn_final.keras').exists()
    transfer_exists = Path('sign_language_transfer_final.keras').exists()

    print("📁 ตรวจสอบไฟล์โมเดล:")
    print(f"   CNN Model: {'✅ Found' if cnn_exists else '❌ Not found'}")
    print(f"   Transfer Model: {'✅ Found' if transfer_exists else '❌ Not found'}")

    if not (cnn_exists or transfer_exists):
        print("\n❌ ไม่พบไฟล์โมเดล!")
        print("💡 กรุณารันโค้ดใน notebook ก่อนเพื่อเทรนโมเดล")
        exit(1)

    # เรียกเมนูหลัก
    main_menu()
