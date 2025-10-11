"""
Benchmark & Performance Analysis
วิเคราะห์ประสิทธิภาพโมเดลอย่างละเอียด

วิธีใช้:
    python benchmark_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ตั้งค่า style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}


def load_test_data():
    """โหลด test dataset"""
    cache_dir = Path.home() / ".cache/kagglehub/datasets/datamunge/sign-language-mnist/versions/1"
    test_csv = cache_dir / "sign_mnist_test.csv"

    if not test_csv.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {test_csv}")

    df = pd.read_csv(test_csv)
    X = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y = df['label'].values

    return X, y


def benchmark_model(model_path, model_name, X_test, y_test, is_transfer=False):
    """วัดประสิทธิภาพของโมเดล"""
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARKING: {model_name}")
    print(f"{'='*70}\n")

    # โหลดโมเดล
    print("1️⃣ Loading model...")
    load_start = time.time()
    model = keras.models.load_model(model_path)
    load_time = time.time() - load_start
    print(f"   ✅ Load time: {load_time:.4f} seconds")

    # เตรียมข้อมูล
    if is_transfer:
        print("\n2️⃣ Preparing data for transfer learning...")
        prep_start = time.time()
        X_input = np.repeat(X_test, 3, axis=-1)
        X_input = tf.image.resize(X_input, [96, 96]).numpy()
        prep_time = time.time() - prep_start
        print(f"   ✅ Preprocessing time: {prep_time:.4f} seconds")
    else:
        X_input = X_test
        prep_time = 0

    # ทำนาย (รวมทั้งหมด)
    print("\n3️⃣ Making predictions (full dataset)...")
    pred_start = time.time()
    predictions = model.predict(X_input, verbose=0)
    pred_time = time.time() - pred_start
    pred_classes = np.argmax(predictions, axis=1)
    print(f"   ✅ Prediction time: {pred_time:.4f} seconds")
    print(f"   ⏱️  Average per image: {(pred_time/len(X_test))*1000:.2f} ms")

    # คำนวณ throughput
    throughput = len(X_test) / pred_time
    print(f"   🚀 Throughput: {throughput:.2f} images/second")

    # ทดสอบ inference time (ภาพเดียว)
    print("\n4️⃣ Testing single image inference...")
    single_times = []
    for i in range(100):
        start = time.time()
        _ = model.predict(X_input[i:i+1], verbose=0)
        single_times.append(time.time() - start)

    avg_single = np.mean(single_times) * 1000
    std_single = np.std(single_times) * 1000
    print(f"   ✅ Average: {avg_single:.2f} ± {std_single:.2f} ms")
    print(f"   📈 Min: {np.min(single_times)*1000:.2f} ms")
    print(f"   📉 Max: {np.max(single_times)*1000:.2f} ms")

    # ประเมินความแม่นยำ
    print("\n5️⃣ Evaluating accuracy...")
    accuracy = accuracy_score(y_test, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_classes, average='weighted', zero_division=0
    )

    print(f"   ✅ Accuracy:  {accuracy*100:.2f}%")
    print(f"   ✅ Precision: {precision*100:.2f}%")
    print(f"   ✅ Recall:    {recall*100:.2f}%")
    print(f"   ✅ F1-Score:  {f1*100:.2f}%")

    # วิเคราะห์แต่ละ class
    print("\n6️⃣ Analyzing per-class performance...")
    per_class_metrics = []

    available_labels = sorted(np.unique(y_test))

    for label in available_labels:
        mask = y_test == label
        class_acc = accuracy_score(y_test[mask], pred_classes[mask])

        per_class_metrics.append({
            'Letter': LABELS[label],
            'Label': label,
            'Accuracy': class_acc * 100,
            'Samples': np.sum(mask)
        })

    df_metrics = pd.DataFrame(per_class_metrics)
    print("\n   Top 5 Best:")
    print(df_metrics.nlargest(5, 'Accuracy')[['Letter', 'Accuracy', 'Samples']].to_string(index=False))

    print("\n   Top 5 Worst:")
    print(df_metrics.nsmallest(5, 'Accuracy')[['Letter', 'Accuracy', 'Samples']].to_string(index=False))

    # ขนาดโมเดล
    print("\n7️⃣ Model size...")
    model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   📦 File size: {model_size:.2f} MB")

    # นับ parameters
    total_params = model.count_params()
    print(f"   🔢 Total parameters: {total_params:,}")

    # สรุปผล
    results = {
        'model_name': model_name,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'load_time': load_time,
        'prep_time': prep_time,
        'pred_time': pred_time,
        'avg_inference_time': avg_single,
        'std_inference_time': std_single,
        'throughput': throughput,
        'model_size_mb': model_size,
        'total_params': total_params,
        'per_class_metrics': df_metrics,
        'predictions': pred_classes,
        'probabilities': predictions
    }

    return results


def compare_benchmarks(results_list):
    """เปรียบเทียบผลการวัดประสิทธิภาพ"""
    print(f"\n{'='*70}")
    print("⚖️  BENCHMARK COMPARISON")
    print(f"{'='*70}\n")

    # สร้างตาราง
    comparison_data = []
    for r in results_list:
        comparison_data.append({
            'Model': r['model_name'],
            'Accuracy (%)': f"{r['accuracy']:.2f}",
            'Precision (%)': f"{r['precision']:.2f}",
            'Recall (%)': f"{r['recall']:.2f}",
            'F1-Score (%)': f"{r['f1']:.2f}",
            'Inference (ms)': f"{r['avg_inference_time']:.2f}",
            'Throughput (img/s)': f"{r['throughput']:.1f}",
            'Size (MB)': f"{r['model_size_mb']:.2f}"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    # แสดงกราฟเปรียบเทียบ
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # 1. Performance Metrics
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row = i // 2
        col = i % 2
        values = [r[metric] for r in results_list]
        names = [r['model_name'] for r in results_list]

        axes[row, col].bar(names, values, color=['#3498db', '#e74c3c'][:len(names)])
        axes[row, col].set_ylabel(f'{name} (%)')
        axes[row, col].set_title(f'{name} Comparison')
        axes[row, col].set_ylim([0, 105])

        for j, v in enumerate(values):
            axes[row, col].text(j, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Inference Time
    names = [r['model_name'] for r in results_list]
    times = [r['avg_inference_time'] for r in results_list]

    axes[0, 2].bar(names, times, color=['#2ecc71', '#f39c12'][:len(names)])
    axes[0, 2].set_ylabel('Time (ms)')
    axes[0, 2].set_title('Average Inference Time')

    for j, v in enumerate(times):
        axes[0, 2].text(j, v + 0.5, f'{v:.2f}ms', ha='center', va='bottom', fontweight='bold')

    # 3. Model Size
    sizes = [r['model_size_mb'] for r in results_list]

    axes[1, 2].bar(names, sizes, color=['#9b59b6', '#1abc9c'][:len(names)])
    axes[1, 2].set_ylabel('Size (MB)')
    axes[1, 2].set_title('Model Size')

    for j, v in enumerate(sizes):
        axes[1, 2].text(j, v + 0.5, f'{v:.2f}MB', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ บันทึกกราฟที่: benchmark_comparison.png")
    plt.show()


def analyze_confusion_patterns(results_list, y_test):
    """วิเคราะห์ Confusion Patterns"""
    print(f"\n{'='*70}")
    print("🔍 CONFUSION PATTERN ANALYSIS")
    print(f"{'='*70}\n")

    available_labels = sorted(np.unique(y_test))
    target_names = [LABELS[i] for i in available_labels]

    fig, axes = plt.subplots(1, len(results_list), figsize=(14*len(results_list), 12))
    if len(results_list) == 1:
        axes = [axes]

    for i, result in enumerate(results_list):
        cm = confusion_matrix(y_test, result['predictions'], labels=available_labels)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=axes[i],
                    cbar_kws={'label': 'Count'})

        axes[i].set_title(f"{result['model_name']}\nConfusion Matrix", fontsize=14, weight='bold')
        axes[i].set_ylabel('True Label', fontsize=12)
        axes[i].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f"✅ บันทึกที่: confusion_matrices.png")
    plt.show()

    # หาคู่ที่สับสนกันมากที่สุด
    for result in results_list:
        print(f"\n{result['model_name']} - Most Confused Pairs:")
        cm = confusion_matrix(y_test, result['predictions'], labels=available_labels)

        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'True': LABELS[available_labels[i]],
                        'Predicted': LABELS[available_labels[j]],
                        'Count': cm[i, j]
                    })

        df_confused = pd.DataFrame(confused_pairs).nlargest(10, 'Count')
        print(df_confused.to_string(index=False))


def generate_report(results_list, y_test):
    """สร้างรายงานสรุป"""
    print(f"\n{'='*70}")
    print("📄 GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*70}\n")

    report = []

    report.append("="*70)
    report.append("SIGN LANGUAGE RECOGNITION - BENCHMARK REPORT")
    report.append("="*70)
    report.append("")

    for result in results_list:
        report.append(f"\n{'-'*70}")
        report.append(f"MODEL: {result['model_name']}")
        report.append(f"{'-'*70}")
        report.append(f"\n📊 Performance Metrics:")
        report.append(f"   Accuracy:  {result['accuracy']:.2f}%")
        report.append(f"   Precision: {result['precision']:.2f}%")
        report.append(f"   Recall:    {result['recall']:.2f}%")
        report.append(f"   F1-Score:  {result['f1']:.2f}%")

        report.append(f"\n⏱️  Speed Metrics:")
        report.append(f"   Model Load Time:     {result['load_time']:.4f}s")
        report.append(f"   Preprocessing Time:  {result['prep_time']:.4f}s")
        report.append(f"   Total Pred Time:     {result['pred_time']:.4f}s")
        report.append(f"   Avg Inference Time:  {result['avg_inference_time']:.2f} ± {result['std_inference_time']:.2f} ms")
        report.append(f"   Throughput:          {result['throughput']:.2f} images/second")

        report.append(f"\n💾 Model Info:")
        report.append(f"   File Size:        {result['model_size_mb']:.2f} MB")
        report.append(f"   Total Parameters: {result['total_params']:,}")

        report.append(f"\n🎯 Per-Class Performance (Top 5 Best):")
        top_5 = result['per_class_metrics'].nlargest(5, 'Accuracy')
        for _, row in top_5.iterrows():
            report.append(f"   {row['Letter']}: {row['Accuracy']:.2f}% ({row['Samples']} samples)")

        report.append(f"\n⚠️  Per-Class Performance (Top 5 Worst):")
        bottom_5 = result['per_class_metrics'].nsmallest(5, 'Accuracy')
        for _, row in bottom_5.iterrows():
            report.append(f"   {row['Letter']}: {row['Accuracy']:.2f}% ({row['Samples']} samples)")

    report.append(f"\n{'='*70}")
    report.append("END OF REPORT")
    report.append(f"{'='*70}")

    # บันทึกรายงาน
    report_text = '\n'.join(report)
    with open('benchmark_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✅ บันทึกรายงานที่: benchmark_report.txt")


def main():
    print("="*70)
    print("🚀 MODEL BENCHMARK & PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} device(s)")

    # โหลดข้อมูล
    print("\n📂 Loading test dataset...")
    try:
        X_test, y_test = load_test_data()
        print(f"✅ Loaded {len(X_test)} test samples")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 กรุณารันโค้ดใน notebook ก่อนเพื่อดาวน์โหลด dataset")
        return

    # ทดสอบแต่ละโมเดล
    results_list = []

    models = [
        ('sign_language_cnn_final.keras', 'CNN Model', False),
        ('sign_language_transfer_final.keras', 'Transfer Learning', True)
    ]

    for model_path, model_name, is_transfer in models:
        if Path(model_path).exists():
            try:
                result = benchmark_model(model_path, model_name, X_test, y_test, is_transfer)
                results_list.append(result)
            except Exception as e:
                print(f"❌ Error benchmarking {model_name}: {e}")
        else:
            print(f"⚠️  Model not found: {model_path}")

    if not results_list:
        print("\n❌ No models found to benchmark!")
        return

    # เปรียบเทียบผล
    if len(results_list) > 1:
        compare_benchmarks(results_list)
        analyze_confusion_patterns(results_list, y_test)

    # สร้างรายงาน
    generate_report(results_list, y_test)

    print("\n✅ Benchmark completed!")
    print("\nGenerated files:")
    print("   - benchmark_comparison.png")
    print("   - confusion_matrices.png")
    print("   - benchmark_report.txt")


if __name__ == '__main__':
    main()
