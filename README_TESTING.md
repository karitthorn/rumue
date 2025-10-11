# 🧪 Sign Language Recognition - Complete Testing Guide

> คู่มือการทดสอบโมเดล AI ภาษามืออย่างครบถ้วน

---

## 📦 ไฟล์ทดสอบที่มีให้

| ไฟล์ | ความสามารถ | ความยากง่าย | เวลาที่ใช้ |
|------|-----------|------------|-----------|
| `simple_test.py` | ทดสอบรวดเร็ว + เมนูง่าย | ⭐ Easy | ~2 นาที |
| `test_model.py` | ทดสอบครบถ้วน + Interactive | ⭐⭐ Medium | ~5-10 นาที |
| `benchmark_model.py` | วิเคราะห์ประสิทธิภาพ | ⭐⭐⭐ Advanced | ~10-15 นาที |
| `webcam_test.py` | ทดสอบ Real-time ด้วยกล้อง | ⭐⭐ Medium | แบบ Real-time |

---

## 🚀 Quick Start (เริ่มต้นง่ายๆ)

### สำหรับผู้เริ่มต้น - ใช้เวลา 2 นาที

```bash
# 1. ทดสอบรวดเร็ว
python simple_test.py

# เลือก: 1 (Quick Test)
# ผลลัพธ์จะแสดงทันที + ภาพตัวอย่าง
```

### สำหรับผู้ที่ต้องการทดสอบละเอียด - ใช้เวลา 10 นาที

```bash
# 1. ทดสอบกับ test dataset ทั้งหมด
python test_model.py --mode test --model both

# 2. วิเคราะห์ประสิทธิภาพ
python benchmark_model.py

# 3. ดูรายงาน
cat benchmark_report.txt
```

### สำหรับทดสอบ Real-time - ต้องมีกล้อง

```bash
# ทดสอบด้วย CNN model
python webcam_test.py --model cnn

# ทดสอบด้วย Transfer Learning model
python webcam_test.py --model transfer
```

---

## 📖 รายละเอียดแต่ละไฟล์

### 1. 🎯 `simple_test.py` - ทดสอบอย่างง่าย

**จุดเด่น:**
- มีเมนูแบบโต้ตอบ
- ใช้งานง่าย
- ผลลัพธ์ออกเร็ว
- เหมาะสำหรับมือใหม่

**ฟีเจอร์:**
```
1. Quick Test        → ทดสอบรวดเร็ว 20 ภาพ
2. Compare Models    → เปรียบเทียบ CNN vs Transfer
3. Test Specific     → ทดสอบตัวอักษรเฉพาะ (A-Y)
4. Exit             → ออกจากโปรแกรม
```

**วิธีใช้:**
```bash
python simple_test.py

# ตัวอย่าง: ทดสอบตัวอักษร 'A'
# เลือก: 3
# พิมพ์: A
```

**ผลลัพธ์:**
- แสดงความแม่นยำ
- แสดงภาพตัวอย่าง (สีเขียว = ถูก, สีแดง = ผิด)
- บันทึกภาพที่ `quick_test_results.png`

---

### 2. 🔬 `test_model.py` - ทดสอบหลัก (Full-Featured)

**จุดเด่น:**
- ทดสอบครบถ้วน
- หลายโหมด
- แสดง Classification Report
- Confusion Matrix

**โหมดต่างๆ:**

#### Mode 1: Test with Dataset
```bash
python test_model.py --mode test --model cnn
python test_model.py --mode test --model transfer
python test_model.py --mode test --model both
```

**ผลลัพธ์:**
- Test Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix (.png)
- Classification Report
- ภาพตัวอย่าง 10 ภาพ

---

#### Mode 2: Predict Single Image
```bash
python test_model.py --mode predict --model cnn --image path/to/image.jpg
python test_model.py --mode predict --model both --image path/to/image.jpg
```

**ผลลัพธ์:**
- การทำนาย + ความมั่นใจ
- Top-5 predictions
- ภาพแสดงผล

---

#### Mode 3: Interactive Mode (แนะนำ!)
```bash
python test_model.py --mode interactive
```

**ความสามารถ:**
- ทดสอบภาพสุ่มจาก dataset (พิมพ์: `random`)
- ทดสอบภาพของคุณ (ใส่ path)
- เปรียบเทียบทั้งสองโมเดลพร้อมกัน
- ออกจากโปรแกรม (พิมพ์: `quit`)

**ตัวอย่างการใช้งาน:**
```bash
$ python test_model.py --mode interactive

📷 ใส่ path ของภาพ (หรือคำสั่ง): random
# → แสดงภาพสุ่ม + การทำนาย

📷 ใส่ path ของภาพ (หรือคำสั่ง): /Users/you/hand_sign.jpg
# → ทำนายภาพของคุณ

📷 ใส่ path ของภาพ (หรือคำสั่ง): quit
# → ออกจากโปรแกรม
```

---

#### Mode 4: Compare Models
```bash
python test_model.py --mode compare
```

**ผลลัพธ์:**
- ทดสอบทั้งสองโมเดล
- เปรียบเทียบความแม่นยำ
- บอกโมเดลไหนดีกว่า

---

### 3. 📊 `benchmark_model.py` - วิเคราะห์ประสิทธิภาพ

**จุดเด่น:**
- วิเคราะห์เชิงลึก
- วัดความเร็ว
- สร้างรายงานละเอียด

**วิธีใช้:**
```bash
python benchmark_model.py
```

**สิ่งที่วัด:**

1. **Accuracy Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-Score

2. **Speed Metrics**
   - Model Load Time
   - Preprocessing Time
   - Total Prediction Time
   - Average Inference Time per Image
   - Throughput (images/second)

3. **Model Info**
   - File Size (MB)
   - Total Parameters
   - Per-class Performance

4. **Analysis**
   - Best/Worst performing letters
   - Confusion patterns
   - Most confused pairs

**ไฟล์ที่สร้าง:**
- `benchmark_comparison.png` - กราฟเปรียบเทียบ 6 แบบ
- `confusion_matrices.png` - Confusion Matrix
- `benchmark_report.txt` - รายงานโดยละเอียด

**ตัวอย่างผลลัพธ์:**
```
Model              Accuracy (%)  Precision (%)  Inference (ms)  Size (MB)
CNN Model          97.23         97.15          12.45           7.23
Transfer Learning  98.56         98.48          23.67           27.68

Winner: Transfer Learning
```

---

### 4. 📹 `webcam_test.py` - Real-time Testing

**จุดเด่น:**
- ทดสอบแบบ Real-time
- ใช้กล้อง webcam
- แสดงผลสด
- บันทึกภาพได้

**ข้อกำหนด:**
- ต้องมีกล้อง webcam
- ติดตั้ง OpenCV (มีใน requirements.txt แล้ว)

**วิธีใช้:**
```bash
# CNN Model
python webcam_test.py --model cnn

# Transfer Learning Model
python webcam_test.py --model transfer

# เลือกกล้องอื่น (ถ้ามีหลายตัว)
python webcam_test.py --model cnn --camera 1
```

**การควบคุม:**
- **SPACE** - ถ่ายภาพและทำนาย
- **C** - ล้างประวัติการทำนาย
- **S** - บันทึกภาพ (ไปที่โฟลเดอร์ `webcam_captures/`)
- **Q** หรือ **ESC** - ออกจากโปรแกรม

**ฟีเจอร์:**
- แสดงกล่องสำหรับวางมือ
- แสดงการทำนายปัจจุบัน + ความมั่นใจ
- แสดง Top-3 predictions
- แสดงการทำนายเฉลี่ยจากประวัติ (ช่วยลด noise)
- แสดงประวัติ 5 ครั้งล่าสุด
- บันทึกภาพพร้อม timestamp

**วิธีใช้ให้ได้ผลดี:**
1. วางมือในกล่องสีเขียว
2. ทำท่าภาษามือชัดเจน
3. กด SPACE เพื่อทำนาย
4. ลองหลายๆ ครั้งเพื่อดูการทำนายเฉลี่ย

---

## 🎓 สถานการณ์การใช้งาน

### สถานการณ์ 1: ฉันเพิ่งเทรนโมเดลเสร็จ อยากทดสอบให้เร็วที่สุด

```bash
python simple_test.py
# เลือก 1 (Quick Test)
```

**เวลาที่ใช้:** 1-2 นาที
**ผลลัพธ์:** ความแม่นยำโดยประมาณ + ภาพตัวอย่าง

---

### สถานการณ์ 2: อยากรู้ว่าโมเดลไหนดีกว่า

```bash
python simple_test.py
# เลือก 2 (Compare Models)

# หรือ
python test_model.py --mode compare
```

**เวลาที่ใช้:** 3-5 นาที
**ผลลัพธ์:** เปรียบเทียบ CNN vs Transfer Learning

---

### สถานการณ์ 3: อยากทดสอบภาพของฉันเอง

```bash
python test_model.py --mode interactive
# พิมพ์ path ของภาพ
```

**เวลาที่ใช้:** เท่าที่อยากทดสอบ
**ผลลัพธ์:** การทำนายพร้อมความมั่นใจ

---

### สถานการณ์ 4: อยากรายงานผลอย่างละเอียด

```bash
python benchmark_model.py
```

**เวลาที่ใช้:** 10-15 นาที
**ผลลัพธ์:** รายงานครบถ้วน + กราฟ + ไฟล์ .txt

---

### สถานการณ์ 5: อยากทดสอบแบบ Real-time

```bash
python webcam_test.py --model cnn
```

**เวลาที่ใช้:** เท่าที่อยากทดสอบ
**ผลลัพธ์:** ทดสอบสดด้วยกล้อง

---

### สถานการณ์ 6: อยากรู้ว่าตัวอักษรไหนทำนายยาก

```bash
python simple_test.py
# เลือก 3
# ลองทดสอบแต่ละตัวอักษร
```

**หรือ:**

```bash
python benchmark_model.py
# ดูที่ "Top 5 Worst" ในรายงาน
```

---

## 📊 การตีความผลลัพธ์

### Accuracy (ความแม่นยำ)

| Range | ประเมิน | ควรทำอย่างไร |
|-------|---------|--------------|
| > 98% | ดีเยี่ยม | พร้อมใช้งานจริง |
| 95-98% | ดีมาก | ใช้งานได้ |
| 90-95% | ดี | อาจปรับปรุง |
| < 90% | ควรปรับปรุง | เทรนใหม่/ปรับ model |

### Inference Time (เวลาทำนาย)

| Time | ประเมิน | เหมาะกับ |
|------|---------|----------|
| < 10ms | เร็วมาก | Real-time app, Mobile |
| 10-20ms | เร็ว | Real-time app |
| 20-50ms | ปานกลาง | Web app, Desktop |
| > 50ms | ช้า | Batch processing only |

### Model Size (ขนาดไฟล์)

| Size | ประเมิน | เหมาะกับ |
|------|---------|----------|
| < 5MB | เล็กมาก | Mobile app, Edge device |
| 5-20MB | เล็ก-กลาง | Mobile, Web |
| 20-50MB | กลาง | Desktop, Server |
| > 50MB | ใหญ่ | Server only |

---

## 🔍 ตัวอย่างผลลัพธ์

### ตัวอย่างที่ 1: Quick Test

```bash
$ python simple_test.py

📊 RESULTS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total Samples: 20
   Correct: 19
   Incorrect: 1
   Accuracy: 95.00%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ บันทึกผลลัพธ์ที่: quick_test_results.png
```

### ตัวอย่างที่ 2: Interactive Mode

```bash
$ python test_model.py --mode interactive

📷 ใส่ path ของภาพ: random

ภาพที่สุ่มได้: Index 1234
True Label: A

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
True Label: A
CNN Prediction: A (97.82%) - ✅
Transfer Prediction: A (99.12%) - ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### ตัวอย่างที่ 3: Benchmark

```
BENCHMARKING: CNN Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Performance Metrics:
   Accuracy:  97.23%
   Precision: 97.15%
   Recall:    97.20%
   F1-Score:  97.17%

⏱️  Speed Metrics:
   Avg Inference Time:  12.45 ± 2.31 ms
   Throughput:          80.32 images/second

💾 Model Info:
   File Size:        7.23 MB
   Total Parameters: 621,049

🎯 Per-Class Performance (Top 5 Best):
   Letter  Accuracy  Samples
   Q       100.00%   289
   R       99.69%    293
   W       99.43%    278
   K       99.12%    267
   L       98.95%    285

⚠️  Per-Class Performance (Top 5 Worst):
   Letter  Accuracy  Samples
   M       92.34%    256
   N       93.12%    271
   S       94.23%    265
   P       94.87%    243
   U       95.34%    279
```

---

## 🎯 Workflow แนะนำ

### สำหรับผู้เริ่มต้น

```
1. ทดสอบรวดเร็ว
   ├─> python simple_test.py (option 1)
   └─> ดู Accuracy

2. ถ้าพอใจผลลัพธ์
   ├─> python test_model.py --mode interactive
   └─> ทดสอบภาพของตัวเอง

3. ถ้าอยากทดสอบ Real-time
   └─> python webcam_test.py --model cnn
```

### สำหรับผู้ที่ต้องการรายงาน

```
1. ทดสอบครบถ้วน
   └─> python test_model.py --mode test --model both

2. วิเคราะห์ประสิทธิภาพ
   └─> python benchmark_model.py

3. อ่านรายงาน
   ├─> benchmark_report.txt
   ├─> benchmark_comparison.png
   └─> confusion_matrices.png
```

### สำหรับการพัฒนาต่อ

```
1. หาจุดอ่อน
   └─> python benchmark_model.py
   └─> ดู "Worst performing letters"

2. ทดสอบตัวอักษรที่อ่อน
   └─> python simple_test.py (option 3)

3. วิเคราะห์ Confusion Patterns
   └─> ดู confusion_matrices.png

4. ปรับปรุงโมเดล
   └─> กลับไปเทรนใหม่
```

---

## 🐛 การแก้ปัญหา

### ปัญหา 1: ไม่พบโมเดล

**Error:**
```
❌ ไม่พบไฟล์โมเดล: sign_language_cnn_final.keras
```

**วิธีแก้:**
```bash
# ตรวจสอบว่ามีไฟล์โมเดลหรือไม่
ls -lh *.keras

# ถ้าไม่มี ให้รันโน้ตบุ๊คก่อน
jupyter notebook sign_language_model.ipynb
# รันจนจบทุก cell
```

---

### ปัญหา 2: ไม่พบ Test Dataset

**Error:**
```
❌ ไม่พบไฟล์: ~/.cache/kagglehub/...
```

**วิธีแก้:**
```python
# รันโค้ดใน notebook เพื่อดาวน์โหลด
import kagglehub
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
```

---

### ปัญหา 3: Webcam ไม่เปิด

**Error:**
```
❌ ไม่สามารถเปิดกล้อง!
```

**วิธีแก้:**
```bash
# 1. ลองกล้องอื่น
python webcam_test.py --model cnn --camera 1

# 2. ตรวจสอบว่ากล้องทำงาน
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# 3. ตรวจสอบ permissions (macOS)
# System Preferences > Security & Privacy > Camera
```

---

### ปัญหา 4: Out of Memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**วิธีแก้:**
```bash
# 1. ทดสอบทีละโมเดล
python test_model.py --mode test --model cnn  # ไม่ใช้ 'both'

# 2. ลดขนาด batch (ถ้าเทรนใหม่)
# ใน notebook: batch_size = 64  # แทน 128

# 3. ปิดโปรแกรมอื่น
```

---

## 📝 Checklist

### ก่อนเริ่มทดสอบ

- [ ] เทรนโมเดลเสร็จแล้ว
- [ ] มีไฟล์ `sign_language_cnn_final.keras`
- [ ] มีไฟล์ `sign_language_transfer_final.keras` (ถ้าต้องการทดสอบ)
- [ ] ดาวน์โหลด test dataset แล้ว
- [ ] ติดตั้ง dependencies (`pip install -r requirements.txt`)

### การทดสอบทั่วไป

- [ ] Quick Test (simple_test.py)
- [ ] ทดสอบกับ full test set
- [ ] เปรียบเทียบทั้งสองโมเดล
- [ ] ดู Confusion Matrix
- [ ] อ่าน Classification Report

### การทดสอบเชิงลึก

- [ ] Run benchmark
- [ ] วิเคราะห์ per-class performance
- [ ] หา confusion patterns
- [ ] วัดความเร็ว (inference time)
- [ ] อ่านรายงานโดยละเอียด

### การทดสอบพิเศษ

- [ ] ทดสอบกับภาพของตัวเอง
- [ ] ทดสอบแต่ละตัวอักษร
- [ ] ทดสอบ Real-time ด้วย webcam
- [ ] บันทึกผลลัพธ์

---

## 🎓 คำแนะนำเพิ่มเติม

### เพื่อผลลัพธ์ที่ดี

1. **ภาพที่ใช้ทดสอบ:**
   - พื้นหลังควรเรียบง่าย
   - มือควรชัดเจน
   - ท่าทางตรงกับ training data

2. **Real-time Testing:**
   - แสงสว่างเพียงพอ
   - มือในกรอบ
   - ท่าทางชัดเจน
   - ทดสอบหลายๆ ครั้ง

3. **การตีความผล:**
   - ดู Confidence ด้วย ไม่ใช่แค่ Accuracy
   - ตรวจสอบ Confusion Matrix
   - ดูว่าตัวอักษรไหนสับสนกัน

### เพื่อประสิทธิภาพสูงสุด

1. **เลือกโมเดลที่เหมาะสม:**
   - Real-time app → CNN (เร็วกว่า)
   - Accuracy สำคัญที่สุด → Transfer Learning
   - Mobile app → CNN (เล็กกว่า)

2. **ปรับปรุงโมเดล:**
   - ดูตัวอักษรที่ทำนายผิดบ่อย
   - เพิ่ม training data สำหรับตัวอักษรนั้น
   - ลอง data augmentation เพิ่ม

---

## 📞 ต้องการความช่วยเหลือ?

1. **อ่าน TEST_INSTRUCTIONS.md** (ไฟล์นี้)
2. **ดู error message ที่แสดง**
3. **ตรวจสอบ checklist**
4. **ลองใช้ simple_test.py ก่อน**

---

**สร้างโดย:** Claude Code
**เวอร์ชัน:** 1.0
**วันที่:** 2024

---

**Happy Testing! 🎉🤖**
