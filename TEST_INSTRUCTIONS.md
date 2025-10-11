# 📚 คู่มือการทดสอบโมเดล Sign Language Recognition

คู่มือนี้อธิบายวิธีการทดสอบโมเดล AI ที่คุณเทรนไว้แล้ว มี 3 สคริปต์หลักสำหรับการทดสอบ

---

## 📋 สารบัญ

1. [ไฟล์ที่จำเป็น](#ไฟล์ที่จำเป็น)
2. [สคริปต์ทดสอบ](#สคริปต์ทดสอบ)
3. [วิธีการใช้งาน](#วิธีการใช้งาน)
4. [ตัวอย่างการใช้งาน](#ตัวอย่างการใช้งาน)

---

## ไฟล์ที่จำเป็น

ก่อนเริ่มทดสอบ ตรวจสอบให้แน่ใจว่ามีไฟล์เหล่านี้:

```
rumue/
├── sign_language_cnn_final.keras          # โมเดล CNN
├── sign_language_transfer_final.keras     # โมเดล Transfer Learning
├── test_model.py                          # สคริปต์ทดสอบหลัก (ครบทุกฟีเจอร์)
├── simple_test.py                         # สคริปต์ทดสอบอย่างง่าย
├── benchmark_model.py                     # วิเคราะห์ประสิทธิภาพ
└── TEST_INSTRUCTIONS.md                   # ไฟล์นี้
```

---

## สคริปต์ทดสอบ

### 1. `test_model.py` - สคริปต์ทดสอบหลัก (แนะนำ)

**ความสามารถ:**
- ทดสอบกับ test dataset ทั้งหมด
- ทำนายภาพเดียว
- โหมดแบบโต้ตอบ (Interactive Mode)
- เปรียบเทียบโมเดล CNN vs Transfer Learning
- แสดง Confusion Matrix
- Classification Report

**วิธีใช้:**

```bash
# โหมดแบบโต้ตอบ (แนะนำสำหรับผู้เริ่มต้น)
python test_model.py --mode interactive

# ทดสอบกับ test dataset
python test_model.py --mode test --model cnn
python test_model.py --mode test --model transfer
python test_model.py --mode test --model both

# ทำนายภาพเดียว
python test_model.py --mode predict --model cnn --image path/to/image.jpg
python test_model.py --mode predict --model both --image path/to/image.jpg

# เปรียบเทียบทั้งสองโมเดล
python test_model.py --mode compare
```

---

### 2. `simple_test.py` - สคริปต์ทดสอบอย่างง่าย

**ความสามารถ:**
- ทดสอบรวดเร็ว (Quick Test)
- เปรียบเทียบโมเดล
- ทดสอบตัวอักษรเฉพาะ (A-Y ยกเว้น J)
- มีเมนูแบบโต้ตอบที่ใช้งานง่าย

**วิธีใช้:**

```bash
# รันแบบมีเมนู
python simple_test.py

# จะแสดงเมนูดังนี้:
# 1. Quick Test (ทดสอบรวดเร็ว)
# 2. Compare Models (เปรียบเทียบโมเดล)
# 3. Test Specific Letter (ทดสอบตัวอักษรเฉพาะ)
# 4. Exit
```

**ตัวอย่างผลลัพธ์:**
```
   1. True: A  | Pred: A  (98.5%) ✅
   2. True: B  | Pred: B  (95.2%) ✅
   3. True: C  | Pred: D  (72.1%) ❌
   ...

   Accuracy: 95.50%
```

---

### 3. `benchmark_model.py` - วิเคราะห์ประสิทธิภาพ

**ความสามารถ:**
- วัดความเร็วในการทำนาย (Inference Time)
- คำนวณ Throughput (ภาพ/วินาที)
- วิเคราะห์ประสิทธิภาพของแต่ละตัวอักษร
- หา Confusion Patterns
- สร้างรายงานโดยละเอียด
- เปรียบเทียบทั้งสองโมเดล

**วิธีใช้:**

```bash
python benchmark_model.py
```

**ไฟล์ที่สร้าง:**
- `benchmark_comparison.png` - กราฟเปรียบเทียบ
- `confusion_matrices.png` - Confusion Matrix
- `benchmark_report.txt` - รายงานโดยละเอียด

**ตัวอย่างผลลัพธ์:**
```
Model              Accuracy (%)  Inference (ms)  Throughput (img/s)  Size (MB)
CNN Model          97.23         12.45           80.3                7.23
Transfer Learning  98.56         23.67           42.2                27.68
```

---

## วิธีการใช้งาน

### A. ทดสอบแบบง่าย (สำหรับผู้เริ่มต้น)

1. **เริ่มต้นด้วย Simple Test:**
   ```bash
   python simple_test.py
   ```

2. เลือกตัวเลือก `1` (Quick Test)

3. ดูผลลัพธ์และภาพที่แสดง

---

### B. ทดสอบแบบโต้ตอบ (Interactive)

1. **รันโหมดโต้ตอบ:**
   ```bash
   python test_model.py --mode interactive
   ```

2. **ทดสอบภาพสุ่ม:**
   - พิมพ์: `random`
   - กด Enter

3. **ทดสอบภาพของคุณเอง:**
   - พิมพ์ path ของภาพ เช่น: `/Users/you/Downloads/hand_sign.jpg`
   - กด Enter

4. **ออกจากโปรแกรม:**
   - พิมพ์: `quit` หรือ `exit`

---

### C. ทดสอบครบถ้วน (Full Testing)

1. **ทดสอบกับ Test Dataset:**
   ```bash
   python test_model.py --mode test --model both
   ```

2. **วิเคราะห์ประสิทธิภาพ:**
   ```bash
   python benchmark_model.py
   ```

3. **ตรวจสอบผลลัพธ์:**
   - ดูกราฟที่สร้างขึ้น (.png)
   - อ่านรายงาน (benchmark_report.txt)

---

## ตัวอย่างการใช้งาน

### ตัวอย่างที่ 1: ทดสอบรวดเร็ว

```bash
$ python simple_test.py

# เลือก 1 (Quick Test)
# ผลลัพธ์:
📊 RESULTS SUMMARY
Total Samples: 20
Correct: 19
Incorrect: 1
Accuracy: 95.00%
```

---

### ตัวอย่างที่ 2: ทดสอบตัวอักษร "A"

```bash
$ python simple_test.py

# เลือก 3 (Test Specific Letter)
# พิมพ์: A
# ผลลัพธ์:
Accuracy for letter 'A': 98.89%
✅ บันทึกผลลัพธ์ที่: test_letter_A.png
```

---

### ตัวอย่างที่ 3: เปรียบเทียบโมเดล

```bash
$ python test_model.py --mode compare

# ผลลัพธ์:
MODEL COMPARISON SUMMARY
Model                          Test Accuracy
CNN (Custom)                   97.23%
Transfer Learning (MobileNetV2) 98.56%
Winner:                         Transfer Learning
Difference:                     1.33%
```

---

### ตัวอย่างที่ 4: Benchmark

```bash
$ python benchmark_model.py

# ผลลัพธ์:
📊 BENCHMARKING: CNN Model
✅ Accuracy:  97.23%
⏱️  Average per image: 12.45 ms
🚀 Throughput: 80.32 images/second
📦 File size: 7.23 MB
```

---

## 💡 เทคนิคการทดสอบ

### เคล็ดลับ 1: ทดสอบตัวอักษรที่สับสน
บางตัวอักษรอาจสับสนกัน เช่น M กับ N, P กับ Q

```bash
python simple_test.py
# เลือก 3 และทดสอบ M, N, P, Q
```

### เคล็ดลับ 2: ตรวจสอบภาพที่ทำนายผิด
ใน Interactive Mode สามารถดูภาพที่โมเดลทำนายผิดได้

```bash
python test_model.py --mode interactive
# พิมพ์ random หลายๆ ครั้ง
```

### เคล็ดลับ 3: เปรียบเทียบความเร็ว
ใช้ benchmark เพื่อเลือกโมเดลที่เหมาะกับการใช้งานจริง

```bash
python benchmark_model.py
# ดู "Inference Time" และ "Throughput"
```

---

## 🔧 การแก้ไขปัญหา

### ปัญหา: ไม่พบโมเดล
```
❌ ไม่พบไฟล์โมเดล: sign_language_cnn_final.keras
```

**วิธีแก้:**
- รันโค้ดใน `sign_language_model.ipynb` ให้จบก่อน
- ตรวจสอบว่าโมเดลถูกบันทึกในโฟลเดอร์เดียวกัน

---

### ปัญหา: ไม่พบ Test Dataset
```
❌ ไม่พบไฟล์ test data
```

**วิธีแก้:**
- รันเซลล์ในโน้ตบุ๊คที่ดาวน์โหลด dataset ก่อน
- ตรวจสอบว่ามีไฟล์ที่: `~/.cache/kagglehub/datasets/...`

---

### ปัญหา: Out of Memory
```
❌ ResourceExhaustedError: OOM when allocating tensor
```

**วิธีแก้:**
- ปิดโปรแกรมอื่นๆ
- ทดสอบทีละโมเดล (ไม่ใช้ `both`)
- ลดจำนวน samples ในการทดสอบ

---

## 📊 การตีความผลลัพธ์

### Accuracy (ความแม่นยำ)
- **> 95%**: ดีมาก
- **90-95%**: ดี
- **< 90%**: ควรปรับปรุงโมเดล

### Inference Time (เวลาทำนาย)
- **< 20ms**: เร็วมาก (เหมาะกับ Real-time)
- **20-50ms**: ปานกลาง
- **> 50ms**: ช้า

### Model Size (ขนาดโมเดล)
- **< 10MB**: เล็ก (เหมาะกับ Mobile/Edge)
- **10-50MB**: ปานกลาง
- **> 50MB**: ใหญ่

---

## 🎯 สรุป

| สคริปต์ | เหมาะกับ | ระดับความยาก |
|---------|---------|-------------|
| `simple_test.py` | ผู้เริ่มต้น, ทดสอบรวดเร็ว | ⭐ ง่าย |
| `test_model.py` | การทดสอบครบถ้วน | ⭐⭐ ปานกลาง |
| `benchmark_model.py` | วิเคราะห์เชิงลึก | ⭐⭐⭐ ขั้นสูง |

---

## 📝 Checklist การทดสอบ

- [ ] รัน `simple_test.py` เพื่อทดสอบรวดเร็ว
- [ ] ทดสอบกับ test dataset ทั้งหมด
- [ ] เปรียบเทียบ CNN vs Transfer Learning
- [ ] ตรวจสอบ Confusion Matrix
- [ ] วัดประสิทธิภาพด้วย benchmark
- [ ] ทดสอบกับภาพของคุณเอง
- [ ] อ่านรายงานโดยละเอียด
- [ ] ตรวจสอบตัวอักษรที่ทำนายผิดบ่อย

---

## 🚀 ขั้นตอนแนะนำสำหรับผู้เริ่มต้น

1. เริ่มต้นด้วย:
   ```bash
   python simple_test.py
   ```
   เลือก option 1 (Quick Test)

2. ต่อด้วย:
   ```bash
   python simple_test.py
   ```
   เลือก option 2 (Compare Models)

3. ทดสอบภาพของคุณเอง:
   ```bash
   python test_model.py --mode interactive
   ```

4. วิเคราะห์เชิงลึก:
   ```bash
   python benchmark_model.py
   ```

---

## 📞 ต้องการความช่วยเหลือ?

หากมีปัญหาหรือข้อสงสัย:

1. ตรวจสอบว่าเทรนโมเดลเสร็จแล้ว
2. ตรวจสอบว่ามีไฟล์โมเดล (.keras) อยู่
3. ตรวจสอบว่า test dataset ถูกดาวน์โหลดแล้ว
4. อ่าน error message ที่แสดง

---

**Happy Testing! 🎉**
