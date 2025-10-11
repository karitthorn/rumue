# 🚀 Quick Start - ทดสอบโมเดล 2 นาที

> เริ่มต้นทดสอบโมเดล Sign Language Recognition อย่างรวดเร็ว

---

## ✅ ก่อนเริ่ม

ตรวจสอบว่ามีไฟล์เหล่านี้:
```
✓ sign_language_cnn_final.keras
✓ sign_language_transfer_final.keras (optional)
✓ Test dataset (ดาวน์โหลดจาก notebook แล้ว)
```

---

## 🎯 เริ่มทดสอบ 3 ขั้นตอน

### 1️⃣ ทดสอบรวดเร็ว (1 นาที)

```bash
python simple_test.py
```

กด `1` แล้ว Enter → เห็นผลทันที!

**ผลลัพธ์:**
- ความแม่นยำ (Accuracy)
- ภาพตัวอย่าง 12 ภาพ
- ไฟล์: `quick_test_results.png`

---

### 2️⃣ ทดสอบภาพของคุณ (2 นาที)

```bash
python test_model.py --mode interactive
```

พิมพ์ `random` → ทดสอบภาพสุ่ม
พิมพ์ `quit` → ออก

**ผลลัพธ์:**
- เปรียบเทียบ CNN vs Transfer Learning
- แสดง Top-3 predictions
- ความมั่นใจ (Confidence)

---

### 3️⃣ ทดสอบด้วยกล้อง (Real-time)

```bash
python webcam_test.py --model cnn
```

**คีย์:**
- `SPACE` → ทำนาย
- `S` → บันทึกภาพ
- `Q` → ออก

---

## 📊 ต้องการรายงานครบถ้วน?

```bash
python benchmark_model.py
```

รอ 10 นาที → ได้รายงานละเอียด + กราฟ

**ไฟล์ที่ได้:**
- `benchmark_report.txt` - รายงานโดยละเอียด
- `benchmark_comparison.png` - กราฟเปรียบเทียบ
- `confusion_matrices.png` - Confusion Matrix

---

## 🎓 เลือกใช้ตามความต้องการ

| ต้องการ | ใช้คำสั่ง | เวลา |
|---------|-----------|------|
| ทดสอบรวดเร็ว | `python simple_test.py` | 1 นาที |
| ทดสอบภาพของตัวเอง | `python test_model.py --mode interactive` | 2-5 นาที |
| ทดสอบ Real-time | `python webcam_test.py --model cnn` | ตามต้องการ |
| รายงานครบถ้วน | `python benchmark_model.py` | 10 นาที |
| เปรียบเทียบโมเดล | `python test_model.py --mode compare` | 5 นาที |

---

## 💡 เคล็ดลับ

1. **เริ่มจาก simple_test.py** - ง่ายที่สุด
2. **ใช้ interactive mode** - ทดสอบภาพได้เยอะ
3. **รัน benchmark** - เมื่อต้องการรายงาน
4. **ลอง webcam** - สนุกดี!

---

## 🔧 แก้ปัญหา

### ไม่พบโมเดล
```bash
# รันโน้ตบุ๊คก่อน
jupyter notebook sign_language_model.ipynb
```

### ไม่พบ dataset
```python
# รันใน notebook
import kagglehub
kagglehub.dataset_download("datamunge/sign-language-mnist")
```

---

## 📚 อ่านเพิ่มเติม

- `TEST_INSTRUCTIONS.md` - คู่มือละเอียด
- `README_TESTING.md` - คู่มือครบถ้วน

---

**เริ่มทดสอบเลย! 🎉**
