# Online Learning System - Sign Language Recognition

ระบบเรียนรู้ต่อเนื่อง (Online Learning) สำหรับโมเดล Sign Language Recognition ที่สามารถเทรนโมเดลใหม่ผ่าน feedback จาก webcam แบบ real-time

---

## 🎯 คุณสมบัติ

- ✅ **Interactive Webcam**: ใช้ webcam ทำนายภาษามือแบบ real-time
- ✅ **User Feedback**: แก้ไข prediction ที่ผิดและบันทึกเป็นข้อมูลเทรน
- ✅ **Incremental Learning**: เทรนโมเดลด้วยข้อมูลใหม่โดยไม่ต้องเทรนใหม่ทั้งหมด
- ✅ **Anti-Forgetting**: ใช้ replay buffer ป้องกันโมเดลลืมข้อมูลเดิม
- ✅ **Statistics Tracking**: ติดตามจำนวน feedback แต่ละตัวอักษร
- ✅ **Model Management**: บันทึกและโหลดโมเดลที่อัพเดทแล้ว

---

## 📁 ไฟล์ใหม่ที่เพิ่มเข้ามา

```
rumue/
├── interactive_webcam.py          # 🆕 Webcam app พร้อม feedback system
├── online_trainer.py              # 🆕 Module สำหรับ incremental training
├── feedback_data/                 # 🆕 เก็บข้อมูล feedback (auto-created)
│   ├── A/                         # แยกตาม label
│   ├── B/
│   ├── ...
│   ├── Y/
│   ├── feedback_log.csv           # บันทึก feedback history
│   └── training_log.json          # บันทึก training history
└── README_ONLINE_LEARNING.md      # 🆕 Documentation นี้
```

**ไม่มีการแก้ไขไฟล์เดิมใดๆ** - ระบบทำงานแยกอิสระ!

---

## 🚀 วิธีใช้งาน

### ขั้นตอนที่ 1: รัน Interactive Webcam

```bash
python interactive_webcam.py
```

หรือระบุโมเดลที่ต้องการใช้:

```bash
# ใช้ CNN model (default)
python interactive_webcam.py sign_language_cnn_final.keras

# ใช้ Transfer Learning model
python interactive_webcam.py sign_language_transfer_final.keras
```

### ขั้นตอนที่ 2: ให้ Feedback

เมื่อ webcam เปิดขึ้นมา:

1. **วางมือในกรอบสีเขียว** (ROI - Region of Interest)
2. **โชว์ท่าภาษามือ** ให้โมเดลทำนาย
3. **กดปุ่มเพื่อให้ feedback:**

| ปุ่ม | หน้าที่ |
|------|---------|
| `C` | **Correct** - บันทึกว่าโมเดลทำนายถูก (positive example) |
| `W` | **Wrong** - บอกว่าโมเดลทำนายผิด แล้วเลือก label ที่ถูก |
| `T` | **Train** - เทรนโมเดลด้วยข้อมูล feedback ที่เก็บไว้ |
| `S` | **Statistics** - แสดงสถิติข้อมูล feedback |
| `R` | **Reload** - โหลดโมเดลที่อัพเดทแล้ว |
| `Q` | **Quit** - ออกจากโปรแกรม |

### ขั้นตอนที่ 3: เทรนโมเดล

เมื่อเก็บ feedback ได้อย่างน้อย **10 samples**:

1. **กด `T`** เพื่อเริ่ม training
2. **ตอบคำถาม:**
   - "Load replay buffer?" → แนะนำ `y` (ป้องกัน catastrophic forgetting)
   - "Load updated model?" → `y` (ใช้โมเดลใหม่ทันที)
3. **รอให้เทรนเสร็จ** (ใช้เวลา 1-2 นาที)
4. **ทดสอบโมเดลใหม่** ผ่าน webcam

---

## 🔧 การใช้งาน Online Trainer แยกต่างหาก

หากต้องการเทรนแบบ manual หรือทดสอบ:

```python
from online_trainer import OnlineTrainer

# Initialize trainer
trainer = OnlineTrainer(
    model_path='sign_language_cnn_final.keras',
    feedback_dir='feedback_data',
    replay_buffer_size=100
)

# ดูสถิติข้อมูล feedback
stats, total = trainer.get_feedback_stats()
print(f"Total feedback: {total}")

# โหลด replay buffer (ถ้ามี original training data)
trainer.load_replay_buffer('sign_mnist_train.csv')

# เทรนโมเดล
trainer.incremental_train(
    epochs=2,
    learning_rate=0.0001,
    use_replay_buffer=True,
    batch_size=8
)
```

### บันทึก Feedback แบบ Manual

```python
import numpy as np
from PIL import Image

# Load image
img = Image.open('my_hand_sign.jpg').convert('L')
img_array = np.array(img.resize((28, 28))) / 255.0

# Save as feedback
trainer.save_feedback(
    image=img_array,
    predicted_label='A',
    correct_label='B',  # แก้ไข label ที่ถูก
    confidence=0.85
)
```

---

## 📊 โครงสร้างข้อมูล Feedback

### feedback_data/

```
feedback_data/
├── A/
│   ├── feedback_20250115_143025_123456.png
│   ├── feedback_20250115_143127_234567.png
│   └── ...
├── B/
│   ├── feedback_20250115_143315_345678.png
│   └── ...
├── ...
├── feedback_log.csv
└── training_log.json
```

### feedback_log.csv

บันทึกทุก feedback ที่ได้รับ:

| timestamp | image_path | predicted_label | correct_label | confidence |
|-----------|------------|-----------------|---------------|------------|
| 2025-01-15T14:30:25 | feedback_data/A/... | A | A | 0.95 |
| 2025-01-15T14:31:15 | feedback_data/B/... | A | B | 0.82 |

### training_log.json

บันทึกประวัติการเทรน:

```json
[
  {
    "timestamp": "2025-01-15T15:00:00",
    "feedback_samples": 25,
    "replay_samples": 100,
    "epochs": 2,
    "learning_rate": 0.0001,
    "final_loss": 0.123,
    "final_accuracy": 0.96,
    "model_path": "sign_language_cnn_updated.keras"
  }
]
```

---

## ⚙️ การตั้งค่า Training Parameters

### ค่า Default (แนะนำ)

```python
epochs = 2                    # น้อยเพื่อป้องกัน overfitting
learning_rate = 0.0001        # เล็กมากเพื่อ fine-tune อย่างระมัดระวัง
use_replay_buffer = True      # ป้องกัน catastrophic forgetting
replay_buffer_size = 100      # จำนวนตัวอย่างเก่าที่จะผสม
batch_size = 8                # เล็กเพราะข้อมูลใหม่มีน้อย
```

### การปรับแต่ง

**ถ้าโมเดลเทรนช้า:**
- เพิ่ม `batch_size` → 16 หรือ 32
- ลด `replay_buffer_size` → 50

**ถ้าโมเดลลืมข้อมูลเดิม (Catastrophic Forgetting):**
- เพิ่ม `replay_buffer_size` → 200 หรือ 500
- ลด `learning_rate` → 0.00005

**ถ้าโมเดลไม่เรียนรู้จากข้อมูลใหม่:**
- เพิ่ม `learning_rate` → 0.0002 หรือ 0.0005
- เพิ่ม `epochs` → 3 หรือ 4

---

## 📈 Best Practices

### 1. การเก็บ Feedback ที่มีคุณภาพ

✅ **DO:**
- เก็บข้อมูลหลายมุมมอง (หมุนมือ, เปลี่ยนแสง)
- เก็บทั้ง correct และ incorrect predictions
- เก็บข้อมูลกระจายทุก letter (อย่าเก็บแค่ตัวที่ผิดบ่อย)
- เก็บข้อมูลในสภาพแสงต่างๆ

❌ **DON'T:**
- อย่าเก็บภาพเบลอหรือมือไม่อยู่ในเฟรม
- อย่าป้อน label ผิด (ตรวจสอบให้ดีก่อนกด)
- อย่าเทรนบ่อยเกินไป (รอให้มีข้อมูลพอก่อน)

### 2. ความถี่ในการ Train

| สถานการณ์ | แนะนำ |
|-----------|-------|
| เริ่มใช้งานใหม่ | เก็บ 50-100 samples ก่อนเทรนครั้งแรก |
| ใช้งานประจำ | เทรนทุกๆ 20-30 samples ใหม่ |
| แก้ปัญหาเฉพาะตัวอักษร | เก็บ 10+ samples ของตัวนั้นแล้วเทรน |
| Fine-tune ความแม่นยำ | เก็บ 100+ samples คุณภาพสูงแล้วเทรนทีเดียว |

### 3. Data Distribution

ตรวจสอบให้มี feedback กระจายทั้ง 24 ตัวอักษร:

```bash
python interactive_webcam.py
# จากนั้นกด 'S' เพื่อดูสถิติ
```

**เป้าหมาย:** แต่ละตัวอักษรควรมีอย่างน้อย **5-10 samples**

### 4. Model Versioning

ระบบจะสร้างโมเดลใหม่เป็น `*_updated.keras`:

```
sign_language_cnn_final.keras         ← Original (ปลอดภัย)
sign_language_cnn_final_updated.keras ← After training (ใหม่)
```

คุณสามารถ:
- เก็บ backup โมเดลเดิมไว้
- เปรียบเทียบ accuracy ระหว่างโมเดล (ใช้ `benchmark_model.py`)
- กลับไปใช้โมเดลเดิมได้ตลอด

---

## 🐛 Troubleshooting

### ปัญหา: "No feedback data available for training!"

**สาเหตุ:** ยังไม่มีการเก็บ feedback

**แก้ไข:**
1. รัน `interactive_webcam.py`
2. กด `C` (correct) หรือ `W` (wrong) เพื่อเก็บข้อมูล
3. เก็บอย่างน้อย 10 samples ก่อนกด `T`

### ปัญหา: "Warning: sign_mnist_train.csv not found"

**สาเหตุ:** ไม่มี original training data สำหรับ replay buffer

**แก้ไข:**
- **Option 1:** ดาวน์โหลด dataset จาก Kaggle (ดูใน [sign_language_model.ipynb](sign_language_model.ipynb:1))
- **Option 2:** เทรนโดยไม่ใช้ replay buffer (เสี่ยงต่อ catastrophic forgetting)
- **Option 3:** ตอบ `n` เมื่อถามว่าจะ "Load replay buffer?"

### ปัญหา: โมเดลแย่ลงหลัง Train (Catastrophic Forgetting)

**สาเหตุ:** เทรนโดยไม่มี replay buffer หรือ learning rate สูงเกิน

**แก้ไข:**
1. ลบโมเดลที่แย่ลง:
   ```bash
   rm sign_language_cnn_updated.keras
   ```
2. กลับไปใช้โมเดลเดิม
3. เทรนใหม่ด้วย:
   - `use_replay_buffer=True`
   - `learning_rate=0.00005` (ครึ่งหนึ่ง)
   - `epochs=1` (น้อยลง)

### ปัญหา: Webcam ไม่เปิด

**แก้ไข:** ใช้ [camera_diagnostic.py](camera_diagnostic.py:1) ตรวจสอบกล้อง

```bash
python camera_diagnostic.py
```

### ปัญหา: Training ช้ามาก

**แก้ไข:**
1. ลด `replay_buffer_size` → 50
2. เพิ่ม `batch_size` → 16
3. ลด `epochs` → 1

---

## 🧪 ตัวอย่างการใช้งาน

### Scenario 1: โมเดลจำตัว "B" ไม่ได้

```bash
# 1. รัน webcam
python interactive_webcam.py

# 2. โชว์ท่า "B" และกด 'C' หลายๆ ครั้ง (10-15 ครั้ง)
#    - เปลี่ยนมุมมือ
#    - เปลี่ยนแสง
#    - เปลี่ยนระยะ

# 3. กด 'S' เพื่อดูสถิติ → ควรมี B อย่างน้อย 10 samples

# 4. กด 'T' เพื่อเทรน
#    ตอบ 'y' เมื่อถาม "Load replay buffer?"
#    ตอบ 'y' เมื่อถาม "Load updated model?"

# 5. ทดสอบท่า "B" อีกครั้ง → ควรแม่นขึ้น!
```

### Scenario 2: โมเดลสับสนระหว่าง "M" กับ "N"

```bash
# 1. เก็บข้อมูล "M" และ "N" อย่างละ 15-20 samples
#    - ใช้ 'W' เพื่อแก้ไข prediction ที่ผิด

# 2. กด 'S' ดูสถิติ:
#    M: 18 samples
#    N: 20 samples

# 3. กด 'T' เทรนด้วย replay buffer

# 4. ทดสอบทั้ง "M" และ "N" → ควรแยกแยะได้ชัดขึ้น
```

### Scenario 3: ปรับโมเดลให้เหมาะกับมือของคุณ

```bash
# 1. เก็บข้อมูลทุกตัวอักษร (A-Y) อย่างละ 5-10 samples
#    ใช้มือของคุณเอง, แสงในที่ทำงานของคุณ

# 2. เทรนด้วย replay buffer ขนาดใหญ่ (200+)

# 3. โมเดลจะปรับตัวเหมาะกับลักษณะมือและสภาพแวดล้อมของคุณ
#    แต่ยังคงความรู้จาก original dataset
```

---

## 📚 Technical Details

### Anti-Catastrophic Forgetting Strategy

ระบบใช้ **Experience Replay** technique:

1. **Replay Buffer**: เก็บตัวอย่างจาก original dataset (100-200 samples)
2. **Mixed Training**: ผสมข้อมูลเก่า + ข้อมูลใหม่ในอัตราส่วน ~80:20
3. **Small Learning Rate**: ใช้ 0.0001 (เล็กกว่าตอน train ครั้งแรก 10 เท่า)
4. **Few Epochs**: เทรนแค่ 1-2 epochs

### Training Process Flow

```
                Start Training
                      |
                      v
        Load Feedback Data (X_new, y_new)
                      |
                      v
          [use_replay_buffer? ]
           /                  \
         Yes                   No
          |                     |
          v                     v
   Load X_old, y_old       Use only new data
          |                     |
          v                     |
   Concatenate datasets         |
          |                     |
          +---------------------+
                      |
                      v
         Compile with small LR (0.0001)
                      |
                      v
              Train 1-2 epochs
                      |
                      v
           Save *_updated.keras
                      |
                      v
        Log training results (JSON)
                      |
                      v
                    Done!
```

### Input Preprocessing

จาก webcam frame → model input:

```
BGR Frame (640x480)
      ↓
Grayscale (ROI: 300x300)
      ↓
GaussianBlur (5x5)
      ↓
Resize to (28x28) or (96x96)
      ↓
Normalize to [0, 1]
      ↓
Reshape to (1, H, W, C)
      ↓
Model Input
```

---

## 🔒 Data Privacy & Safety

- ✅ **ข้อมูลเก็บ Local เท่านั้น** - ไม่ส่งออกไปไหน
- ✅ **ลบได้ตลอดเวลา** - แค่ลบโฟลเดอร์ `feedback_data/`
- ✅ **โมเดลเดิมปลอดภัย** - สร้างไฟล์ใหม่แยกต่างหาก (`*_updated.keras`)
- ✅ **Rollback ได้** - สามารถกลับไปใช้โมเดลเดิมได้เสมอ

---

## 📞 Support & Troubleshooting

### ต้องการความช่วยเหลือ?

1. **ดู documentation:**
   - [README.md](README.md) - โครงการหลัก
   - [README_TESTING.md](README_TESTING.md) - การทดสอบโมเดล
   - [QUICK_START.md](QUICK_START.md) - เริ่มต้นใช้งาน

2. **ตรวจสอบระบบ:**
   ```bash
   python camera_diagnostic.py  # ตรวจสอบกล้อง
   python benchmark_model.py    # วัดประสิทธิภาพโมเดล
   ```

3. **ดู logs:**
   - `feedback_data/feedback_log.csv` - ประวัติ feedback
   - `feedback_data/training_log.json` - ประวัติการเทรน

---

## 🎓 แนวคิดเพิ่มเติม

### ต่อยอดได้:

1. **Active Learning**: ให้โมเดลขอ feedback เฉพาะตัวที่มั่นใจน้อย
2. **Confidence Threshold**: บันทึก feedback อัตโนมัติเมื่อ confidence ต่ำ
3. **Multi-user Learning**: รวม feedback จากหลายคน
4. **Transfer to New Signs**: ขยายไปภาษามืออื่นๆ
5. **Ensemble Models**: ผสมหลายโมเดลที่เทรนจาก feedback ต่างกัน

---

## ✅ Checklist การเริ่มใช้งาน

- [ ] ติดตั้ง dependencies (`pip install -r requirements.txt`)
- [ ] มีโมเดลที่เทรนแล้ว (`sign_language_cnn_final.keras`)
- [ ] ทดสอบกล้องด้วย `camera_diagnostic.py`
- [ ] รัน `interactive_webcam.py` ครั้งแรก
- [ ] เก็บ feedback 10+ samples
- [ ] เทรนโมเดลครั้งแรก (กด `T`)
- [ ] เปรียบเทียบ accuracy ก่อน-หลัง (ใช้ `benchmark_model.py`)
- [ ] ปรับแต่ง parameters ตามความต้องการ

---

## 🎉 สรุป

ระบบ Online Learning นี้ช่วยให้โมเดลของคุณ:

- 🚀 **เรียนรู้ต่อเนื่อง** จาก real-world usage
- 🎯 **ปรับตัว** ให้เหมาะกับผู้ใช้แต่ละคน
- 🔒 **ปลอดภัย** ไม่ลืมข้อมูลเดิม (anti-forgetting)
- 📊 **Trackable** ติดตามประวัติและสถิติได้ชัดเจน
- 🛠️ **Easy to use** ใช้งานง่ายผ่าน keyboard shortcuts

**เริ่มต้นใช้งานได้ทันที!**

```bash
python interactive_webcam.py
```

Happy Learning! 🤖✨
