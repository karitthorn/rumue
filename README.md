# Sign Language Recognition AI 🤟

โปรเจกต์ AI สำหรับจำแนกภาษามือ (Sign Language) โดยใช้ Deep Learning และ Sign Language MNIST Dataset

## คำอธิบายโปรเจกต์

โปรเจกต์นี้สร้าง AI Model สำหรับจำแนกภาษามือแบบอัลฟาเบต (A-Z ยกเว้น J และ Z) โดยใช้ภาพจากมือขนาด 28x28 pixels ทำการเทรน 2 แบบ:

1. **CNN Model (Custom)**: สร้าง Convolutional Neural Network แบบกำหนดเองตั้งแต่ต้น
2. **Transfer Learning Model**: ใช้ MobileNetV2 pre-trained model พร้อม fine-tuning

## Dataset

- **ชื่อ**: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **แหล่งที่มา**: Kaggle
- **จำนวนคลาส**: 24 (A-Y, ยกเว้น J ที่ต้องใช้การเคลื่อนไหว)
- **ขนาดภาพ**: 28x28 pixels (grayscale)
- **Training set**: ~27,000 ภาพ
- **Test set**: ~7,000 ภาพ

## โครงสร้างโปรเจกต์

```
rumue/
│
├── sign_language_model.ipynb    # Jupyter Notebook หลักสำหรับ training
├── requirements.txt             # Python dependencies
├── README.md                    # เอกสารนี้
├── .gitignore                   # ไฟล์ที่ไม่ต้อง commit
│
├── best_cnn_model.keras         # Best CNN model (จะสร้างหลัง training)
├── best_transfer_model.keras    # Best Transfer Learning model
├── sign_language_cnn_final.keras        # Final CNN model
└── sign_language_transfer_final.keras   # Final Transfer Learning model
```

## การติดตั้ง

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/rumue.git
cd rumue
```

### 2. สร้าง Virtual Environment (แนะนำ)

```bash
python -m venv venv
source venv/bin/activate  # สำหรับ macOS/Linux
# หรือ
venv\Scripts\activate     # สำหรับ Windows
```

### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 4. ตั้งค่า Kaggle API (สำหรับดาวน์โหลด Dataset)

1. สร้างบัญชี Kaggle ที่ https://www.kaggle.com
2. ไปที่ Account Settings → API → Create New API Token
3. ย้ายไฟล์ `kaggle.json` ไปยัง:
   - macOS/Linux: `~/.kaggle/`
   - Windows: `C:\Users\<Username>\.kaggle\`

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## วิธีใช้งาน

### 1. เปิด Jupyter Notebook

```bash
jupyter notebook sign_language_model.ipynb
```

### 2. รัน Cells ตามลำดับ

Notebook ประกอบด้วย:

1. **Import Libraries และ Download Dataset**: โหลด dependencies และดาวน์โหลดข้อมูล
2. **Exploratory Data Analysis (EDA)**: สำรวจและแสดงผลข้อมูล
3. **Data Preprocessing**: เตรียมข้อมูล, normalization, data augmentation
4. **CNN Model**: สร้างและเทรน Custom CNN
5. **Transfer Learning Model**: สร้างและเทรน MobileNetV2 model
6. **Model Evaluation**: ประเมินผลและเปรียบเทียบ models
7. **Testing**: ทดสอบการทำนายกับภาพจริง

### 3. ใช้งาน Trained Model

```python
import numpy as np
from tensorflow import keras

# โหลด model
model = keras.models.load_model('sign_language_cnn_final.keras')

# เตรียมภาพ (28x28x1, normalized to 0-1)
image = your_image / 255.0
image = image.reshape(1, 28, 28, 1)

# ทำนาย
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# Label mapping
label_map = {i: chr(65 + i) if i < 9 else chr(65 + i + 1) for i in range(24)}
print(f"Predicted Sign: {label_map[predicted_class]}")
```

## Model Architecture

### CNN Model

```
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout
    ↓
Flatten → Dense(256) → BatchNorm → Dropout → Dense(128) → Dropout
    ↓
Dense(24, softmax)
```

### Transfer Learning Model

```
MobileNetV2 (frozen) → GlobalAveragePooling2D
    ↓
Dense(256) → Dropout → Dense(128) → Dropout
    ↓
Dense(24, softmax)
```

## ผลลัพธ์ที่คาดหวัง

- **CNN Model Accuracy**: ~95-98%
- **Transfer Learning Accuracy**: ~96-99%

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Keras 2.13+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV
- Kagglehub

ดูรายละเอียดเพิ่มเติมใน `requirements.txt`

## การพัฒนาต่อ

1. ลองใช้ model architectures อื่นๆ (EfficientNet, ResNet, Vision Transformer)
2. เพิ่ม data augmentation techniques เพิ่มเติม
3. Ensemble multiple models
4. สร้าง web application ด้วย Flask/FastAPI
5. พัฒนา mobile app ด้วย TensorFlow Lite
6. Real-time sign language detection ด้วย webcam
7. รองรับภาษามือแบบประโยค (continuous sign language)

## Troubleshooting

### GPU ไม่ทำงาน

ตรวจสอบว่าติดตั้ง TensorFlow-GPU และ CUDA อย่างถูกต้อง:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Out of Memory

ลด `batch_size` ใน training cells:

```python
batch_size = 64  # ลดจาก 128
```

### Dataset Download Error

ตรวจสอบว่าตั้งค่า Kaggle API credentials อย่างถูกต้อง

## License

MIT License - ดูไฟล์ [LICENSE](LICENSE) สำหรับรายละเอียด

## Contributing

Pull requests ยินดีต้อนรับเสมอ! สำหรับการเปลี่ยนแปลงใหญ่ๆ โปรดเปิด issue ก่อนเพื่อหารือว่าคุณต้องการเปลี่ยนแปลงอะไร


## อ้างอิง

- [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
