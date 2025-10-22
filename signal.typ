#import "@preview/charged-ieee:0.1.4": ieee
#set page(paper: "a4")
#show page: set text(font: "TH sarabun new")
#show: ieee.with(
  title: text(font: "TH sarabun new")[การพัฒนาระบบจำแนกภาษามือด้วยเทคนิคการเรียนรู้เชิงลึก],
  authors: (
    (
      name: "คริษฐ์ธร บำรุงพิพัฒนพร",
      department: text(font: "TH sarabun new")[6610501998],
      organization: text(font: "TH sarabun new")[วิศวกรรมคอมพิวเตอร์],
      location: text(font: "TH sarabun new")[มหาวิทยาลัยเกษตรศาสตร์ บางเขน],
      email: "karitthorn.b@ku.th",
    ),
        (
      name: "คริษฐ์ธร บำรุงพิพัฒนพร",
      department: text(font: "TH sarabun new")[6610501998],
      organization: text(font: "TH sarabun new")[วิศวกรรมคอมพิวเตอร์],
      location: text(font: "TH sarabun new")[มหาวิทยาลัยเกษตรศาสตร์ บางเขน],
      email: "karitthorn.b@ku.th",
    ),
  ),
  figure-supplement: [Fig.],
)

// Apply font settings after IEEE template to override its defaults
#set text(font: "TH sarabun new", size: 11pt)
#set par(justify: true, leading: 0.65em)
#show figure.caption: set align(center)

#set align(center)
= บทนำ

#set align(left)
โครงงานนี้มีวัตถุประสงค์เพื่อพัฒนาระบบจำแนกภาษามือ (Sign Language Recognition) โดยใช้เทคนิคการเรียนรู้เชิงลึก (Deep Learning) เพื่อแปลงท่าภาษามือเป็นตัวอักษร A-Z โดยใช้ชุดข้อมูล Sign Language MNIST จาก Kaggle ซึ่งประกอบด้วยภาพมือที่แสดงท่าทางต่างๆ ของตัวอักษร 24 ตัว (ยกเว้น J และ Z ที่ต้องใช้ท่าทางเคลื่อนไหว)

ระบบที่พัฒนาขึ้นนี้สามารถนำไปประยุกต์ใช้ในการสื่อสารกับผู้บกพร่องทางการได้ยินและการพูด รวมถึงเป็นเครื่องมือช่วยในการเรียนรู้ภาษามือสำหรับบุคคลทั่วไป

= ชุดข้อมูลและการเตรียมข้อมูล

== ชุดข้อมูล Sign Language MNIST

ชุดข้อมูลที่ใช้ในโครงงานนี้คือ Sign Language MNIST จาก Kaggle โดยมีรายละเอียดดังนี้:

- *ขนาดข้อมูล:* ข้อมูลฝึกสอน 27,455 ภาพ และข้อมูลทดสอบ 7,172 ภาพ
- *จำนวนคลาส:* 24 คลาส (A-Y ยกเว้น J ที่ต้องใช้การเคลื่อนไหว)
- *ขนาดภาพ:* 28×28 พิกเซล (grayscale)
- *รูปแบบข้อมูล:* ไฟล์ CSV ที่มี 785 คอลัมน์ (1 label + 784 pixels)

== การเตรียมข้อมูล

กระบวนการเตรียมข้อมูลประกอบด้วยขั้นตอนดังนี้:

+ *Reshape:* แปลงข้อมูลจาก flat array เป็นรูปภาพขนาด 28×28×1
+ *Normalization:* ปรับค่าพิกเซลให้อยู่ในช่วง [0, 1] โดยหารด้วย 255.0
+ *Train/Validation Split:* แบ่งข้อมูลแบบ stratified เป็น 80:20 ได้ข้อมูล 21,964 ภาพสำหรับการฝึกสอนและ 5,491 ภาพสำหรับการตรวจสอบ
+ *One-Hot Encoding:* แปลง labels เป็น categorical format สำหรับ 25 คลาส
+ *การแปลงสำหรับ Transfer Learning:* แปลง grayscale เป็น RGB และปรับขนาดเป็น 96×96 พิกเซล

== Data Augmentation

เพื่อเพิ่มความหลากหลายของข้อมูลและลดปัญหา overfitting ได้ใช้เทคนิค Data Augmentation ดังนี้:

*สำหรับ CNN Model:*
- การหมุนภาพ: ±10 องศา
- การเลื่อนแนวนอน/แนวตั้ง: 0.1
- การซูม: 0.1
- การเอียง (shear): 0.1

*สำหรับ Transfer Learning Model:*
- การหมุนภาพ: ±15 องศา
- การเลื่อนแนวนอน/แนวตั้ง: 0.15
- การซูม: 0.15
- การเอียง (shear): 0.15
- ไม่ใช้การพลิกแนวนอน เนื่องจากภาษามือมีทิศทางที่สำคัญ

= สถาปัตยกรรมของโมเดล

โครงงานนี้ได้พัฒนาโมเดลสองแบบเพื่อเปรียบเทียบประสิทธิภาพ

== โมเดล CNN แบบกำหนดเอง

โมเดล Convolutional Neural Network ที่ออกแบบเองประกอบด้วย:

*Convolutional Blocks (3 blocks):*
- Block 1: Conv2D(32, 3×3) + BatchNorm + Conv2D(32, 3×3) + BatchNorm + MaxPooling(2×2) + Dropout(0.25)
- Block 2: Conv2D(64, 3×3) + BatchNorm + Conv2D(64, 3×3) + BatchNorm + MaxPooling(2×2) + Dropout(0.25)
- Block 3: Conv2D(128, 3×3) + BatchNorm + Conv2D(128, 3×3) + BatchNorm + MaxPooling(2×2) + Dropout(0.25)

*Dense Layers:*
- Flatten() → Dense(256) + BatchNorm + Dropout(0.5)
- Dense(128) + BatchNorm + Dropout(0.5)
- Dense(25, softmax)

== โมเดล Transfer Learning (MobileNetV2)

โมเดลที่ใช้ Transfer Learning ด้วย MobileNetV2 ที่ฝึกมาจาก ImageNet:

*Base Model:* MobileNetV2 (input: 96×96×3, pretrained weights: ImageNet)

*Custom Layers:*
- GlobalAveragePooling2D()
- BatchNormalization()
- Dense(256, relu) + Dropout(0.5)
- BatchNormalization()
- Dense(128, relu) + Dropout(0.3)
- Dense(25, softmax)

การฝึกสอนแบ่งเป็น 2 phases:
- Phase 1: Freeze base model ทั้งหมด
- Phase 2: Fine-tuning โดย unfreeze ชั้นบนสุด (ชั้น 100 แรกยัง frozen)

= การฝึกสอนโมเดล

== พารามิเตอร์การฝึกสอน

*สำหรับ CNN Model:*
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Categorical Crossentropy
- Batch Size: 128
- Epochs: 5

*สำหรับ Transfer Learning Model:*
- Phase 1: Adam (lr = 0.001), batch size = 64, epochs = 5
- Phase 2: Adam (lr = 1×10⁻⁵), batch size = 64, epochs = 5

== Callbacks

ใช้ Callbacks เพื่อปรับปรุงการฝึกสอน:
- *ModelCheckpoint:* บันทึก best model ตาม validation accuracy
- *EarlyStopping:* หยุดการฝึกสอนเมื่อ validation loss ไม่ดีขึ้นใน 10 epochs
- *ReduceLROnPlateau:* ลด learning rate อัตโนมัติเมื่อ loss ไม่ดีขึ้น (factor = 0.5, patience = 5)

= ผลการทดลองและการประเมินผล

== ผลลัพธ์ของโมเดล

#table(
  columns: (auto, auto, auto),
  align: center,
  [*โมเดล*], [*Test Accuracy*], [*Test Loss*],
  [Custom CNN], [99.00%], [0.0360],
  [Transfer Learning], [93.54%], [0.1654],
)

จากผลการทดลอง พบว่าโมเดล CNN ที่ออกแบบเองให้ผลลัพธ์ที่ดีกว่าโมเดล Transfer Learning อย่างมีนัยสำคัญ โดยมีความแม่นยำ 99.00% บนชุดข้อมูลทดสอบ

== การประเมินผลละเอียด

การประเมินผลของโมเดล CNN แสดงให้เห็นว่า:
- *Precision, Recall, F1-Score:* มีค่าเกือบ 1.00 ในทุกคลาส
- *Confusion Matrix:* แสดงการทำนายที่แม่นยำสูงในทุกคลาส
- *Training History:* แสดงการลู่เข้าที่ดีโดยไม่มีปัญหา overfitting

== เครื่องมือการประเมินผล

ใช้เครื่องมือหลายอย่างในการประเมินผล:
+ *Classification Report:* แสดงค่า precision, recall, f1-score สำหรับแต่ละคลาส
+ *Confusion Matrix:* Visualize การทำนายในรูปแบบ heatmap
+ *Training History Plots:* กราฟแสดง accuracy และ loss ของ training/validation
+ *Random Sample Testing:* ทดสอบด้วยภาพสุ่มพร้อมแสดง Top-3 predictions

= สรุปและอภิปรายผล

โครงงานนี้แสดงให้เห็นว่าการพัฒนาโมเดล CNN ที่ออกแบบเองสามารถให้ผลลัพธ์ที่ดีเยี่ยมในงานจำแนกภาษามือ โดยมีความแม่นยำถึง 99.00% ซึ่งสูงกว่าการใช้ Transfer Learning จาก MobileNetV2 (93.54%)

*จุดเด่นของโครงงาน:*
+ การออกแบบสถาปัตยกรรม CNN ที่เหมาะสมกับข้อมูล
+ การใช้ Batch Normalization และ Dropout อย่างเหมาะสม
+ Data Augmentation ที่คำนึงถึงลักษณะของภาษามือ (ไม่พลิกแนวนอน)
+ การใช้ Stratified Split เพื่อรักษาสัดส่วนคลาส
+ Callbacks ที่ครบถ้วนสำหรับ optimization

*การพัฒนาต่อยอด:*
+ เพิ่มคลาส J และ Z โดยใช้ video sequences
+ พัฒนา real-time recognition ด้วย webcam
+ ขยายไปยังภาษามือภาษาอื่นๆ
+ สร้าง mobile application สำหรับใช้งานจริง

โมเดลที่พัฒนาขึ้นมีศักยภาพในการนำไปประยุกต์ใช้งานจริง เช่น ระบบแปลภาษามือแบบ real-time หรือแอปพลิเคชันช่วยการเรียนรู้ภาษามือ ซึ่งจะเป็นประโยชน์ต่อการสื่อสารกับผู้บกพร่องทางการได้ยินและการพูด
