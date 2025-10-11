"""
Camera Diagnostic Tool
ตรวจสอบกล้องที่ใช้งานได้ในระบบ
"""

import cv2
import platform

print("="*60)
print("🔍 CAMERA DIAGNOSTIC TOOL")
print("="*60)
print(f"Platform: {platform.system()}")
print(f"OpenCV Version: {cv2.__version__}")
print("="*60 + "\n")

# ทดสอบกล้องหลายตัว
max_cameras = 5
working_cameras = []

print("กำลังค้นหากล้องที่ใช้งานได้...\n")

for i in range(max_cameras):
    print(f"Testing camera index {i}...", end=" ")
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✅ SUCCESS - Resolution: {width}x{height}")
            working_cameras.append(i)
        else:
            print("⚠️  Opened but cannot read frame")
        cap.release()
    else:
        print("❌ Failed to open")

print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)

if working_cameras:
    print(f"\n✅ Found {len(working_cameras)} working camera(s):")
    for idx in working_cameras:
        print(f"   - Camera index: {idx}")

    print(f"\n💡 To use with webcam_test.py:")
    print(f"   python webcam_test.py --camera {working_cameras[0]}")

    # ทดสอบแสดงภาพจากกล้องแรก
    if len(working_cameras) > 0:
        print(f"\n🎥 Opening camera {working_cameras[0]} for 3 seconds...")
        print("   Press any key to close the window\n")

        cap = cv2.VideoCapture(working_cameras[0])
        if cap.isOpened():
            for _ in range(30):  # แสดง ~1 วินาที (30 frames)
                ret, frame = cap.read()
                if ret:
                    # เพิ่มข้อความ
                    cv2.putText(frame, f"Camera {working_cameras[0]} - Press any key to close",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Camera Test', frame)
                    if cv2.waitKey(100) != -1:  # รอ 100ms หรือจนกว่าจะกดปุ่ม
                        break
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera test completed!")
else:
    print("\n❌ No working cameras found!")
    print("\n🔧 Troubleshooting tips:")
    print("   1. Check camera permissions in System Settings")
    print("   2. Make sure no other app is using the camera")
    print("   3. Try reconnecting external camera (if using)")
    print("   4. On macOS: System Settings > Privacy & Security > Camera")
    print("   5. Grant permission to Terminal or your Python IDE")

print("\n" + "="*60)
