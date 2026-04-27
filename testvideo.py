from ultralytics import YOLO
import cv2

# 1. Tải model mà bạn đã train (sửa lại đường dẫn nếu cần)
model = YOLO('runs/detect/my_alpr_model2/weights/best.pt')

# 2. Đường dẫn tới file video của bạn
video_path = 'test_video.mp4' # Đổi tên này thành tên video bạn đã tải về
cap = cv2.VideoCapture(video_path)

print("Đang mở video... Bấm phím 'q' trên bàn phím để thoát.")

while cap.isOpened():
    # Đọc từng khung hình (frame) của video
    success, frame = cap.read()
    
    if not success:
        print("Đã phát hết video!")
        break

    # 3. Đưa khung hình vào cho AI dự đoán
    # Tham số verbose=False để Terminal đỡ bị trôi dòng liên tục
    results = model(frame, verbose=False)

    # 4. Vẽ khung nhận diện lên khung hình
    annotated_frame = results[0].plot()

    # 5. Hiển thị lên màn hình
    # Nếu video to quá tràn màn hình, bạn có thể bỏ comment dòng dưới đây để thu nhỏ lại
    annotated_frame = cv2.resize(annotated_frame, (720, 1080)) 
    
    cv2.imshow("Nhan dien Video - AI Cua Toi", annotated_frame)

    # Lệnh này giúp video chạy mượt và chờ bạn bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp bộ nhớ sau khi tắt
cap.release()
cv2.destroyAllWindows()