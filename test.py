from ultralytics import YOLO
import cv2

# 1. Tải model bạn vừa huấn luyện xong
# Đường dẫn này trỏ tới file 'best.pt' được lưu tự động lúc nãy.
# (Nếu bạn đặt tên thư mục khác ở Bước trước, hãy sửa lại đoạn 'my_alpr_model_416' cho đúng nhé)
model = YOLO('runs/detect/my_alpr_model2/weights/best.pt')

# 2. Đường dẫn tới bức ảnh bạn muốn test
# Lời khuyên: Lên mạng tải 1 ảnh ngã tư đường phố Việt Nam, hoặc lấy 1 ảnh trong thư mục 'test/images' của bạn.
image_path = 'test_image.jpg' # Sửa lại tên file ảnh này cho đúng

# 3. Yêu cầu AI nhìn vào ảnh và dự đoán
print("Đang xử lý...")
results = model(image_path)

# 4. Vẽ khung nhận diện lên ảnh và hiển thị ra màn hình
# results[0] là kết quả của bức ảnh đầu tiên
annotated_frame = results[0].plot()

# Hiển thị ảnh
cv2.imshow("Ket qua nhan dien - AI Cua Toi", annotated_frame)
cv2.waitKey(0) # Cửa sổ sẽ giữ nguyên cho đến khi bạn bấm phím bất kỳ trên bàn phím
cv2.destroyAllWindows()