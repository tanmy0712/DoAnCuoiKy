from ultralytics import YOLO

# Khởi tạo model bản Small để cân bằng giữa tốc độ và độ chính xác
model = YOLO('yolo11s.pt') 

# Train với kích thước ảnh 640
model.train(
    data='path/to/your/data.yaml',
    epochs=100,
    imgsz=640,  # Tăng từ 416 lên 640
    batch=16,   # Điều chỉnh tùy theo VRAM của bạn
    mosaic=1.0, # Giúp nhận diện vật thể nhỏ tốt hơn
    device=0    # Chạy trên GPU
)