from ultralytics import YOLO

# 1. Tải model pre-trained (Khuyến nghị dùng bản 'nano' cho nhẹ và nhanh để test thử)
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    # 2. Bắt đầu quá trình huấn luyện
    results = model.train(
        data='data.yaml', # ĐƯỜNG DẪN TỚI FILE YAML CỦA BẠN
        epochs=200,       # Số vòng học (thử 50 trước, nếu kết quả chưa tốt thì tăng lên 100-200)
        imgsz=416,       # Kích thước ảnh đầu vào chuẩn của YOLO
        batch=16,        # Số ảnh đưa vào GPU mỗi lần (Nếu máy báo lỗi hết bộ nhớ/OOM, hãy giảm xuống 8 hoặc 4)
        name='my_alpr_model', # Tên thư mục lưu kết quả
        device='cpu'         # Sử dụng GPU (Nếu máy bạn không có GPU NVIDIA, hãy đổi thành device='cpu')
    )