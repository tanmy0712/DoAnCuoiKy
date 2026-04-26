from ultralytics import YOLO
import cv2

# load model đã train
model = YOLO("runs/detect/train/weights/best.pt")

# đọc ảnh test
img_path = "test.jpg"   
img = cv2.imread(img_path)

# predict
results = model(img)

# duyệt kết quả
for r in results:
    boxes = r.boxes.xyxy 

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # vẽ khung
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ghi chữ
        cv2.putText(img, "plate", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# hiển thị ảnh
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# lưu ảnh kết quả
cv2.imwrite("result.jpg", img)