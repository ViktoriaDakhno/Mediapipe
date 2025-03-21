import cv2 as cv  # Імпортуємо OpenCV
from ultralytics import YOLO  # Імпортуємо YOLOv8
import logging

# Завантажуємо модель YOLOv8s
yolo = YOLO('yolov8s.pt', verbose=False)  # verbose=False вимикає зайві логи
logging.getLogger("ultralytics").setLevel(logging.NOTSET)

# Відкриваємо відеопотік з вебкамери
video = cv.VideoCapture(0)
img = cv.imread("plain.jfif")

class_names = yolo.names


results = yolo.track(img, stream=True)  # Запускаємо YOLO

for result in results:
    for box in result.boxes:
        if box.conf[0] > 0.4:  # Фільтруємо за впевненістю
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координати рамки
            cls = int(box.cls[0])  # Номер класу
            class_name = class_names[cls]  # Назва класу

            # Малюємо червону рамку і текст
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(img, f'{class_name} {box.conf[0]*100:.2f}', (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv.imshow('frame', img)  # Відображаємо кадр
cv.imwrite('newFile.jpg', img)

cv.waitKey(0)
cv.destroyAllWindows()