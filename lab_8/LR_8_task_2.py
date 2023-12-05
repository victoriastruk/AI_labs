# Імпорт необхідних бібліотек
import cv2
import numpy as np

# Зчитання зображення з файлу
img = cv2.imread("struk.png")

# Визначення ядра для морфологічних операцій (розширення та ерозія)
kernel = np.ones((5,5), np.uint8)

# Конвертація зображення у відтінки сірого
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Застосування гаусівського розмиття до зображення у відтінках сірого
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)

# Застосування детектора країв Canny до оригінального зображення
imgCanny = cv2.Canny(img, 150, 200)

# Розширення контурів на зображенні для збільшення їхньої товщини
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)

# Ерозія розширених контурів для зменшення товщини
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

# Відображення зображень за допомогою OpenCV
cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blur Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Dialation Image",imgDialation)
cv2.imshow("Eroded Image",imgEroded)

# Очікування натискання клавіші перед закриттям вікон зображень
cv2.waitKey(0)
