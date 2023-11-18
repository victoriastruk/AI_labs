import numpy as np
import cv2

img = cv2.imread('coins_2.JPG')

# Відображення вихідного зображення
cv2.imshow("Coins Original", img)
cv2.waitKey(0)

# Перетворення в відтінки сірого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Бінаризація за допомогою OTSU
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Видалення шуму
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Фонова область
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Пошук впевненої області переднього плану
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Пошук невідомого регіону
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Маркування міток
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
# область невідомого
markers[unknown == 255] = 0

# Використання функції watershed для визначення маркерних ліній
markers = cv2.watershed(img, markers)

# Додаткова частина коду для ідентифікації монет за розміром
for label in np.unique(markers):
    if label == 0:
        continue  # Пропустити фоновий маркер
    mask = np.zeros_like(markers, dtype=np.uint8)
    mask[markers == label] = 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Обчислення прямокутника, обведеного контуром
        x, y, w, h = cv2.boundingRect(contours[0])
        # Вибір кольору на основі розміру
        color = [w % 255, h % 255, (w * h) % 255]
        cv2.drawContours(img, contours, -1, color, -1)

# Відображення результату
cv2.imshow("Coins Marked", img)
cv2.waitKey(0)
