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

# Розфарбовування об'єктів на зображенні на основі моментів
colors = {}
for i in range(1, ret + 1):
    mask = (markers == i).astype(np.uint8)
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        color = img[centroid_y, centroid_x]
        # Збільшення інтенсивності кольорів
        color = np.clip(color * 1.5, 0, 255).astype(np.uint8)
        colors[i] = color

# Розфарбовування об'єктів на зображенні
for i in range(1, ret + 1):
    img[markers == i] = colors[i]

# Відображення результату
cv2.imshow("Coins Marked", img)
cv2.waitKey(0)
