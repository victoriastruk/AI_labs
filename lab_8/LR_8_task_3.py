import cv2

img = cv2.imread("struk.png")

# Завантаження передньо навченого класифікатора виявлення обличчя
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Перетворення зображення в градації сірого для виявлення обличчя
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Виконання виявлення обличчя
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Перевірка, чи виявлено обличчя
if len(faces) > 0:
    # Припускаючи, що хочемо вирізати перше виявлене обличчя
    x, y, w, h = faces[0]
    imgCropped = img[y:y + h, x:x + w]

    # Показ вирізаного обличчя
    cv2.imshow("Image Cropped", imgCropped)
    cv2.waitKey(0)
else:
    print("Обличчя не виявлено.")
