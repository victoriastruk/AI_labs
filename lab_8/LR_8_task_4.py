# Імпорт бібліотеки OpenCV
import cv2

# Завантаження класифікатора для виявлення обличчя
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Зчитування зображення з файлу
img = cv2.imread('messi_face.JPG')

# Конвертація зображення у відтінки сірого
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Застосування класифікатора для виявлення обличчя на зображенні
faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)

# Прохід по кожному виявленому обличчю та намалювання прямокутника навколо нього
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Відображення результату з намальованими прямокутниками навколо обличчя
cv2.imshow("Result", img)

# Очікування натискання клавіші перед закриттям вікна зображення
cv2.waitKey(0)
