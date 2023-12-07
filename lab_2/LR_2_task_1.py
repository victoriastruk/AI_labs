import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            y.append(0)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            y.append(1)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i].astype(int)
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1]
y = X_encoded[:, -1]

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення SVМ-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual=False))

# Навчання класифікатора
classifier.fit(X_train, y_train)

# Оцінка моделі за допомогою крос-валідації
num_folds = 3
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
precision = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
recall = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)

# Виведення метрик
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
print("Accuracy score: " + str(round(100 * accuracy.mean(), 2)) + "%")
print("Precision score: " + str(round(100 * precision.mean(), 2)) + "%")
print("Recall score: " + str(round(100 * recall.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
              'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Використання класифікатора для тестової точки даних та виведення результату
predicted_class = classifier.predict(input_data_encoded)
predicted_class_label = label_encoder[-1].inverse_transform([predicted_class])[0]
print(predicted_class_label)
