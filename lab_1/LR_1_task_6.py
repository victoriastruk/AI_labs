import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створення моделі Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear')

# Тренування класифікатора
svm_classifier.fit(X, y)

# Прогнозування значень для тренувальних даних
y_pred_svm = svm_classifier.predict(X)

# Обчислення якості класифікатора
accuracy_svm = 100.0 * (y == y_pred_svm).sum() / X.shape[0]
print("Accuracy of SVM classifier =", round(accuracy_svm, 2), "%")

# Візуалізація результатів роботи класифікатора
visualize_classifier(svm_classifier, X, y)

# Розбивка даних на навчальний та тестовий набори
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.2, random_state=3)
svm_classifier_new = SVC(kernel='linear')
svm_classifier_new.fit(X_train_svm, y_train_svm)
y_test_pred_svm = svm_classifier_new.predict(X_test_svm)

# Обчислення якості класифікатора
accuracy_svm_new = 100.0 * (y_test_svm == y_test_pred_svm).sum() / X_test_svm.shape[0]
print("Accuracy of the new SVM classifier =", round(accuracy_svm_new, 2), "%")

# Візуалізація роботи класифікатора
visualize_classifier(svm_classifier_new, X_test_svm, y_test_svm)

num_folds = 3
accuracy_values_svm = cross_val_score(svm_classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy (SVM): " + str(round(100 * accuracy_values_svm.mean(), 2)) + "%")
precision_values_svm = cross_val_score(svm_classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision (SVM): " + str(round(100 * precision_values_svm.mean(), 2)) + "%")
recall_values_svm = cross_val_score(svm_classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall (SVM): " + str(round(100 * recall_values_svm.mean(), 2)) + "%")
f1_values_svm = cross_val_score(svm_classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 (SVM): " + str(round(100 * f1_values_svm.mean(), 2)) + "%")
