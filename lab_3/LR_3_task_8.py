# Імпорт необхідних бібліотек
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
import numpy as np

# Завантаження даних про іриси
iris = load_iris()
X = iris['data']
y = iris['target']

# Ініціалізація моделі KMeans з параметрами
# n_clusters=5 - кількість кластерів, яку алгоритм буде намагатися знайти
# n_init=10 - кількість різних початкових наборів центроїдів, щоб обрати оптимальний
kmeans = KMeans(n_clusters=5, n_init=10)
# Навчання моделі KMeans на вхідних даних X
kmeans.fit(X)
# Передбачення кластерів для кожної точки вхідних даних за допомогою навченої моделі KMeans
y_kmeans = kmeans.predict(X)

# Відображення точок та центроїдів на графіку
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Визначення функції для знаходження кластерів
def find_clusters(X, n_clusters, rseed=2):
    # Створення псевдовипадкового генератора з фіксованим зерном для відтворюваності результатів
    rng = np.random.RandomState(rseed)
    # Створення випадкової перестановки індексів від 0 до кількості рядків у вихідних даних
    i = rng.permutation(X.shape[0])[:n_clusters]
    # Вибір випадкових рядків з вихідних даних для створення початкових центроїдів кластерів
    centers = X[i]
    while True:
        # Визначення кластерів для кожної точки на основі найближчого центроїда
        labels = pairwise_distances_argmin(X, centers)
        # Обчислення нових центроїдів, взявши середнє значення точок у кожному кластері
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # Перевірка, чи нові центроїди не змінились
        if np.all(centers == new_centers):
            # Якщо центроїди залишились незмінними, вийти з циклу
            break
        # Оновлення центроїдів для наступної ітерації
        centers = new_centers

    return centers, labels

# Виклик функції find_clusters та відображення графіку
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Інший виклик функції find_clusters та відображення графіку
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

#  Виклик KMeans та відображення графіку
labels = KMeans(3, n_init=10, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
