import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.5, random_state = 0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)

# Виведення коефіцієнтів регресії та важливих показників
print("Коефіцієнти регресії:")
print("Коефіцієнти:", regr.coef_)
print("Перехоплення:", regr.intercept_)

# Обчислення та виведення на екран показників
print("\nПоказники:")
print("R2 score:", r2_score(ytest, ypred))
print("Mean Absolute Error:", mean_absolute_error(ytest, ypred))
print("Mean Squared Error:", mean_squared_error(ytest, ypred))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()