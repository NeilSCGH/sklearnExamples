from sklearn import linear_model

x_Set = [[0, 0], [0, 1], [1, 1], [1, 0]]
y_Set = [0, 2, 3, 1]

reg = linear_model.Lasso(alpha=0.1)
reg.fit(x_Set, y_Set)

predicted = reg.predict(x_Set)
for x, y, y_pred in zip(x_Set, y_Set, predicted):
	print(x, y , y_pred)