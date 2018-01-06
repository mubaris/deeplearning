import pickle
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

X = pickle.load(open("x_data.pkl", "rb"))
y = pickle.load(open("y_data.pkl", "rb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(25, input_dim=4, activation="tanh"))
model.add(Dense(20, activation="tanh"))
model.add(Dense(16, activation="tanh"))
model.add(Dense(10, activation="tanh"))
model.add(Dense(2, activation="linear"))

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=100, verbose=2)

yhat = model.predict(X_test)

for i in range(len(yhat)):
	print(yhat[i], y_test[i])
