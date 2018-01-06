import pickle
import numpy as np

MIN = -6
MAX = 6
NUM = 20

space = np.linspace(MIN, MAX, NUM)

X = np.zeros((NUM * NUM * NUM * NUM, 4))

count = 0

for i in space:
	for j in space:
		for k in space:
			for l in space:
				X[count] = np.array([i, j, k, l])
				count += 1

x1 = X[:, 0]
y1 = X[:, 1]
x2 = X[:, 2]
y2 = X[:, 3]

p = (x1*x2 + y1*y2) / (x2*x2 + y2*y2)
q = (x2*y1 - x1*y2) / (x2*x2 + y2*y2)

y = np.stack((p, q), axis=-1)

ip = open("x_data.pkl", "wb")
op = open("y_data.pkl", "wb")

pickle.dump(X, ip)
pickle.dump(y, op)
