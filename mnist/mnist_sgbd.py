from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle

# Carregar MNIST como NumPy arrays
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(np.uint8)

# Separar treino e teste
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Criar rótulos binários: 5 ou não
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Treinar modelo
model_sgd = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
model_sgd.fit(X_train, y_train_5)

s = y_test.size

y_pred = []
for i in range(s):
    y_pred.append(model_sgd.predict([X_test[i, :]]))

# Exportar modelo para arquivo .pkl
with open('models/model-sgd.pkl', 'wb') as f:
    pickle.dump(model_sgd, f)
