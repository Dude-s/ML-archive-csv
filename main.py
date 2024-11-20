import pandas as pd
import numpy as np

# Función para aplicar la sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función para entrenar una regresión logística
def train_logistic_regression(x, y, lr=0.01, epochs=1000):
    m, n = x.shape
    weights = np.zeros(n)  # Inicializar pesos
    bias = 0  # Inicializar sesgo

    for _ in range(epochs):
        # Predicción
        linear_model = np.dot(x, weights) + bias
        predictions = sigmoid(linear_model)

        # Gradientes
        dw = (1 / m) * np.dot(x.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)

        # Actualizar pesos y sesgo
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

# Función para realizar predicciones
def predict(x, weights, bias, threshold=0.5):
    linear_model = np.dot(x, weights) + bias
    probabilities = sigmoid(linear_model)
    return (probabilities >= threshold).astype(int)

# Cargar el archivo CSV
data = pd.read_csv('train.csv')

# Seleccionar columnas relevantes y manejar valores faltantes
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# Rellenar valores faltantes con la media
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Convertir la columna 'Sex' en valores numéricos
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Seleccionar características (x) y la clase (y)
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y = data['Survived'].values

# Mezclar los datos de forma aleatoria
indices = np.arange(len(x))  # Crear un arreglo con los índices
np.random.shuffle(indices)   # Barajar los índices

x = x[indices]  # Reordenar las características según los índices barajados
y = y[indices]  # Reordenar las etiquetas según los índices barajados

# Dividir los datos manualmente en entrenamiento y prueba (80% - 20%)
train_size = int(len(x) * 0.8)  # Calcular el tamaño del conjunto de entrenamiento
x_train, x_test = x[:train_size], x[train_size:]  # División de x
y_train, y_test = y[:train_size], y[train_size:]  # División de y

# Entrenar el modelo de regresión logística
weights, bias = train_logistic_regression(x_train, y_train, lr=0.01, epochs=5000)

# Probar el modelo con los datos de prueba
y_pred = predict(x_test, weights, bias)

# Calcular precisión usando las predicciones
accuracy = np.mean(y_pred == y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Mostrar bloques de datos y predicciones
print("\nDatos de prueba (x_test):")
print(x_test[:5])  # Mostrar las primeras 5 filas de x_test

print("\nValores reales (y_test):")
print(y_test)  # Mostrar los primeros 5 valores reales

print("\nPredicciones del modelo (y_pred):")
print(y_pred)  # Mostrar las primeras 5 predicciones


print("\nValores reales(Primeras 10 muestras) (y_test):")
print(y_test[:10])  # Mostrar los primeros 5 valores reales

print("\nPredicciones del modelo(Primeras 10 muestras) (y_pred):")
print(y_pred[:10])  # Mostrar las primeras 5 predicciones
