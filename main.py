import numpy as np
import pandas as pd

# Cargar el archivo .csv con pandas
datos = pd.read_csv('train.csv')


# Procesamiento de los datos
def procesar_datos(df):
    caracteristicas = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    objetivo = 'Survived'

    # Codificar la columna 'Sex' (male = 0, female = 1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Codificar la columna 'Embarked' (C = 0, Q = 1, S = 2)
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Imputar valores faltantes en 'Age' y 'Fare' con la media de la columna
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    # Selección de las características (X) y la etiqueta (y)
    X = df[caracteristicas].values
    y = df[objetivo].values

    return X, y


# Implementación de KNN
def predecir_knn(X_entrenamiento, y_entrenamiento, X_prueba, k=3):
    distancias = np.sqrt(((X_entrenamiento - X_prueba) ** 2).sum(axis=1))  # Calcula la distancia euclidiana
    indices_ordenados = np.argsort(distancias)  # Ordena los índices según la distancia
    etiquetas_vecinos = y_entrenamiento[indices_ordenados[:k]]  # Obtiene las etiquetas de los k vecinos más cercanos
    prediccion = np.bincount(etiquetas_vecinos).argmax()  # Predicción basada en la mayoría
    return prediccion

# Solicitar datos al usuario
def obtener_datos_usuario():
    print("Introduce los siguientes datos para predecir si el pasajero estaría vivo o muerto.")
    pclass = int(input("Pclass (1, 2, 3): "))
    sexo = input("Sexo (male/female): ").strip().lower()
    sexo = 0 if sexo == "male" else 1
    edad = float(input("Edad: "))
    sibsp = int(input("Número de hermanos/esposos a bordo: "))
    parch = int(input("Número de padres/hijos a bordo: "))
    tarifa = float(input("Tarifa pagada: "))
    embarque = input("Puerto de embarque (C, Q, S): ").strip().upper()
    embarque = {'C': 0, 'Q': 1, 'S': 2}.get(embarque, 2)  # Asigna el valor correspondiente

    return np.array([[pclass, sexo, edad, sibsp, parch, tarifa, embarque]])


# Predecir usando el modelo KNN
def hacer_prediccion():
    # Pedir los datos del usuario
    datos_usuario = obtener_datos_usuario()

    # Usar los primeros 80% de los datos para entrenamiento y el 20% para pruebas
    indice_division = int(0.8 * len(X))
    X_entrenamiento, y_entrenamiento = X[:indice_division], y[:indice_division]

    # Realizar la predicción usando KNN
    prediccion = predecir_knn(X_entrenamiento, y_entrenamiento, datos_usuario[0])

    # Mostrar el resultado
    if prediccion == 1:
        print("El pasajero estaría vivo.")
    else:
        print("El pasajero estaría muerto.")



# Procesar los datos de entrenamiento
X, y = procesar_datos(datos)

#Hacer prediccion
hacer_prediccion()
