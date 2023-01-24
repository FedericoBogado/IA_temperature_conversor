import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""---------------------------------------- Ejemplos para el entrenamiento -------------------------------------------"""
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)


"""------------------------------ Creacion de la estructura de la red neuronal con sus capas -------------------------"""
def creacion_del_modelo_y_entrenamiento(num):
    # capa = tf.keras.layers.Dense(units=1, input_shape=[1])
    # modelo = tf.keras.Sequential([capa])
    oculta1 = tf.keras.layers.Dense(units=5, input_shape=[1])
    oculta2 = tf.keras.layers.Dense(units=5)
    salida = tf.keras.layers.Dense(units=1)
    modelo = tf.keras.Sequential([oculta1, oculta2, salida])

    # Entrenamiento de la red
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

    print("Comenzamos el entrenamiento...")
    historial = modelo.fit(celsius, fahrenheit, epochs=200, verbose=False)
    print("Ya esta entrenada")
    
    # Prediccion
    resultado = modelo.predict([num])
    print("Son: ", resultado, " fahrenheit")
    #print(capa.get_weights())

    return historial


"""-------------------------------------- Grafico de la curva de aprendizaje ----------------------------------------"""
def grafico(historial):
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history["loss"])
    plt.show()


if __name__ == "__main__":
    num = int(input("Introduce los celcius a calcular: "))
    grafico(creacion_del_modelo_y_entrenamiento(num))