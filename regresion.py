import numpy as np

class RegresionLinealSimple:
    def _init_(self):
        self.b1 = None
        self.b0 = None

    def ajustar(self, X, y):
        n = len(X)
        sumX = np.sum(X)
        sumY = np.sum(y)
        sumXY = np.sum(X * y)
        sumX2 = np.sum(X**2)

        self.b1 = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX**2)
        self.b0 = (sumY - self.b1 * sumX) / n

    def coeficientes(self):
        return self.b1, self.b0
    
    def predecir(self, X):
        return self.b1 * X + self.b0

# Datos de Sales y Advertising
advertising = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) #X
sales = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18]) #Y


# Instancia de la clase RegresionLinealSimple
modelo = RegresionLinealSimple()

# Ajustar el modelo a los datos de sales y advertising
modelo.ajustar(advertising, sales)

# Obtener los coeficientes de regresión
b1, b0 = modelo.coeficientes()

# Imprimir los coeficientes de regresión
print("Coeficiente de pendiente (b1):", b1)
print("Coeficiente de intercepción (b0):", b0)

# Prediccion de valores
valores_advertising = [10, 11, 12]
print("Predicciones de valores de Advertising:")
for valor in valores_advertising:
    prediccion = modelo.predecir(valor)
    print(f"Advertising: {valor}, Predicción de sales: {prediccion}")