
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def modeling(data,region_number,x,y, x_train, y_train, x_valid, y_valid, model, predict_v, RMSE):
    
    '''
    La función se encargara por región de:
    - Separar la data en 75% para el entrenamiento y 25% para validación
    - Hacer un escaldado estándar en la data de entrenamiento
    - Extraer el modelo de regresión lineal, entrenar, hacer las predicciones y calcular el RMSE
    '''
    
    x[region_number] = data[region_number].drop(columns = ['product'])
    y[region_number] = data[region_number]['product']
    x_train[region_number], x_valid[region_number], y_train[region_number], y_valid[region_number] = train_test_split(x[region_number], y[region_number], random_state= 42, test_size= 0.25)
    
    scaler = StandardScaler()
    scaler.fit(x_train[region_number])
    x_train[region_number] = pd.DataFrame(scaler.transform(x_train[region_number]), index = x_train[region_number].index, columns = x_train[region_number].columns)
    x_valid[region_number] = pd.DataFrame(scaler.transform(x_valid[region_number]), index = x_valid[region_number].index, columns = x_valid[region_number].columns)
    
    model[region_number] = LinearRegression()
    model[region_number].fit(x_train[region_number], y_train[region_number])
    predict_v[region_number] = model[region_number].predict(x_valid[region_number])
    RMSE[region_number] = mean_squared_error(y_valid[region_number], predict_v[region_number])** 0.5
    
    print('\nRegión', region_number)
    print('Promedio de stock de producto por pozo en las predicciones {} = {:.2f}'.format(region_number, predict_v[region_number].mean()))
    print('RMSE de la región {} = {:.2f}\n'.format(region_number, RMSE[region_number]))