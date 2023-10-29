import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import drop_column
from modeling import modeling
from profit_calculation import conclusions

# Cargar y visualizar los datos
data = dict()

def view_the_data(region_number):
    
    '''
    La función hará lo siguiente por cada región:
    - Lectura de los datos para ser ejecutado en terminal o ventana interactiva
    - Primer contacto con los datos (describe, head,info)
    - Revisión de duplicados
    - Histogramas
    '''
    try:
        data[region_number] = pd.read_csv(f'../data/geo_data_{region_number}.csv')   
    except:
        data[region_number] = pd.read_csv(f'data/geo_data_{region_number}.csv')
        
    print('\nDescribe:')
    print(data[region_number].describe())
        
    print('\nDataframe:')
    print(data[region_number].head())
        
    print('\nInfo:')
    data[region_number].info()
        
    print('\nDuplicates:')
    print(data[region_number].duplicated().sum())
        
    print('\nRegion', region_number, 'graphics:')
    data[region_number].hist(bins = 100, figsize = (15,8),)
    plt.show()
        

for i in range(0,3):
    view_the_data(i)
    
#Preprocesamiento

for i in range(0,3):
    drop_column(data,i)
    
#Modelado

x = dict()
y = dict()
x_train = dict()
y_train = dict()
x_valid = dict()
y_valid = dict()
model = dict()
predict_v = dict()
RMSE = dict()

for i in range(0,3):
    modeling(data,i ,x ,y , x_train, y_train, x_valid, y_valid, model, predict_v, RMSE)
    
#Calculo de ganancias por región

budget = 100_000_000 #Presupuesto para abrir los pozos
barrel = 4500 #Costo de una unidad de productoS
max_points = 500 #Cantidad de pozos estudiados
points = 200 #Cantidad de pozos que se abrirán

volume = budget / barrel
max_volume = int(volume / points)
print('El volumen necesario por pozo para no generar perdidas debe ser en promedio:', max_volume, 'de producto.')

mean = dict()
confidence_interval_max = dict()
confidence_interval_min = dict()
probability_of_loss = dict()

#Aplicar el bootstrapping con las gráficas
for i in range(0,3):
    conclusions(i, mean, confidence_interval_max, confidence_interval_min, probability_of_loss, predict_v, x_valid, y_valid, max_points, points, barrel, budget)