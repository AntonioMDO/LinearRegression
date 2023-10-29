import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def revenue(y,predictions,count, barrel, budget):
    
    '''
    La función utilizara los datos indicados en las condiciones,
    junto con las predicciones y los datos objetivos para realizar el calculo
    '''
    
    predict_sorted = predictions.sort_values(ascending = False)
    selected = y[predict_sorted.index][:count]
    return (selected.sum() * barrel) - budget



def bootstrapping(region_number, predict_v, x_valid, y_valid, max_points, points, mean, confidence_interval_max, confidence_interval_min, probability_of_loss, barrel, budget):
  
  '''
  El bootstrapping se aplica con los datos predichos por el modelo y se seleccionaran 500 valores aleatorios en diferentes muestras.
  Seguidamente, se aplica la función para calcular las ganancias de los 200 puntos con mayor cantidad de producto.
  Y finalmente se calcula:
  - La ganancia media
  - Los intervalos de confianza al 95%
  - La probabilidad y porcentaje de perdida
  '''

  predictions = pd.Series(predict_v[region_number], index = x_valid[region_number].index)
  
  loss = 0
  values = []
  state = np.random.RandomState(42)
  
  for i in range(1000):
    subsample_v = y_valid[region_number].sample(n = max_points, replace = True, random_state = state )
    subsample_p = pd.Series(predictions[subsample_v.index])
  
    profit = revenue(subsample_v, subsample_p, points, barrel, budget)
    
    values.append(profit)
    
    mean[region_number] = np.mean(values)
    confidence_interval_max[region_number] = np.quantile(values, 0.025)
    confidence_interval_min[region_number] = np.quantile(values, 0.975)
    if profit < 0:
      loss += 1
  probability_of_loss[region_number] = (loss /max_points)
  
  return values



def conclusions(region_number,mean, confidence_interval_max, confidence_interval_min, probability_of_loss, predict_v, x_valid, y_valid, max_points, points, barrel, budget):
    
    '''
    La función cumplirá el objetivo de:
    - Graficar los histogramas de los pozos seleccionados
    - Imprimir los resultados por región de los cálculos realizados en la función de bootstrapping
    '''
    
    fig = plt.figure(figsize=(10, 4))
    ax = sns.histplot(bootstrapping(region_number, predict_v, x_valid, y_valid, max_points, points, mean, confidence_interval_max, confidence_interval_min, probability_of_loss, barrel, budget),
                      bins = 40,
                      ax = fig.add_subplot(),
            )
    plt.title(f'Distribution of profit in the region {region_number}', fontweight = 'bold')
    plt.ylabel('Count')
    plt.show()
    
    print('\nRegión', region_number)
    print('Ganancia media de {:.2f} millones.'.format(mean[region_number]))
    print('Intervalo de confianza del 95% entre: {:.2f} - {:.2f} millones.'.format(confidence_interval_max[region_number], confidence_interval_min[region_number]))
    print('Y con la probabilidad de perdida de {:.2f}, representando un porcentaje de {:.2%}.'.format(probability_of_loss[region_number], probability_of_loss[region_number]))
