
def drop_column(data,region_number):
    
    '''
    La función se encargara de eliminar la columna id de cada región
    '''
    
    data[region_number].drop(['id'], axis= 1, inplace = True)