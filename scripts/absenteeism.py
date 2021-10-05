# Modelo de clusters
# DETERMINAR O NUMERO DE CLUSTERS

import os
import matplotlib.pyplot as plt
import numpy
import math
import pandas
import pickle
from scipy.spatial.distance import cdist
from sklearn import cluster
from sklearn.cluster import KMeans

import os
import pandas

def main():
    #Abrir arquivo de dados
    absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Absenteeism_at_work.csv')
    normalized_absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Normalized_absenteeism_at_work.csv')
    data = pandas.read_csv(absenteeism_path,delimiter=';')
    print(data)
    #Normalizar
    data.info()
    data_x = data.copy()
    #Aplicar One Hot encoding na coluna reason for absence
    data_dummies = pandas.get_dummies(data_x['Reason for absence'], prefix = 'Reason')
    data_x = pandas.concat([data_x,data_dummies],axis = 1)
    #Remover colunas ID e Reason for absence, não representam nada na analise
    norm_data = data_x.drop(labels=['ID','Reason for absence'], axis=1)
    print('\n_______\n')
    print(norm_data)
    norm_data.to_csv(normalized_absenteeism_path,sep=';',encoding='utf-8')

if __name__ == '__main__':
    main()