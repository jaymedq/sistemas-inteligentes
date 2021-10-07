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

def optimal_cluster_number(mapping):
    #determinar as coordenadas do primeiro ponto da curva
    x1 = 2
    y1 = mapping[0]

    #determinar o ultimo ponto da curva
    x2 = len(mapping)
    y2 = mapping[-1]

    distances = [] # para armazenar o comprimento das perpendiculares à cortante que será tracada
    for i in range(len(mapping)):
        current_x = 2+i
        current_y = mapping[i]
        numerator = abs((y2-y1)*current_x - (x2-x1)*current_y + x2*y1 - y2*x1)
        denominator = math.sqrt((y2-y1)**2+(x2-x1)**2)
        distances.append(numerator/denominator)
    return distances.index(max(distances))+2

def main():
    #Abrir arquivo de dados
    absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Absenteeism_at_work.csv')
    normalized_absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Normalized_absenteeism_at_work.csv')
    data = pandas.read_csv(absenteeism_path,delimiter=';')
    print(data)
    #------------------- Normalizacao ----------------------------
    data.info()
    #Aplicar One Hot encoding na coluna reason for absence
    data_dummies = pandas.get_dummies(data['Reason for absence'], prefix = 'Reason')
    data = pandas.concat([data,data_dummies],axis = 1)
    #Remover colunas ID e Reason for absence, não representam nada na analise
    norm_data = data.drop(labels=['ID','Reason for absence'], axis=1)
    print('\n_______\n')
    print(norm_data)
    norm_data.to_csv(normalized_absenteeism_path,sep=',',encoding='utf-8')

    #------------------- Clusterizacao ----------------------------
    # Coverter os dados em uma matriz numerica (nd_array)
    data_x = norm_data.values
    sampleRange = range(2,100)
    inertias = list()
    for k in sampleRange:
        kmeansmodel = KMeans(n_clusters=k, random_state=4).fit(data_x)
        inertias.append(kmeansmodel.inertia_)
    
    nk_inertias = optimal_cluster_number(inertias)
    print(f'ideal number of clusters: inertias [ {nk_inertias} ]')
    kmeansmodel = KMeans(n_clusters=nk_inertias, random_state=4).fit(data_x)
    #Salvar o modelo para uso posterior
    with open("models/kmeansmodel_absenteeism.pkl", "wb") as kmeansmodel_file:
        pickle.dump(kmeansmodel, kmeansmodel_file)
    with open("models/kmeansmodel_absenteeism.pkl", 'rb') as kmeansmodel_file:
        absenteeism_kmeans_model = pickle.load(kmeansmodel_file)
    print('centroides obtidos')
    print(absenteeism_kmeans_model.cluster_centers_)
    
    #####################

if __name__ == '__main__':
    main()