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

absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Absenteeism_at_work.csv')
normalized_absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Normalized_absenteeism_at_work.csv')

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

def normalizar(data):
    #------------------- Normalizacao -----------------------------------------
    data.info()
    #Aplicar One Hot encoding na coluna reason for absence
    data_dummies = pandas.get_dummies(data['Reason for absence'], prefix = 'Reason')
    #Remover colunas ID e Reason for absence, não representam nada na analise
    num_data = data.drop(labels=['ID','Reason for absence'], axis=1)

    #########################
    from sklearn import preprocessing

    # normalizer = preprocessing.MinMaxScaler()
    # data_normalizer_model = normalizer.fit(num_data.values)
    # normalized_data = data_normalizer_model.fit_transform(num_data)
    # normalized_data = pandas.DataFrame(normalized_data,columns = num_data.columns.values)
    normalized_data = num_data.join(data_dummies,how='left')
    #########################
    print('\n_______\n')
    print(normalized_data)
    normalized_data.to_csv(normalized_absenteeism_path,sep=',',encoding='utf-8')
    return normalized_data

def clusterizar(norm_data):
    #------------------- Clusterizacao - Indutivo -----------------------------
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
    return absenteeism_kmeans_model

def main():
    #------------------- Abrir arquivo de dados -------------------------------
    data = pandas.read_csv(absenteeism_path,delimiter=';')
    print(data)
    norm_data = normalizar(data)
    norm_data_y = norm_data['Absenteeism time in hours']
    norm_data_x = norm_data.drop(labels='Absenteeism time in hours',axis=1)

    #------------------- Inferencia - Interface -----------------------------
    cluster_model = clusterizar(norm_data)
    print('centroides obtidos')
    print(cluster_model.cluster_centers_)

    #NOVA INSTANCIA
    new_patient = [[12,3,2,155,12,14,34,280.54900000000004,98,0,1,2,1,0,0,95,196,25,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]]
    print(f'new patient input:{new_patient}')

    print('index of the clusters that the new patient may be')
    print(
        cluster_model.predict(new_patient)
    )
    print('\ncluster center that the new patient may be:\n')
    cluster_center = cluster_model.cluster_centers_[cluster_model.predict(new_patient)]
    print(cluster_center)
    print(len(cluster_center[0]))
    print(len(new_patient[0]))
    print(len(norm_data.columns.values))
if __name__ == '__main__':
    main()