# Modelo de clusters
# DETERMINAR O NUMERO DE CLUSTERS

import os
import matplotlib.pyplot as plt
import numpy
import math
import pandas
import pickle
from pandas.core.frame import DataFrame
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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

def normalizar(data:DataFrame):
    #------------------- Normalizacao -----------------------------------------
    data.info()
    #Aplicar One Hot encoding na coluna reason for absence
    data_dummies = pandas.get_dummies(data['Reason for absence'], prefix = 'Reason')
    #Remover colunas ID e Reason for absence, não representam nada na analise
    num_data = data.drop(labels=['ID','Reason for absence'], axis=1)

    #########################
    from sklearn import preprocessing

    normalizer = preprocessing.MinMaxScaler()
    data_normalizer_model = normalizer.fit(num_data.values)
    normalized_data_array = data_normalizer_model.fit_transform(num_data)
    normalized_data_frame = pandas.DataFrame(normalized_data_array,columns = num_data.columns)
    normalized_data = normalized_data_frame.join(data_dummies,how='left')
    #########################
    print('\n___Normalizado____\n')
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
    # data_y = data[['Absenteeism time in hours']]
    # data_x = data.drop(labels='Absenteeism time in hours',axis=1)
    norm_data = normalizar(data)
    modelo = clusterizar(norm_data)
    new_input = [[0.5833333333333333,0.25,0.0,0.4074074074074074,0.19148936170212766,0.5357142857142857,0.9999999999999999,0.3392959350627578,0.6315789473684212,0.0,0.0,0.5,0.0,0.0,0.125,0.1730769230769229,0.2727272727272725,0.1578947368421053,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    pred = modelo.predict(new_input)
    print(f'indice do cluster encaixado: {pred}')
    match_centroide=modelo.cluster_centers_[pred]
    print(match_centroide)
    print(f'horas previstas para faltar: {match_centroide[0][18]*120}')
if __name__ == '__main__':
    main()