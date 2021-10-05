# Modelo de clusters
# DETERMINAR O NUMERO DE CLUSTERS

import os
import matplotlib.pyplot as plt
import numpy
import math
import pandas
import pickle
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import subprocess
# 1 Abrir os dados
path = os.path.join(os.getcwd(),'database','Normalized-Fertility-Diagnosis.csv')
data = pandas.read_csv(path)

# imprimir as colunas do arquivo
#print(data.columns)

# Separar os atributos da classe
# data_x = data.drop(columns=['Output'])
# Coverter os dados em uma matriz numerica (nd_array)
data_x = data.values
#print(type(data_x))

#preparar um objeto para a plotagem do grafico das distorcoes
fig, ax = plt.subplots()

# METODO DAS DISTORCOES

#calcular as distorcoes
#ensaiar entre 2 e 10 clusters por modelo
distortions = [] # matriz para armazenar as distorcoes
sampleRange = range(2,100)
for k in sampleRange:
    kmeansmodel = KMeans(n_clusters = k,random_state=4).fit(data_x) # o fit obtem o modelo binario para os clusters
    distortions.append(
        sum(
           numpy.min(
               cdist(data_x,kmeansmodel.cluster_centers_),axis=1
           )/data_x.shape[0]
        )
    )
#print(distortions)
#plotar as dirtorcoes
ax.plot(sampleRange,distortions,'bx-')
ax.set(xlabel='N Clusters', ylabel = 'Distortions', title = 'Elbow method')
ax.grid()
fig.savefig('images/elbow_distortions.png')
plt.show()

################
#METODO DAS INERCIAS
inertias = list()
for k in sampleRange:
    kmeansmodel = KMeans(n_clusters=k, random_state=4).fit(data_x)
    inertias.append(kmeansmodel.inertia_)

ax.plot(sampleRange,inertias, 'bx-')
ax.set(xlabel='N Clusters', ylabel = 'inertias', title = 'inertias method')
ax.grid()
fig.savefig('images/elbow_inertias.png')
plt.show()

#METODO PARA DETERMINAR O NUMERO IDEAL DE GRUPO
#Funcao
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
########################
nk_distortions = optimal_cluster_number(distortions)
#print(f'numero ideal de clusters: distortions [ {nk_distortions} ]')
#print(nk_distortions)
nk_inertias = optimal_cluster_number(inertias)
print(f'numero ideal de clusters: inertias [ {nk_inertias} ]')
print(nk_inertias)

# OBTER E SALVAR O MODELO DE CLUSTERS 
# COM BASE NO NUMERO IDEAL DE GRUPOS CALCULADO ANTERIORMENTE
kmeansmodel = KMeans(n_clusters=nk_inertias, random_state=4).fit(data_x)
#Salvar o modelo para uso posterior
# print('centroides obtidos')
# print(kmeansmodel.cluster_centers_)


with open("models/kmeansmodel_fertility.pkl", "wb") as kmeansmodel_file:
    pickle.dump(kmeansmodel, kmeansmodel_file)

#########################
#UTILIZAR O MODELO PERSISTIDO PARA INFERIR SOBRE UMA NOVA INSTANCIA
# Abrir o modelo salvo em disco
with open("models/kmeansmodel_fertility.pkl", 'rb') as kmeansmodel_file:
    fertility_kmeans_model = pickle.load(kmeansmodel_file)
new_patient = [[1,0.67,1,0,0,0,1,-1,0.25]] # Estes dados foram normalizados com o mdesmo modelo normalizador utilizado sobre a base
print('number of clusters that the new patient may be')
print(
    fertility_kmeans_model.predict(new_patient)
)
print('cluster center that the new patient may be')
print(
    fertility_kmeans_model.cluster_centers_[fertility_kmeans_model.predict(new_patient)]
)