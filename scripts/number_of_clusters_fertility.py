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
#Fertility ja eh uma base normalizada mas possui dados categoricos na coluna output

import os
import pandas

#Abrir arquivo de dados
path = os.path.join(os.getcwd(),'database','fertility-Diagnosis.csv')
data = pandas.read_csv(path)

#Separar dados numericos e categoricos
temp_data = data.drop(columns=['Output'])
cat_data = data['Output']

#Normalizar output com hot encoding
normalized_cat_data = pandas.get_dummies(cat_data,prefix=cat_data.name)
#concatenar dados
final_data = temp_data.join(normalized_cat_data,how='left')
final_data.to_csv(os.path.join(os.getcwd(),'database','Normalized-Fertility-Diagnosis.csv'), sep=';', encoding='utf-8')

# Coverter os dados em uma matriz numerica (nd_array)
data_x = final_data.values

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

################
#METODO DAS INERCIAS
inertias = list()
for k in sampleRange:
    kmeansmodel = KMeans(n_clusters=k, random_state=4).fit(data_x)
    inertias.append(kmeansmodel.inertia_)

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
print(f'ideal number of clusters: inertias [ {nk_inertias} ]')

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

#NOVA INSTANCIA
new_patient = [[1,0.67,1,0,0,0,1,-1,0.25,0,0]] # Estes dados foram normalizados com o mdesmo modelo normalizador utilizado sobre a base

print(f'new patient input:{new_patient}')

print('index of the clusters that the new patient may be')
print(
    fertility_kmeans_model.predict(new_patient)
)
print('\ncluster center that the new patient may be:\n')
cluster_center = fertility_kmeans_model.cluster_centers_[fertility_kmeans_model.predict(new_patient)]

def find_nearest(array,value):
    for i in range(1,len(array)):
        if array[i-1] < value <= array[i]:
            return array[i]
        else:
            continue
    return array[i]

def parse_fertility_cluster_columns(cluster):
    cluster = cluster[0].copy()
    season_map = {
        -1:'winter',
        -0.33:'spring',
        0.33:'summer',
        1:'fall'
    }
    alcohol_consumption_map = {
        0:'several times a day', # Valor nao existe no dataset, mas pode ser uma entrada
        0.2:'several times a day', 
        0.4:'every day', 
        0.6:'several times a week', 
        0.8:'once a week', 
        1:'hardly ever or never'
    }
    season = season_map.get(find_nearest(list(season_map.keys()), cluster[0]),'NA'),
    age = 18+(36-18)*cluster[1]
    childish_diseases= 'yes' if cluster[2]<0.5 else 'no'
    accident_or_serious_trauma = 'yes' if cluster[3]<0.5 else 'no'
    surgical_intervention = 'yes' if cluster[4]<0.5 else 'no'
    high_fevers_in_the_last_year = 'in 3 months' if cluster[5] < 0 else 'more than 3 months ago' if cluster[5] == 0 else 'more than a year'
    alcohol_index = find_nearest(list(alcohol_consumption_map.keys()), cluster[6])
    frequency_of_alcohol_consumption = alcohol_consumption_map.get(alcohol_index,'NA')
    smoking_habit = 'never' if cluster[7] < 0 else 'occasional' if cluster[7] == 0 else 'daily'
    number_of_hours_spent_sitting_per_day= cluster[8]*16
    diagnosis = 'normal' if cluster[9] else 'altered'

    parsed_result = f"""
    Season: {season}\n    age: {age}\n    childish_diseases: {childish_diseases}
    accident_or_serious_trauma: {accident_or_serious_trauma}\n    surgical_intervention: {surgical_intervention}\n    high_fevers_in_the_last_year: {high_fevers_in_the_last_year}
    frequency_of_alcohol_consumption:{frequency_of_alcohol_consumption}\n    smoking_habit:{smoking_habit}\n    number_of_hours_spent_sitting_per_day: {number_of_hours_spent_sitting_per_day}
    diagnosis:{diagnosis}
    """
    return parsed_result

parsed_cluster_center = parse_fertility_cluster_columns(cluster_center)
print(parsed_cluster_center)
