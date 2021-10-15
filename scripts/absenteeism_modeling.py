# Modelo de clusters
# DETERMINAR O NUMERO DE CLUSTERS
import os
import math
import pandas
import pprint
import pickle
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np

import os
import pandas

pp = pprint.PrettyPrinter(indent=4)
absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Absenteeism_at_work.csv')
normalized_absenteeism_path = os.path.join(os.getcwd(),'database','Absenteeism_at_work_AAA','Normalized_absenteeism_at_work.csv')
RANDOM_STATE = 4

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

def normalizeDataArray(dataArray:DataFrame):
    #Remover colunas ID e Reason for absence, não representam nada na analise
    num_data = dataArray.drop(labels=['ID','Reason for absence'], axis=1)
    with open('absenteeism_data_normalizer_model.pkl', 'rb') as normalizer_file:
        data_normalizer_model=pickle.load(normalizer_file)
    normalized_data_array = data_normalizer_model.fit_transform(num_data)

    hot_encoded_reason = [0]*28
    hot_encoded_reason[dataArray['Reason for absence'][0]] = 1

    normalized_data_array=np.concatenate((normalized_data_array[0],hot_encoded_reason))
    print(f'\nEntrada do usuario normalizada: {normalized_data_array}')
    return normalized_data_array

def normalizeDataSet(data:DataFrame):
    #------------------- Normalizacao -----------------------------------------
    # data.info()
    #Aplicar One Hot encoding na coluna reason for absence
    data_dummies = pandas.get_dummies(data['Reason for absence'], prefix = 'Reason')
    #Remover colunas ID e Reason for absence, não representam nada na analise
    num_data = data.drop(labels=['ID','Reason for absence'], axis=1)

    #########################
    from sklearn import preprocessing

    normalizer = preprocessing.MinMaxScaler()
    data_normalizer_model = normalizer.fit(num_data.values)
    pickle.dump(data_normalizer_model, open('absenteeism_data_normalizer_model.pkl', 'wb'))
    normalized_data_array = data_normalizer_model.fit_transform(num_data)
    normalized_data_frame = pandas.DataFrame(normalized_data_array,columns = num_data.columns)
    normalized_data = normalized_data_frame.join(data_dummies,how='left')
    #########################
    print('\n___ Dataframe Normalizado ____\n')
    print(normalized_data)
    normalized_data.to_csv(normalized_absenteeism_path,sep=',',encoding='utf-8')
    return normalized_data

def clusterizarKmeans(norm_data):
    #------------------- Clusterizacao - Indutivo -----------------------------
    # Coverter os dados em uma matriz numerica (nd_array)
    data_x = norm_data.values
    sampleRange = range(2,100)
    inertias = list()
    for k in tqdm(sampleRange,'Clusterizando com kmeans'):
        kmeansmodel = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(data_x)
        inertias.append(kmeansmodel.inertia_)
    nk_inertias = optimal_cluster_number(inertias)
    # print(f'ideal number of clusters: inertias [ {nk_inertias} ]')
    kmeansmodel = KMeans(n_clusters=nk_inertias, random_state=RANDOM_STATE).fit(data_x)
    with open("models/kmeansmodel_absenteeism.pkl", 'wb') as kmeansModelFile:
        pickle.dump(kmeansmodel, kmeansModelFile)
    return kmeansmodel

def clusterizarGMM(norm_data,nClusters):
    gmm = GaussianMixture(n_components=nClusters,random_state=RANDOM_STATE)
    gmmModel = gmm.fit(norm_data)
    with open("models/gmm_absenteeism.pkl", 'wb') as gmmModelFile:
        pickle.dump(gmmModel, gmmModelFile)
    return gmmModel

def main():
    print("""
    Um estudo sobre o absentismo no trabalho foi efetuado na Universidade Nove de Julho.
    Este sistema utiliza o dataset disponibilizado em https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work (Disponivel em 08/10/2021).
    Para a clusterização dos grupos de funcionarios, possibilitando a inferencia de um novo funcionario para verificar a que grupo ele se encaixa.
    Dessa maneira, podemos estimar o numero de horas de absentismo deste funcionario, dadas as condições Descritas durante a entrada de informações.
    """)

    #------------------- Abrir arquivo de dados -------------------------------
    data = pandas.read_csv(absenteeism_path,delimiter=';')
    #
    #-------------------        Normalizar      -------------------------------
    normData = normalizeDataSet(data)
    #-------------------         Inducao        -------------------------------

    # Cluster - Kmeans
    kmeansModel = clusterizarKmeans(normData)
    #Teste
    inferenciaPadrao = [[0.5833333333333333,0.25,0.0,0.4074074074074074,0.19148936170212766,0.5357142857142857,0.9999999999999999,0.3392959350627578,0.6315789473684212,0.0,0.0,0.5,0.0,0.0,0.125,0.1730769230769229,0.2727272727272725,0.1578947368421053,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    kmeansPred = kmeansModel.predict(inferenciaPadrao)
    # print(f'indice do cluster encaixado: {kmeansPred}')
    kmeansCentroidMatch=kmeansModel.cluster_centers_[kmeansPred]
    # print(kmeansCentroidMatch)
    # print(f'horas previstas para faltar: {kmeansCentroidMatch[0][18]*120}')

    # Cluster - EM
    gmmModel = clusterizarGMM(normData, nClusters = kmeansModel.n_clusters)
    # print(f'indice do cluster encaixado: {gmmModel.predict(inferenciaPadrao)}')
    gmmCentroidMatch = gmmModel.means_[gmmModel.predict(inferenciaPadrao)]
    # print(f'horas previstas para faltar: {gmmCentroidMatch[0][18]*120}')

    
    normNewInput = newInput(data.columns)
    normNewInput=inferenciaPadrao
    #FIXME usar absenteeismInference
    kmeansNewInputPred = kmeansModel.predict([normNewInput])
    print(f'KMEANS - indice do cluster encaixado: {kmeansPred}')
    kmeansCentroidMatch=kmeansModel.cluster_centers_[kmeansNewInputPred]
    print(kmeansCentroidMatch)
    print(f'KMEANS - horas previstas para faltar: {kmeansCentroidMatch[0][18]*120}')

    print(f'GMM - indice do cluster encaixado: {gmmModel.predict(inferenciaPadrao)}')
    gmmCentroidMatch = gmmModel.means_[gmmModel.predict(inferenciaPadrao)]
    print(f'GMM - horas previstas para faltar: {gmmCentroidMatch[0][18]*120}')


def newInput(dataColumns):
    print("""
    \n\nInferencia: Insira os dados do funcionario conforme a descrições a seguir:\n\n
    """)
    reasonMap = {
        1: "Certas doenças infecciosas e parasitárias",
        2: "Neoplasias",
        3: "Doenças do sangue e dos órgãos formadores de sangue e certos distúrbios que envolvem o mecanismo imunológico",
        4: "Doenças endócrinas, nutricionais e metabólicas",
        5: "Transtornos mentais e comportamentais",
        6: "Doenças do sistema nervoso",
        7: "Doenças dos olhos e anexos",
        8: "Doenças do ouvido e do processo mastóide",
        9: "Doenças do aparelho circulatório",
        10: "Doenças do aparelho respiratório",
        11: "Doenças do sistema digestivo",
        12: "Doenças da pele e tecido subcutâneo",
        13: "Doenças do sistema musculoesquelético e tecido conjuntivo",
        14: "Doenças do aparelho geniturinário",
        15: "Gravidez, parto e puerpério",
        16: "Certas condições originadas no período perinatal",
        17: "Malformações congênitas, deformações e anomalias cromossômicas",
        18: "Sintomas, sinais e achados clínicos e laboratoriais anormais, não classificados em outra parte",
        19: "Lesões, envenenamento e certas outras consequências de causas externas",
        20: "Causas externas de morbidade e mortalidade",
        21: "Fatores que influenciam o estado de saúde e o contato com os serviços de saúde.",
        22: "Follow-up de paciente",
        23: "Consulta medica",
        24: "Doacao de sangue",
        25: "Exame em laboratorio",
        26: "Nao justificada",
        27: "Fisioterapia",
        28: "Consulta odontologica"
    }
    employee = list()
    employee.append(0)#ID
    print("Insira a razao da falta seguin do o dicionario a seguir:")
    pp.pprint(reasonMap)
    employee.append(int(input()))
    employee.append(int(input('Mes da falta: ')))
    employee.append(int(input('Dia da semana (segunda (2), terça (3), quarta (4), quinta (5), sexta (6)): ')))
    employee.append(int(input('Estações do ano (verão (1), outono (2), inverno (3), primavera (4)): ')))
    employee.append(int(input('Despesa de transporte: ')))
    employee.append(int(input('Distância da residência ao trabalho (quilômetros): ')))
    employee.append(int(input('Tempo de serviço em anos: ')))
    employee.append(int(input('Idade: ')))
    employee.append(int(input('Média de carga de trabalho / dia: ')))
    employee.append(int(input('Hit target: ')))
    employee.append(int(input('Falha disciplinar (sim = 1; não = 0): ')))
    employee.append(int(input('Educação (ensino médio (1), graduação (2), pós-graduação (3), mestrado e doutorado (4)): ')))
    employee.append(int(input('Filhos (número de filhos): ')))
    employee.append(int(input('Bebe socialmente (sim = 1; não = 0): ')))
    employee.append(int(input('Fuma socialmente (sim = 1; não = 0): ')))
    employee.append(int(input('Animal de estimação (número de animais de estimação): ')))
    employee.append(int(input('Peso(kg): ')))
    employee.append(int(input('Altura(cm): ')))
    employee.append(int(input('Índice de massa corporal: ')))
    employee.append(int(input('Tempo de absenteísmo em horas: ')))
    newInputDf = pandas.DataFrame([employee],columns = dataColumns)
    normNewInput = normalizeDataArray(newInputDf)
    return normNewInput

def absenteeismInference(newData):
    with open("models/kmeansmodel_absenteeism.pkl", 'rb') as kmeansModelFile:
        kmeansModel = pickle.load(kmeansModelFile)

    with open("models/gmm_absenteeism.pkl", 'rb') as gmmModelFile:
        gmmModel = pickle.load(gmmModelFile)

    normNewInput = newInput(newData.columns)
    kmeansNewInputPred = kmeansModel.predict(normNewInput)
    print(f'KMEANS - indice do cluster encaixado: {kmeansPred}')
    kmeansCentroidMatch=kmeansModel.cluster_centers_[kmeansNewInputPred]
    print(kmeansCentroidMatch)
    print(f'KMEANS - horas previstas para faltar: {kmeansCentroidMatch[0][18]*120}')

    print(f'GMM - indice do cluster encaixado: {gmmModel.predict(inferenciaPadrao)}')
    gmmCentroidMatch = gmmModel.means_[gmmModel.predict(inferenciaPadrao)]
    print(f'GMM - horas previstas para faltar: {gmmCentroidMatch[0][18]*120}')




if __name__ == '__main__':
    main()