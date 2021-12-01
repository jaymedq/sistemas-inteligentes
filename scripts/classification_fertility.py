# Classificadores
# Primeira parte será obter modelo classificador com a base Fertility
# 1. Balanceamento dos dados
# 2. Obter modelo (treinar)
# 3. Avaliar a acuracia do modelo
##################################
# EXEMPLO COM O ARQUIVO FERTILITY
##################################

# 1, Avaliar se as classes estão balanceadas
import pandas as pd
dados = pd.read_csv('database/fertility-Diagnosis.csv')
print("#Frequencia das classes (atributo Output)")
print(dados.Output.value_counts())
#Balancear dados
from imblearn.over_sampling import SMOTE

# a) Segmentar os dados em atributos e classes
dados.classes = dados['Output'] # Somente a coluna output
dados.atributos = dados.drop(columns="Output")

# b) Construir um objeto a partir do SMOTE e executar o método fit_resample
resampler = SMOTE()
#Executar o balanceamento
dados.atributos_b, dados.classes_b = resampler.fit_resample(dados.atributos, dados.classes)

#Verificar a frequencia das classes balanceadas
print("##### FREQUENCIA DAS CLASSES APÓS BALANCEAMENTO ###")
from collections import Counter
class_count = Counter(dados.classes_b)
print(class_count)

# d) Integrar os dados em um objeto data frame
dados.atributos_b = pd.DataFrame(dados.atributos_b)
dados.classes_b = pd.DataFrame(dados.classes_b)

# Resolver os rotulos das colunas


dados.atributos_b.columns = dados.columns[:-1]
dados.classes_b.columns = [dados.columns[-1]]
dados_b = dados.atributos_b.join(dados.classes_b,how='left')
print(type(dados.atributos_b))
print(type(dados.classes_b))
print(dados.atributos_b)
print(dados.classes_b)
print(dados_b)
print(type(dados_b))


#TREINAMENTO E TESTES COM USO DO CROSS VALIDATION
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#2.4(indutor) CONSTRUIR UM OBJETO A PARTIR DO DECISIOTREECLASSIFIER
tree = DecisionTreeClassifier()
rf = RandomForestClassifier(random_state = 42)

#2.5 TREINAR O MODELO DE ML
fertility_tree_cross = tree.fit(dados.atributos_b, dados.classes_b)
fertility_rf = rf.fit(dados.atributos_b, dados.classes_b)

class_predict = fertility_rf.predict(Atr_test)

#Testar o modelo CROSS VALIDATION
from sklearn.model_selection import cross_validate
scoring = ['precision_macro','recall_macro']
score_cross = cross_validate(tree, dados.atributos_b, dados.classes_b, cv=10, scoring = scoring)
print('Matriz de sensibilidade: ', score_cross['test_precision_macro'])
print('Matriz de especificidade: ', score_cross['test_recall_macro'])
#Métricas finais: calcular as medidas

print('Sensibilidade: ', score_cross['test_precision_macro'].mean())
print('Especificidade: ', score_cross['test_recall_macro'].mean())

from pickle import dump
dump(fertility_tree_cross, open('fertility_tree_cross_model.pkl', 'wb'))



