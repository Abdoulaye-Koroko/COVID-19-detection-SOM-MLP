# Importation des librairies nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

#Importation des données
data=pd.read_csv("word_countries.csv")
print(data.shape)
data.head()

#Attribution d'identifiants aux différents pays du dataset
ID=[i for i in range(155015,155015+len(data))]
coresp={k:v for k,v in zip(ID,data["Country"])}
Country=list(coresp.values())
data.insert(2, "ID", ID)
data=data.drop(labels=["Country","Region"],axis=1)
data.head()
print(data.dtypes)

#Convertion des valeurs nulériques string en float
def convert(x):
    
    return float(x.replace(',','.'))

print(convert("6,5"))
columns=data.loc[:, data.dtypes == object]
columns=columns.columns
columns=list(columns)
data[columns] = data[columns].astype(str)
for a in columns:
    data[a]=data[a].apply(lambda x: convert(x))
print(data.dtypes)
data.dtypes

#Affichage des valeurs manquantes de chaque colonne
msno.bar(data)

#Traitement des valeurs manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X=imputer.fit_transform(data)
print(X.shape)

#Normalisation des données
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X)

#Construction et entraînement de la SOM
som = MiniSom(x = 5, y = 4, input_len = 19, sigma = 0.8, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 1000)

#Affichage de la carte de sortie
bone()
pcolor(som.distance_map().T)
colorbar()

#Pays derrières chaque neurone de sortie
mapps = som.win_map(X)
keys=list(mapps.keys())
values=list(mapps.values())
values=[len(v) for v in values ]
result={key:value for key,value in zip(keys,values)}
print(result)
bone()
pcolor(som.distance_map().T)
colorbar()
dictio={}
j=0
for k,v in mapps.items():
    j=j+1
    plt.text(k[0]+0.5,k[1]+0.5,"G"+str(j),color="red")
    liste=[]
    for i in range(len(v)):
        country=v[i]
        country=country.reshape(1, -1)
        country=scaler.inverse_transform(country)
        country=coresp[country[0][0]]
        liste.append(country[:4])
    dictio["Groupe"+str(j)]=liste
print(dictio)
df=pd.DataFrame.from_dict(dictio, orient='index')
df.head()