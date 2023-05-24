#Uvodjenje par bioblioteka koje su nam podrebne za citanje podataka
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

skup=pd.read_csv('FullNormal.csv')
skup.head()

#Odvajanje atributa i klasa
x=skup.iloc[:,0:11].values
y=skup.iloc[:,11].values

X_obucavajuci, X_testirajuci, Y_obucavajuci, Y_testirajuci = train_test_split(x, y, test_size=0.33, random_state=1)
print(X_obucavajuci.shape,X_testirajuci.shape,Y_obucavajuci.shape,Y_testirajuci.shape )

n_features = X_obucavajuci.shape[1]
print(n_features)

import tensorflow as tf
import tensorflow.keras
tf.__version__

from tensorflow.keras import Sequential #Koristimo sekvencijalni model
from tensorflow.keras.layers import Dense #Biblioteka za skrivene slojeve

model = Sequential()
#Definisanje modela (broj skrivenih slojeva i neurona)
model.add(Dense(11, activation='relu', kernel_initializer='uniform', input_shape=(n_features,)))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

#Kompajliranje
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_obucavajuci, Y_obucavajuci, epochs=2000, batch_size=32, verbose=0)

#Evаulacija modela
loss, acc = model.evaluate(X_testirajuci, Y_testirajuci, verbose=0)
print('Tacnost modela je: %.3f' % acc)

#Predikcija nad podacima koji nisu u data setu
warnings.filterwarnings("ignore")
model.save('HeartFailurePrediction.model')
new_model = tensorflow.keras.models.load_model('HeartFailurePrediction.model')
P1=[0.40,0,0,0.7,0.479270315091211,0,0,0.788732394366197,0,0.295454545454545,1]
P2=[0.77,0,0.66,0.625,0.504145936981758,0,1,0.71830985915493,1,0.295454545454545,1]

predictions = new_model.predict([P1,P2])
print(predictions)