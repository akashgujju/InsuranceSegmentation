#importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:79]
y = dataset.iloc[:,-1]

#summing the columns into single column
temp = dataset.iloc[:,79:-1]
temp = temp.sum(axis=1)
X['Disease_Count'] = temp

#dropping columns with large missing values
p = X.columns[X.isnull().any()]
q = X[p].isnull().sum(axis = 0)
plt.barh(p, q)
plt.xlabel('Count Of Missing Values')
plt.ylabel('Columns')
plt.title('Missing Data Chart')
plt.legend('Missing Values')
plt.legend(loc="best")

#removing uncecessary columns
l = ['Medical_History_10','Medical_History_24','Medical_History_32','Family_Hist_5']
X = X.drop(l,axis=1)

#replacing null values
X = X.replace(np.nan, 0)

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X_transformer = ColumnTransformer(
    transformers=[
        ("Country",        
         OneHotEncoder(), 
         [1]
         )
    ], remainder='passthrough'
)
X = X_transformer.fit_transform(X)
X = pd.DataFrame(X)

#hadling imbalanced data
from collections import Counter
c = Counter(y)
plt.bar(c.keys(),c.values(),color='green')
plt.xlabel('Risk Level')
plt.ylabel('Count Of The Values')
plt.title('Risk Levels vs Count')

#performing smote analysis
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_resample(X, y)

#vizualizing smote results
c = Counter(y)
plt.bar(c.keys(),c.values(),color='green')
plt.xlabel('Risk Level')
plt.ylabel('Count Of The Values')
plt.title('Risk Levels vs Count')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K 
y_binary = to_categorical(y_train)
y_binary = y_binary[:,1:]
K.clear_session()
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'tanh', input_dim = 93))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'tanh'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_binary, batch_size = 32, nb_epoch = 20)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.model_selection import cross_val_score
cross_val_score(classifier,X_train,y_train,cv=10)