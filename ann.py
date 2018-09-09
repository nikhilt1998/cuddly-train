# Artificial Neural Networks (ANN)

#Install Tensorflow
#Install Keras

# Part -1 - Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]                      # To avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part - 2- Now lets make the ANN!

# Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim =11))                      # no. of input layers = no. of independent vairiale here 11
                                             # no. of output layer = when binary outcome then 1

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))                                             
                                              
# Rectifier(relu) activation function is used add hidden layer whereas for output layer Sigmoid fn is used

# Adding the Output Layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid')) 

# Compile our Artificial Neural Network(ANN) i.e. applying schostic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part -3 - Making the predictions and Evaluating the model
   
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)             # to make result binary

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


