from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Importing the parent: DataPreprocessing class from data_preprocess.py
from src.data_preprocess import DataPreprocessing 


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)
        #self.accuracy = None  # Initialize accuracy attribute

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier = DecisionTreeClassifier()
        #ann classifier

        #Train the model
        DT_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = DT_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return DT_classifier
    
    def ann(self, X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=200):
        #ANN_classifier = MLPClassifier(hidden_layer_sizes=(100,),)
        ANN_classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            learning_rate_init=learning_rate_init, 
            max_iter=max_iter
        )
        #ANN_predicted = ANN_classifier.predict(X_test)

         #Create mlp model
        ANN_classifier = MLPClassifier()

        #Train the model
        ANN_classifier.fit(X_train, y_train)

        #Test the model
        ANN_predicted = ANN_classifier.predict(X_test)

        

        error = 0
        for i in range(len(y_test)):
            error += np.sum(ANN_predicted != y_test)

            total_accuracy = 1 - (error / len(y_test))

        self.accuracy = accuracy_score(y_test, ANN_predicted)

        return ANN_classifier