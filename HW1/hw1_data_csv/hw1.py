
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from NaiveBayes import StudenttNB
get_ipython().magic(u'matplotlib inline')


def calculateAccuracy(matrix):
    matrix = matrix.astype(float)
    return np.trace(matrix)/np.sum(matrix)
    
def showImage(Q,vector):
    image = Q.dot(vector).reshape(28,28)
    plt.figure()
    plt.imshow(image)


def missClassifyIndex(n,Ytest,Ypred): 
    Ytest_pred = pd.concat([Ytest,Ypred],axis=1)
    Ytest_pred.columns = ['Ytest','Ypred']
    Ytest_pred['missed'] = abs(Ytest_pred['Ytest'] - Ytest_pred['Ypred'])
    return Ytest_pred[Ytest_pred["missed"] == 1].head(n=n).index.tolist()

def ambiguousPredictions(n,Yproba):
    Yproba["diff"] = abs(Yproba[0] - Yproba[1])
    return Yproba.sort(["diff"],ascending=1).head(n=n).index.tolist()
    
def getImages(indexes,Xtest,Yproba):
    images = []
    proba = []
    for i in range(len(indexes)):
        idx = indexes[i]
        images.append(Xtest.iloc[idx])
        proba.append(Yproba.iloc[idx])
    return images, proba

def main():
    #Load Data
    Xtrain = pd.read_csv("./hw1_data_csv/Xtrain.csv",header=None)
    Xtest = pd.read_csv("./hw1_data_csv/Xtest.csv",header=None)
    Ytrain = pd.read_csv("./hw1_data_csv/Ytrain.csv",header=None)
    Ytest = pd.read_csv("./hw1_data_csv/Ytest.csv",header=None)
    Q =  pd.read_csv("./hw1_data_csv/Q.csv",header=None)
    
    #Load Model
    model = StudenttNB(Xtrain,Ytrain)
    
    #train Model
    model.fit()
    
    #predictions
    Ypred = pd.DataFrame(model.predict(Xtest))
    
    #probabilities
    Yproba = pd.DataFrame(model.predict_proba(Xtest))
    
    #confusion matrix
    conf_matrix = confusion_matrix(Ytest,Ypred)
    
    #accuracy 
    accuracy = calculateAccuracy(conf_matrix)
    
    #miss classified images
    mis_images, mis_proba = getImages(missClassifyIndex(3,Ytest,Ypred),Xtest,Yproba)
    
    #ambigous images
    amb_images, amb_proba = getImages(ambiguousPredictions(3,Yproba),Xtest,Yproba)
    
    print "Confusion Matrix: "
    print conf_matrix
    print "Accuracy: ", accuracy
    for i in range(len(mis_images)):
        print "Image: " , i
        print mis_proba[i]
        showImage(Q,mis_images[i])
    
    for j in range(len(amb_images)):
        print "Image: ", j
        print amb_proba[j]
        showImage(Q,amb_images[j])
        
if __name__ == "__main__":
    main()


# In[ ]:



