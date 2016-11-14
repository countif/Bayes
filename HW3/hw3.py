import pandas as pd
import numpy as np
from VI import VI

X_set1 = pd.read_csv("./data_csv/X_set1.csv")
X_set2 = pd.read_csv("./data_csv/X_set2.csv")
X_set3 = pd.read_csv("./data_csv/X_set3.csv")
y_set1 = pd.read_csv("./data_csv/y_set1.csv")
y_set2 = pd.read_csv("./data_csv/y_set2.csv")
y_set3 = pd.read_csv("./data_csv/y_set3.csv")
z_set1 = pd.read_csv("./data_csv/z_set1.csv")
z_set2 = pd.read_csv("./data_csv/z_set2.csv")
z_set3 = pd.read_csv("./data_csv/z_set3.csv")

X = np.asarray(X_set1)
y = np.asarray(y_set1)

model = VI(X,y,n_iter=100)
model.iterate()
a,b,e,f, mu, sgm = model.a, model.b,model.e,model.f, model.mu,model.sgm
#pd.Series(model.ViLikelihoods).plot(kind="line")
#pd.Series(1/(a/b)).plot(kind="line")
y_pred = X.dot(mu)
z = np.asarray(z)
