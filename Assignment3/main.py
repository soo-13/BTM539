from sklearn import tree, metrics, svm
from sklearn.cluster import KMeans 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os 
import pandas as pd


df = pd.read_csv(os.path.join("Project1",  "data", "online_retail_II_preprocessed.csv")) # read the csv file

### measure variables - frequency, and monetary value
rec = df.groupby(['Customer ID']).mean()['Recency'].reset_index()
freq = df.groupby(['Customer ID']).count()['Invoice'].reset_index() # frequency number of purchases since the first purchase 
freq.rename(columns = {'Invoice' : 'Frequency'}, inplace = True)
# monetary value average spending per order
df['Spending'] = df['Price']*df['Quantity']
mon = df.groupby(['Customer ID']).mean()['Spending'].reset_index() # monetary value average spending per order
mon.rename(columns = {'Spending' : 'MonetaryValue'}, inplace = True)
# merge rfm information and merge to df
rfm = rec.merge(freq, on='Customer ID')
rfm = rfm.merge(mon, on='Customer ID')

### input for kmeans clustering
rfm_input = rfm.get(['Recency', 'Frequency', 'MonetaryValue'])
scaler = StandardScaler()  
rfm_input = scaler.fit_transform(rfm_input)

### creating label (5 groups of clustering label)
kmeans = KMeans(n_clusters=8, n_init='auto', random_state=0) # clustering to eight clusters
kmeans.fit(rfm_input)
rfm.insert(loc=1, column='Cluster_Raw', value=kmeans.labels_)
print("Number of customers in each cluster")
print(rfm['Cluster_Raw'].value_counts()) # check the clustering result
# Renumber clusters so that the largest cluster is Cluster1 and the smallest cluster is Cluster5 (merge the smallest 4 clusters into one)
renumber = {0:1, 1:5, 2:3, 3:5, 4:5, 5:2, 6:4, 7:5}
rfm['Cluster'] = rfm['Cluster_Raw']
for i in range(8):
    rfm.loc[rfm['Cluster_Raw']==i, 'Cluster'] = renumber[i]
rfm.drop('Cluster_Raw', axis=1, inplace=True)
print(rfm['Cluster'].value_counts())

### split train-test data
X_train, X_test, y_train, y_test = train_test_split(rfm.get(['Recency', 'Frequency', 'MonetaryValue']), rfm.get(['Cluster']).squeeze(), shuffle = True, random_state = 1234)

### fit and predict 
#decision Tree
dt = tree.DecisionTreeClassifier().fit(X_train, y_train) 
dt_pred = dt.predict(X_test)
# Random Forest
rfc = RandomForestClassifier().fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
# SVM
svmt = svm.SVC().fit(X_train, y_train)  
svm_pred = svmt.predict(X_test)
# K-nearest neighbors
knn = KNeighborsClassifier().fit(X_train, y_train)
knn_pred = knn.predict(X_test)

### performance evaluation
y_preds = [dt_pred, rfc_pred, svm_pred, knn_pred]
models = ['Decision Tree', 'Random Forest', 'SVM', 'KNN']
for i in range(4):
    print(models[i])
    print(metrics.classification_report(y_test, y_preds[i], digits=3))

    
    