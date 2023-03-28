from sklearn.cluster import KMeans 
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
mon =  df.groupby(['Customer ID']).mean()['Spending'].reset_index() # monetary value average spending per order
mon.rename(columns = {'Spending' : 'MonetaryValue'}, inplace = True)
# merge rfm information and merge to df
rfm = rec.merge(freq, on='Customer ID')
rfm = rfm.merge(mon, on='Customer ID')
df = df.merge(freq, on='Customer ID')
df = df.merge(mon, on='Customer ID')

### input for kmeans clustering
cID = rfm['Customer ID']
rfm = rfm.get(['Recency', 'Frequency', 'MonetaryValue'])

### elbow method to determine the optimal number of clusters
sse = []
for k in range(2, 28):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    kmeans.fit(rfm)
    sse.append(kmeans.inertia_)
    
### visualize results
plt.plot(range(2, 28), sse)
plt.xticks(range(2, 28))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

### k-means clustering with seven clusters 
kmeans = KMeans(n_clusters=7, n_init='auto', random_state=0)
kmeans.fit(rfm)
clstinfo = cID.to_frame()
clstinfo.insert(loc=1, column='Cluster_Raw', value=kmeans.labels_)
print("Number of customers in each cluster")
print(clstinfo['Cluster_Raw'].value_counts()) 
df = df.merge(clstinfo, on='Customer ID')
# Renumber clusters so that the largest cluster is Cluster1 and the smallest cluster is Cluster7
renumber = {0:1, 1:7, 2:4, 3:5, 4:2, 5:6, 6:3}
df['Cluster'] = df['Cluster_Raw']
for i in range(7):
    df.loc[df['Cluster_Raw']==i, 'Cluster'] = renumber[i]
df.drop('Cluster_Raw', axis=1, inplace=True)
    
### describe group characteristics
pd.options.display.float_format = '{:.2f}'.format
# RFM variables
bycustomer = df.groupby('Customer ID').max().reset_index()
print(bycustomer.groupby('Cluster').mean().get(['Recency', 'Frequency', 'MonetaryValue']))
for i in range(7):
    print("Descriptive statistics of RFM variables: Cluster{}".format(i+1))
    print(bycustomer[bycustomer['Cluster']==i+1].describe().get(['Recency', 'Frequency', 'MonetaryValue']))  
# visualization scatter plot
plt.clf()
scatter = plt.scatter(bycustomer['Recency'], bycustomer['Frequency'], s=bycustomer['MonetaryValue'], c=bycustomer['Cluster'], cmap='Accent', alpha=0.8)
plt.legend(handles=scatter.legend_elements()[0], labels=range(1,8))
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.show()
