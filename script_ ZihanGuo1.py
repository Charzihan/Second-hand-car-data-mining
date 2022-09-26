import pandas as pd
from apyori import apriori
from sklearn import cluster
import matplotlib.pyplot as plt


# Read the data set
data = pd.read_csv('car.data', sep=',')
data.columns = ['Buying_price', 'Maintenance', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Evaluation']

# Association analysis

# Prepare the data, co the the rules are more clearly identified
associate_set = data.copy()
associate_set['Maintenance'] = associate_set['Maintenance'].map({'vhigh': 'vhigh_main', 'high': 'high_main',
                                                                 'med': 'med_main', 'low': 'low_main'})
associate_set['Doors'] = associate_set['Doors'].map({'2': '2doors', '3': '3doors', '4': '4doors',
                                                     '5more': '5more_doors'})
associate_set['Persons'] = associate_set['Persons'].map({'2': '2persons', '4': '4persons', 'more': '5more_persons'})
associate_set['Lug_boot'] = associate_set['Lug_boot'].map({'small': 'small_lug_boot', 'med': 'med_lug_boot',
                                                           'big': 'high_lug_boot'})
associate_set['Safety'] = associate_set['Safety'].map({'low': 'low_safety', 'med': 'med_safety', 'high': 'high_safety'})

# Prepare the evaluation data into a list each
car_eval = []
for i in range(associate_set.shape[0]):
    car_eval.append(associate_set.iloc[i].dropna().tolist())

# Use the apriori function to create rules for the data set
rules = apriori(car_eval, min_support=0.22, min_confidence=0.6)

# Print out all the rules
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
          'Support:', rule.support, 'Confidence:', rule.ordered_statistics[0].confidence)


# K-mean Clustering

# Prepare the data by changing the attributes into numeric value
data['Buying_price'] = data['Buying_price'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
data['Maintenance'] = data['Maintenance'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
data['Doors'] = data['Doors'].map({'2': 2, '3': 3, '4': 4, '5more': 5})
data['Persons'] = data['Persons'].map({'2': 2, '4': 4, 'more': 6})
data['Lug_boot'] = data['Lug_boot'].map({'small': 1, 'med': 2, 'big': 3})
data['Safety'] = data['Safety'].map({'low': 1, 'med': 2, 'high': 3})

# Drop the class attribute
cluster_set = data.drop(['Evaluation'], axis=1)

# Find the best k value by drawing the the SSE vs clusters graph and then finding the elbow
SSE_ls = []
k_val = []
for k in range(1, 7):
    kmeans = cluster.KMeans(k, random_state=14).fit(cluster_set)
    SSE_ls.append(kmeans.inertia_)
    k_val.append(k)
plt.plot(k_val, SSE_ls, 'rv-')
plt.show()

# Best k value is 3 and do the K-mean analysis with this value
kmeans = cluster.KMeans(3, random_state=14).fit(cluster_set)
labels = kmeans.labels_
clusters = pd.DataFrame(labels, index=data['Evaluation'], columns=['Cluster ID'])
print('\nUnacceptable car:\n', clusters.loc[['unacc']].value_counts(normalize=True)*100, '\n')
print('acceptable car:\n', clusters.loc[['acc']].value_counts(normalize=True)*100, '\n')
print('Good car:\n', clusters.loc[['good']].value_counts(normalize=True)*100, '\n')
print('Very Good car:\n', clusters.loc[['vgood']].value_counts(normalize=True)*100, '\n')
