import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import time

def load_data(filepath):
    list_dicts = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            list_dicts.append(row)
    return list_dicts

def calc_features(row):
    arr = np.zeros(6, dtype=np.float64)
    arr[0] = row['Population']
    arr[1] = row['Net migration']
    arr[2] = row['GDP ($ per capita)']
    arr[3] = row['Literacy (%)']
    arr[4] = row['Phones (per 1000)']
    arr[5] = row['Infant mortality (per 1000 births)']
    return arr

def hac(features):
    # clusters is dict with keys being "indexes" and values being rows (countries)
    clusters = {i:[features[i]] for i in range(len(features))}
    # distance matrix is twice the size to accomodate original matrix distances and merged matrix distances
    distance_matrix = np.zeros((2 * len(features), 2 * len(features)))
    # initializing ans (return) and used_clusters
    ans = np.zeros((len(features) - 1, 4))
    used_clusters = set()
    counter = 0
    #initializing distance_matrix with original matrix distances
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distance_matrix[i][j] = np.linalg.norm(clusters[i][0] - clusters[j][0])

    # iterating through each row
    for row in range(len(features) - 1):
        # 0 - min distance found b/w clusters, 1 - index of first cluster, 2 - index of 2nd cluster
        mins = (float('inf'),float('inf'),float('inf'))
        # going to len(features) + row b/c we want to account for original and new clusters
        for i in range(len(features) + row):
            # comparing every other cluster to the current one (except ones that come before it)
            for j in range(i + 1, len(features) + row):
                # ensuring we don't recalculate distances
                if j not in used_clusters and i not in used_clusters:
                    curr_mins = (distance_matrix[i][j], i, j)
                    counter += 1
                    #print(counter)
                    #curr_mins = (np.linalg.norm(clusters[i][0] - clusters[j][0]), i, j)
                    mins = min(mins, curr_mins)

        # i and j will represent the indexes of the clusters being merged
        i,j = mins[1],mins[2] 

        # new_cluster index is n (length of features list) + current row 
        merged_cluster = len(features)+row
        clusters.update({merged_cluster:clusters[i] + clusters[j]})

        # updates the distances from each cluster to the new cluster
        for x in clusters:        
            distance_matrix[x][merged_cluster] = max(distance_matrix[min(i,x)][max(i,x)],
                                                      distance_matrix[min(j,x)][max(j,x)])
        
        used_clusters.add(i)
        used_clusters.add(j)
        ans[row][0] = i
        ans[row][1] = j 
        ans[row][2] = mins[0]
        ans[row][3] = len(clusters[merged_cluster])
    print(counter)
    return ans

def fig_hac(Z, names):
    fig = plt.figure()
    hierarchy.dendrogram(Z,labels=names)
    plt.xticks(rotation = 'vertical')
    fig.tight_layout()
    return fig

def normalize_features(features):
    averages = [0,0,0,0,0,0]
    sds = [[],[],[],[],[],[]]
    for row in features:
        for i in range(6):
            averages[i] += row[i]
            sds[i].append(row[i])

    real_sds = []
    for j in range(6):
        averages[j] /= len(features)
        real_sds.append(np.std(sds[j]))
    
    ans_list = []
    for r in features:
        ans_list.append(r)
        for i in range(6):
            r[i] = (r[i] - averages[i]) / real_sds[i]
    return ans_list





if __name__=="__main__":
    start_time = time.time()
    country_names = []
    with open('countries.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        country_names = [row[1] for row in csv_reader]
    print(len(country_names))
    list_dict = load_data("countries.csv")
    list = []
    for i in range(len(country_names)):
        list.append(calc_features(list_dict[i]))
    normalize_features(list)
    Z = hac(list)
    print(len(Z))
    f = fig_hac(Z, country_names)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()
    
    
    