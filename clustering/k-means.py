
# df =  open("/Users/ronnatarajan/Desktop/WebBench-Practice/testing/web_bench_stats_random_100.json", "r")
import json
import site
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
# from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import csv
from urllib.parse import urlparse
import sys
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import datasets
import random
from sklearn.metrics.pairwise import pairwise_distances
import math
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt


WEBSITE_NAMES = []

# "/Users/ronnatarajan/Desktop/WebBench-Practice/testing/web_bench_stats_random_100.json"
# **** name of true testing file ***** #
# web_bench_stats_random_100.json
def loadJSON(url):
    df = open(url, "r")

    #JSON file loaded into manipulatable variable
    df = json.loads(df.read())
    return df
def epsilon(features):
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.xlim([40,50])
    plt.ylim([0.3,0.8])
    plt.savefig("../clustering/epsilon.png")

def gatherUsedAPIs(df):
    #list of used APIs within web_bench_stats_random_100 JSON file
    apis = []
    browser_events = []

    #loop through JSON file and gather all the API's used by the gathered websites
    for site_name in df:
        if ('apis' in df[site_name]):
            for key in df[site_name]['apis']:
                if key not in apis:
                    apis.append(key)
        if ('event_class_stats_summary' in df[site_name]):
            for key in df[site_name]['event_class_stats_summary']:
                if key not in browser_events:
                    browser_events.append(key)


    return apis, browser_events

def buildFeatureSet(df, apis, browser_events):
    skipped = 0
    total = 0



    with open('../clustering/websites.csv', 'w') as f:


        # create the csv writer
        writer = csv.writer(f)
        header = ['Webname', 'Apis', 'RAM-Metric', 'Browser Events', 'Browser Events','Browser Events','Browser Events','CPU' ]
        writer.writerow(header)

    # 2d array to track the feature list ( JS API calls) of each website
    features = []
    #loop through JSON file again and build 2D Feature List
    for site_name in df:
        total += 1

        WEBSITE_NAMES.append(site_name)

        #array to save each website's features
        #this will be added to the features array to build the 2d array at the end of each loop
        arr = []


        #kick out of iteration if information about website is lacking
        if ('phase_timings' not in df[site_name] or 'mem' not in df[site_name] or 'apis' not in df[site_name] or 'mem_js' not in df[site_name] or 'event_class_stats_summary' not in df[site_name]):
            continue

        #load Lighthouse report
        site_file = site_name.replace("/","_").replace(".","__") + ".json"
        file_path = "/Users/ronnatarajan/Desktop/WebBench/testing/" + site_file

        browsertime_json = open(file_path, "r")

        #JSON file loaded into manipulatable variable
        try:
            browsertime_json = json.loads(browsertime_json.read())
        except:
            continue


        #if lighthouse report is not useful, skip this website
        if  "audits" not in browsertime_json or "network-requests" not in browsertime_json["audits"] or "details" not in browsertime_json["audits"]["network-requests"] or "items" not in browsertime_json["audits"]["network-requests"]["details"] or len(browsertime_json["audits"]["network-requests"]["details"]["items"]) == 0:
            continue

        # ----------------------------------------------------------- #
        # ***                    get api calls                    *** #
        # ----------------------------------------------------------- #
        num_apis = 0
        for key in apis:
            if key in df[site_name]['apis']:
                num_apis += df[site_name]['apis'][key]
        arr.append(num_apis)
        # ----------------------------------------------------------- #
        # ***                     Memory                          *** #
        # ----------------------------------------------------------- #
        #get memory usage for JS
        arr.append(df[site_name]['mem_js'])
        arr.append(df[site_name]['mem'] - df[site_name]['mem_js'])

        # ----------------------------------------------------------- #
        # ***               Number of Browser Events              *** #
        # ----------------------------------------------------------- #
        num_browser=0
        for val in browser_events:
            if val in df[site_name]['event_class_stats_summary']:
                num_browser += df[site_name]['event_class_stats_summary'][val]['num_events']
        arr.append(num_browser)
        # ----------------------------------------------------------- #
        # ***     Get timing for CPU usage analysis               *** #
        # ----------------------------------------------------------- #
        #Get timing for CPU usage analysis
        num = []
        val = df[site_name]['phase_timings']
        # invalid = False

        for key in val:
            num.append(val[key][0])
            arr.append(df[site_name]['phase_timings'][key][0])
        if len(num) == 0 or not len(num) == 4:
            continue
        # if invalid:
        #     continue



        features.append(arr)


    print(len(features))
    print(len(features[0]))


    for col in range(len(features[0])):
        arr = []
        for row in range(len(features)):
            arr.append(features[row][col])
        for row in range(len(features)):
            features[row][col] = features[row][col] / max(arr)

    with open('../clustering/websites.csv', 'a') as f:


        # create the csv writer
        writer = csv.writer(f)

        str_line = [site_name]
        string = site_name
        for i in features:
            writer.writerow(i)
    return features
def trainGaussian(features, kstart, kend, kstep):
    #list used to interpret ideal number of clusters
    error_list = []


    #test different cluster numbers and save them in the error list
    k_list = [x for x in range(kstart,kend, kstep)]
    for k in k_list:
        cluster = GaussianMixture(n_components=k, random_state=0).fit(features)
        pred = cluster.predict(features)
        if len(pred) <=1:
            del k_list[k]
            continue
        error_list.append(metrics.silhouette_score(features, pred))

    return error_list, k_list

def plotGaussian(error_list, k_list):
    plt.plot(k_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../GaussianTraining/clusters_analysis.png")

def trainKMeans(features, kstart, kend, kstep):
    #list used to interpret ideal number of clusters
    error_list = []
    copy_list = []


    #test different cluster numbers and save them in the error list
    k_list = list(range(kstart,kend, kstep))
    for k in k_list:
        cluster = KMeans(n_clusters=k, random_state=0).fit(features)
        if len(set(cluster.labels_.tolist())) <= 1:
            continue

        copy_list.append(k_list.index(k))
        error_list.append(cluster.inertia_)
        # error_list.append(metrics.silhouette_score(features, cluster.labels_))

    return error_list, copy_list

def sampling(budget, features, selector):#budget, selector, clusters, features):
    cluster = DBSCAN(eps =1.2, min_samples = 2).fit(features)


    labels = cluster.labels_

    unique_clusters = set(labels)
    #find percent each cluster makes up of the overal number of websites and apply that percent to the budget
    num_web = {}


    try:
        arr = [1,2,3,4]
        d = np.array(arr)
        print(arr[np.where(d < 4)])
    except:
        print("bruh whyyyyyyyyyyyyyyyy")

    npfeatures = np.array(features)
    points = {i: npfeatures[np.where(cluster.labels_ == i)] for i in unique_clusters}
    # print(points[0].tolist())
    points_of_cluster = []
    centroid_of_cluster = []
    for row in points:
        points_of_cluster.append(points[row])
        centroid_of_cluster.append(np.mean(points[row], axis=0))
    for c in unique_clusters:
        num_web[c] = [(float(len(points_of_cluster[c])) / float(len(labels))), labels.tolist().count(c)]
    smaller_web = []

    total = len(unique_clusters)
    curr = 0
    print("total: " + str(total))
    j = 0
    if selector == 'closest to cluster':
        for c in unique_clusters:
            num = num_web[c][0] * budget

            for i in range(round(num)):
                closest, _ = pairwise_distances_argmin_min(centroid_of_cluster, features)
                smaller_web.append(WEBSITE_NAMES[closest[c]])
                del features[closest[c].item()]
            j += 1
        return smaller_web
    elif selector == 'random':
        for c in unique_clusters:
            num = num_web[c][0] * budget
            for i in range(round(num)):
                random_point = random.choice(points_of_cluster[c])
                index = features.index(random_point.tolist())
                smaller_web.append(WEBSITE_NAMES[index])
                del features[index]
        return smaller_web
    elif selector == 'smallest avg distance':


        for c in unique_clusters:
            distances = {}
            sum = 0
            num = num_web[c][0] * budget
            for i in range(round(num)):
                for p in points_of_cluster[c]:
                    for cp in points_of_cluster[c]:
                        if cp.tolist() == p.tolist():
                            continue
                        list = points_of_cluster[c].tolist()
                        pointone = {list.index(p.tolist()) : p}
                        pointTwo = {list.index(cp.tolist()) : cp}
                        sum += math.dist(p.tolist(), cp.tolist())

                        feature_index_P= features.index(p.tolist())
                    distances[feature_index_P] = float(sum) / float((len(points_of_cluster[c])))
                best_web= min(distances)
                smaller_web.append(WEBSITE_NAMES[best_web])
        return smaller_web

#BEST VALUE FOR NUMCLUSTERS WAS CALCULATED TO BE  **** 4 ***
def final_plot_Kmeans(num_clusters,k, a, features):


    with open('../KMEANS_CLUSTER_CSV/clusters.csv', 'w') as f:
        clusters = KMeans(n_clusters= num_clusters, random_state=0).fit(features)

        # create the csv writer
        writer = csv.writer(f)
        header = ['predicted_cluster', 'webname']
        writer.writerow(header)
        for i in range(len(clusters.labels_)):
            predicted_cluster = clusters.predict([features[i]])
            str_line = [str(predicted_cluster),  WEBSITE_NAMES[i]]

            # write a row to the csv file
            writer.writerow(str_line)
    with open('../KMEANS_CLUSTER_CSV/gaussian.csv', 'w') as f:
        cluster = GaussianMixture(n_components=k, random_state=0).fit(features)


        # create the csv writer
        writer = csv.writer(f)
        header = ['predicted_cluster', 'webname']
        writer.writerow(header)
        for i in range(len(clusters.labels_)):
            predicted_cluster = clusters.predict([features[i]])
            str_line = [str(predicted_cluster),  WEBSITE_NAMES[i]]

            # write a row to the csv file
            writer.writerow(str_line)
    with open('../KMEANS_CLUSTER_CSV/agglo.csv', 'w') as f:
        cluster = AgglomerativeClustering(n_clusters=a).fit(features)


        # create the csv writer
        writer = csv.writer(f)
        header = ['predicted_cluster', 'webname']
        writer.writerow(header)
        for i in range(len(clusters.labels_)):
            predicted_cluster = clusters.predict([features[i]])
            str_line = [str(predicted_cluster),  WEBSITE_NAMES[i]]

            # write a row to the csv file
            writer.writerow(str_line)




# def sample_KMEANS(clusters):


def trainDBSCAN(features):
    epsilon = [x/10 for x in range(1,10000,1)]
    min_samples = [y for y in range(2,100, 1)]

    print("Feature Length -------> " + str(len(features)))
    print("Feature Width -------> " + str(len(features[0])))



    sil_avg = {}
    db = DBSCAN(eps=1.2, min_samples = 2).fit(features)
    labels = db.labels_


    sp = DBSCAN(eps=1.2, min_samples = 3).fit(features)
    splabels = sp.labels_
    print( "Num clusters: " + str(len(set(labels.tolist()))))
    print( "Num clusters: " + str(len(set(splabels.tolist()))))

    # with open('../DBSCAN-testing/numclusters.csv', 'w') as f:
    #
    #     # create the csv writer
    #     writer = csv.writer(f)
    #     header = ['Epsilon', 'Minsamples', "Clusters"]
    #     writer.writerow(header)
#     for i in range(len(epsilon)):
#         for j in range(len(min_samples)):
#
#             db = DBSCAN(eps =epsilon[i], min_samples = min_samples[j]).fit(features)
#
#             # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#             # core_samples_mask[db.core_sample_indices_] = True
#
#             labels = db.labels_
#
#             # with open('../DBSCAN-testing/numclusters.csv', 'a') as f:
#             #     # create the csv writer
#             #     writer = csv.writer(f)
#             #
#             #     temp = ""
#             #     str_line = [str(epsilon[i]), str(min_samples[j])]
#             #     for label in labels:
#             #         index = [x[0] for x, value in np.ndenumerate(labels) if value== label]
#             #         str_line.append(str(len(index)))
#             #
#             #
#             #     # write a row to the csv file
#             #     writer.writerow(str_line)
#
#             # if
#
#
#             # if np.unique(labels) <= 1:
#             if len(set(labels.tolist())) <= 1 or len(set(labels.tolist())) == len(features):
#                 continue
#
#             silhouette_avg = metrics.silhouette_score(features, labels)
#
# # silhouette_avg > 0.4 and len(set(labels.tolist())) > 2 and
#             # if not len(set(labels.tolist())) == 50 and not len(set(labels.tolist())) == 1:
#             #     print("epsilon: " + str(epsilon[i]) + "---------- minsamples : " + str(min_samples[j]) + "------ clusters: " + str(len(set(labels.tolist()))) + " ------error: " + str(silhouette_avg))
#             sil_avg[repr([epsilon[i],min_samples[j]])] = silhouette_avg

    # print(len(sil_avg))
    return sil_avg, epsilon, min_samples #Epsilon: 1.2, min_samples: 3    (Seems like 2 clusters are best : 0.73 error val)



def plotDBSCAN(error_list, k_list, m_list):

#    plt.plot(k_list, error_list, label="m")
    for val in m_list:
        error_listM = []
        x_list = []
        numclusters = 0
        for row in error_list:
            list_row = eval(row)
            if list_row[1] == val:
                error_listM.append(error_list[row])
                x_list.append(list_row[0])


        # print("Epsilon len: " + str(len(x_list)))
        # print("Error len: " + str(len(error_listM)))

        fig1, ax1 = plt.subplots()
        ax1.plot(x_list, error_listM, label="m = {m_val}".format(m_val = val))
        ax1.set_title('The Elbow Method Graph')
        ax1.set_xlabel('EPS and min_samples')
        ax1.set_ylabel('Error')
        fig1.savefig("../DBSCAN-testing/clusters_analysis" + str(val)+".png")


def agglo(features, kstart, kend, kstep):
    #list used to interpret ideal number of clusters
    error_list = []


    #test different cluster numbers and save them in the error list
    k_list = list(range(kstart,kend, kstep))
    copy_list = []
    for k in k_list:
        cluster = AgglomerativeClustering(n_clusters=k).fit(features)
        # if np.unique(labels) <= 1:
        if len(set(cluster.labels_.tolist())) <= 1:
            continue
        copy_list.append(k_list.index(k))
        error_list.append(metrics.silhouette_score(features, cluster.labels_))

    return error_list, copy_list

def plotAgglo(error_list, k_list):
    #plot the error levels found in the different tests to be later analyzed



    plt.plot(k_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../agglo-testing/agglo_analysis.png")

def k_means_plot(error_list, k_list):
    #plot the error levels found in the different tests to be later analyzed

    plt.plot(k_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../k-means-testing/clusters_analysis.png")

def comp(features, kstart, kend, kstep):
    #list used to interpret ideal number of clusters
    error_list = []


    #test different cluster numbers and save them in the error list
    k_list = [x for x in range(kstart,kend, kstep)]

    copy_list = []
    for k in k_list:
        try:
            cluster = GaussianMixture(n_components=k, random_state=0).fit(features)
        except:
            continue
        pred = cluster.predict(features)
        if len(pred) <=1:
            continue
        if len(pred) <= 1 or len(pred) < k:
            continue
        error_list.append(cluster.bic(np.array(features)))
        copy_list.append(k)
        # error_list.append(metrics.silhouette_score(features, pred))

    plt.plot(copy_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../clustering/gaussian.png")

    # return error_list, k_list

def main():
    # df = loadJSON("/Users/ronnatarajan/Desktop/WebBench/testing/smaller_webList/web_list.json")
    df = loadJSON("/Users/ronnatarajan/Desktop/WebBench/testing/web_bench_stats_random_100.json")

    apis, browser_events = gatherUsedAPIs(df)
    features = buildFeatureSet(df, apis, browser_events)

    # epsilon(features)

    # cluster = DBSCAN(eps=0.5, min_samples = 4).fit(features)
    #
    # error = metrics.silhouette_score(features, cluster.labels_)
    # print(str(error))

    comp(features, 1, 100, 1)
    # # sampling(features)
    # error_list, k_list= trainKMeans(features, 2, 25, 1)
    # k_means_plot(error_list, k_list)
    #
    # G_error_list, G_k_list = trainGaussian(features, 1, 100, 1)
    # plotGaussian(G_error_list, G_k_list)
    #
    # A_error_list, A_k_list = agglo(features, 1, 100, 1)
    # plotAgglo(A_error_list, A_k_list)
    #
    # sil_avg, epsilon, min_samples = trainDBSCAN(features)

    # list = sampling(5, features, 'smallest avg distance')
    # print(len(list))
    # plotDBSCAN(sil_avg, epsilon, min_samples)
    # error_list, k_list = trainKMeans(features,1,49,1)
    # #
    # k_means_plot(error_list, k_list)
    #
    #
    #
    # final_plot_Kmeans(4,7, 4,features)




main()
