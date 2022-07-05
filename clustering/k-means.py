
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



WEBSITE_NAMES = []

# "/Users/ronnatarajan/Desktop/WebBench-Practice/testing/web_bench_stats_random_100.json"
# **** name of true testing file ***** #
# web_bench_stats_random_100.json
def loadJSON(url):
    df = open(url, "r")

    #JSON file loaded into manipulatable variable
    df = json.loads(df.read())
    return df

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
    # 2d array to track the feature list ( JS API calls) of each website
    features = []
    #loop through JSON file again and build 2D Feature List
    for site_name in df:
        WEBSITE_NAMES.append(site_name)

        #array to save each website's features
        #this will be added to the features array to build the 2d array at the end of each loop
        arr = []


        #kick out of iteration if information about website is lacking
        if ('phase_timings' not in df[site_name] or 'mem' not in df[site_name] or 'apis' not in df[site_name] or 'mem_js' not in df[site_name] or 'event_class_stats_summary' not in df[site_name]):
            continue

        # ----------------------------------------------------------- #
        # ***                    get api calls                    *** #
        # ----------------------------------------------------------- #

        for key in apis:
            if key in df[site_name]['apis']:
                arr.append(df[site_name]['apis'][key])
            else:
                arr.append(0)

        #get memory usage for JS
        arr.append(df[site_name]['mem_js'])

        # ----------------------------------------------------------- #
        # ***               Number of Browser Events              *** #
        # ----------------------------------------------------------- #

        for val in browser_events:
            if val in df[site_name]['event_class_stats_summary']:

                arr.append(df[site_name]['event_class_stats_summary'][val]['num_events'])
            else:
                arr.append(0)

        # ----------------------------------------------------------- #
        # ***     Get timing for CPU usage analysis               *** #
        # ----------------------------------------------------------- #
        #Get timing for CPU usage analysis
        val = df[site_name]['phase_timings']
        for key in val:

            arr.append(df[site_name]['phase_timings'][key][0])


        # ----------------------------------------------------------- #
        # *** CALCULATE THE NUMBER OF SERVERS CONTACTED BY WEBSITE *** #
        # ----------------------------------------------------------- #

        site_file = site_name.replace("/","_").replace(".","__") + ".json"
        print(site_file)
        file_path = "/Users/ronnatarajan/Desktop/WebBench-Practice/testing/" + site_file
        browsertime_json = open(file_path, "r")

        #JSON file loaded into manipulatable variable
        browsertime_json = json.loads(browsertime_json.read())
        if "network-requests" not in browsertime_json or "details" not in browsertime_json["network-requests"] or "items" not in browsertime_json["network-requests"]["details"] or browsertime_json["network-requests"]["details"]["items"] == None:
            continue

        items = browsertime_json["network-requests"]["details"]["items"]

        hosts = []
        for item in items:
            o = urlparse(item["url"])
            if o.hostname not in hosts:
                hosts.append(o.hostname)
        arr.append(len(hosts))

        # ----------------------------------------------------------- #
        # ***                    Get RAM usage                    *** #
        # ----------------------------------------------------------- #

        arr.append(df[site_name]["mem"])
        features.append(arr)
    return features
def trainGaussian(features, kstart, kend, kstep):
    #list used to interpret ideal number of clusters
    error_list = []


    #test different cluster numbers and save them in the error list
    k_list = range(kstart,kend, kstep)
    for k in k_list:
        cluster = GaussianMixture(n_components=k, random_state=0).fit(features)
        pred = cluster.predict(features)

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
        error_list.append(metrics.silhouette_score(features, cluster.labels_))

    return error_list, copy_list

def final_plot_Kmeans(num_clusters, features):

    clusters = KMeans(n_clusters= num_clusters, random_state=0).fit(features)

    with open('../KMEANS_CLUSTER_CSV/clusters.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        header = ['predicted_cluster', 'webname']
        writer.writerow(header)
        for i in range(100):
            predicted_cluster = clusters.predict([features[i]])
            str_line = [str(predicted_cluster),  WEBSITE_NAMES[i]]

            # write a row to the csv file
            writer.writerow(str_line)




def trainDBSCAN(features, kstart, kend, kstep, mstart, mend, mstep):
    epsilon = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1,1.25,1.5,1.75, 2,2.25,2.5,2.75, 3,3.25,3.5,3.75, 4]
    min_samples = [1,2,3,4,5,6,7,8,9, 10,15,20,25]


    sil_avg = {}
    max_value = [0,0,0,0]

    for i in range(len(epsilon)):
        for j in range(len(min_samples)):

            db = DBSCAN(min_samples = min_samples[j], eps =epsilon[i]).fit(features)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_


            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            # if np.unique(labels) <= 1:
            if len(set(labels.tolist())) <= 1:
                continue

            silhouette_avg = metrics.silhouette_score(features, labels)
            if silhouette_avg > max_value[3]:
                max_value=(epsilon[i], min_samples[j], n_clusters_, silhouette_avg)
            sil_avg[repr([i,j])] = silhouette_avg
    return sil_avg, epsilon, min_samples



def plotDBSCAN(error_list, k_list, m_list):

#    plt.plot(k_list, error_list, label="m")
    for val in m_list:
        error_listM = []
        x_list = []
        for row in error_list:
            list_row = eval(row)
            if list_row[1] == val:
                error_listM.append(error_list[row])
                x_list.append(list_row[0])



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

def main():
    df = loadJSON("/Users/ronnatarajan/Desktop/WebBench-Practice/testing/web_bench_stats_random_100.json")
    apis, browser_events = gatherUsedAPIs(df)
    features = buildFeatureSet(df, apis, browser_events)
    # error_list, k_list = trainKMeans(features,1,10,1)
    #
    # k_means_plot(error_list, k_list)






    # final_plot_Kmeans(4,features)




main()
