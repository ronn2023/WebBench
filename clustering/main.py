
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


def loadJSON(url):
    '''
            DESCIPRTION
            loads json with website info
            :param:<url, String>: path of json file
            :return: JSON variable
            :rtype: JSON
    '''

    df = open(url, "r")

    #JSON file loaded into manipulatable variable
    df = json.loads(df.read())
    return df


def epsilon(features):
    '''
            DESCIPRTION
            Used to find optimal epsilon value for DBSCAN
            :param:<features, 2D list>: feature list
            :return: none
            :rtype: none
    '''
    #kdistance graph
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)


    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    #restrict x and y axes and plot
    plt.plot(distances)
    plt.xlim([40,50])
    plt.ylim([0.3,0.8])
    plt.savefig("../clustering/epsilon.png")

def gatherUsedAPIs(df):
    '''
            DESCIPRTION
            Build List of API's in JSON
            :param:<df, JSON>: feature list
            :return: apis, browser_events --
            :rtype: touple of lists
    '''


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
    '''
            DESCIPRTION
            Build Feature list
            :param:<df, apis, browser_events; JSON, list, list>: website data JSON, list of known API's and browser events
            :return: 2D feature list
            :rtype: 2D list
    '''


    #commented code below can be used to create a CSV to visualize the featurelist

    # with open('../clustering/websites.csv', 'w') as f:
    #
    #
    #     # create the csv writer
    #     writer = csv.writer(f)
    #     header = ['Webname', 'Apis', 'RAM-Metric', 'Browser Events', 'Browser Events','Browser Events','Browser Events','CPU' ]
    #     writer.writerow(header)


    # 2d array to track the feature list ( JS API calls) of each website
    features = []
    #loop through JSON file again and build 2D Feature List
    for site_name in df:

        #log the website names
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

        #count number of total events
        num_apis = 0
        for key in apis:
            if key in df[site_name]['apis']:
                num_apis += df[site_name]['apis'][key]
        arr.append(num_apis)
        # ----------------------------------------------------------- #
        # ***                     Memory                          *** #
        # ----------------------------------------------------------- #
        #get memory usage for JS and NON js
        arr.append(df[site_name]['mem_js'])
        arr.append(df[site_name]['mem'] - df[site_name]['mem_js'])

        # ----------------------------------------------------------- #
        # ***               Number of Browser Events              *** #
        # ----------------------------------------------------------- #

        #count number of total browser events
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

        for key in val:
            num.append(val[key][0])
            arr.append(df[site_name]['phase_timings'][key][0])

        #ensure that CPU usage analysis data exists
        if len(num) == 0 or not len(num) == 4:
            continue




        features.append(arr)


    print(len(features))
    print(len(features[0]))

    #normalize data: MAX NORMALIZATION
    for col in range(len(features[0])):
        arr = []
        for row in range(len(features)):
            arr.append(features[row][col])
        for row in range(len(features)):
            features[row][col] = features[row][col] / max(arr)

    # commented code to add to visualization file

    # with open('../clustering/websites.csv', 'a') as f:
    #
    #
    #     # create the csv writer
    #     writer = csv.writer(f)
    #
    #     str_line = [site_name]
    #     string = site_name
    #     for i in features:
    #         writer.writerow(i)
    return features
def trainGaussian(features, kstart, kend, kstep):
    '''
            DESCIPRTION
            Train gaussian model (Get info on error with different parameter values)
            :param:<features, kstart, kend, kstep; 2D list, int, int, int>: featurelist, starting model value for n_components,
                                                                            end value for n_componenets, step val
            :return: error_list, k_list
            :rtype: list, list
    '''


    #list used to interpret ideal number of clusters
    error_list = []


    #test different cluster numbers and save them in the error list
    k_list = [x for x in range(kstart,kend, kstep)]
    for k in k_list:
        cluster = GaussianMixture(n_components=k, random_state=0).fit(features)
        pred = cluster.predict(features)
        #ensure that num labels isn't one
        if len(pred) <=1:
            del k_list[k]
            continue
        #use silhouette_avg to analyze clustering
        error_list.append(metrics.silhouette_score(features, pred))

    return error_list, k_list

def plotGaussian(error_list, k_list):
    '''
            DESCIPRTION
            Plot gausian values with their corresponding error
            :param:<error_list, k_list; list, list>: values with gaussian model error and model arguments
            :return: none
            :rtype: none
    '''
    plt.plot(k_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../GaussianTraining/clusters_analysis.png")

def trainKMeans(features, kstart, kend, kstep):
    '''
            DESCIPRTION
            Test different KMEANs parameters
            :param:<features, kstart, kend, kstep; JSON, list, list>: featurelist / end, start, and step vals for model training
            :return: error_list, k_list
            :rtype: list, list
    '''

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

def sampling(budget, features, selector):
    '''
            DESCIPRTION
            Sampling method
            :param:<budget, features, selector; int, 2D list, string>: desired num websites, 2D list of features, selector for sampling techniques
            :return: smaller_web
            :rtype: list
    '''

    # clustering object
    # cluster = DBSCAN(eps =1.2, min_samples = 2).fit(features)
    cluster = KMeans(n_clusters=10, random_state=0).fit(features)
    labels = cluster.labels_

    unique_clusters = set(labels)
    #find percent each cluster makes up of the overal number of websites and apply that percent to the budget
    num_web = {}

    #copy featurelist into np array and bui;d this into a list of points from the clustering
    npfeatures = np.array(features)
    points = {i: npfeatures[np.where(cluster.labels_ == i)] for i in unique_clusters}
    #build 2D lists that track points of each cluster and centroids of each cluster
    #build map that tracks the weightage of each cluster and the number of points in that cluster
    points_of_cluster = []
    centroid_of_cluster = cluster.cluster_centers_

    cpy = []
    for row in points:
        points_of_cluster.append(points[row])
        cpy.append(np.mean(points[row], axis=0))
    for c in unique_clusters:
        num_web[c] = [(float(len(points_of_cluster[c])) / float(len(labels))), labels.tolist().count(c)]


    smaller_web = []
    total = len(unique_clusters)

    #build smaller list using the closest websites to the centroids
    if selector == 'closest to cluster':
        for c in unique_clusters:
            num = num_web[c][0] * budget
            if int(math.ceil(num)) == 0:
                print("incorrect budget, suggested num websites is: " + str(round(num)))
                exit(1)
            #keep selecting websites num times
            for i in range(int(math.ceil(num))):
                closest, _ = pairwise_distances_argmin_min(centroid_of_cluster, features)
                smaller_web.append(WEBSITE_NAMES[closest[c]])
                del features[closest[c].item()]
        return smaller_web
    #build smaller list using random websites from each cluster
    elif selector == 'random':
        for c in unique_clusters:
            num = num_web[c][0] * budget
            if int(math.ceil(num)) == 0:
                print("cannot produce specified number of clusters")
                exit(1)
            #keep selecting websites num times
            copy = points_of_cluster[c]
            print("aajsdjklhaskjhfjkasdjfaklsdfasd")
            print(type(points_of_cluster[c][0]))
            found = []
            for i in range(int(math.ceil(num))):

                #random point
                random_point = random.choices(copy)[0]
                while random_point.tolist() in found:
                    random_point = random.choices(copy)[0]


                index = features.index(random_point.tolist())
                smaller_web.append(WEBSITE_NAMES[index])
                found.append(random_point.tolist())
        #

        return smaller_web
    elif selector == 'smallest avg distance':


        for c in unique_clusters:
            #distances map
            distances = {}
            #var to track sum of each cluster
            sum = 0
            #num websites to select from cluster
            num = num_web[c][0] * budget

            if int(math.ceil(num)) == 0:
                print("incorrect budget")
                exit(1)
            #keep selecting websites num times
            copy = points_of_cluster[c]

            for p in points_of_cluster[c]:
                for cp in points_of_cluster[c]:
                    if cp.tolist() == p.tolist():
                        continue
                    list = points_of_cluster[c].tolist()

                    sum += math.dist(p.tolist(), cp.tolist())

                    feature_index_P= features.index(p.tolist())
                distances[feature_index_P] = float(sum) / float((len(points_of_cluster[c])))
            for i in range(int(math.ceil(num))):
                best_web= min(distances)
                smaller_web.append(WEBSITE_NAMES[best_web])
                del distances[best_web]
        return smaller_web

#BEST VALUE FOR NUMCLUSTERS WAS CALCULATED TO BE  **** 4 ***
def final_plots(num_clusters,k, a, features):
    '''
            DESCIPRTION
            Plot everything
            :param:<num_clusters, k, a , features; int, int, int, int>: parameter vals
            :return: none
            :rtype: none
    '''

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



def agglo(features, kstart, kend, kstep):
    '''
            DESCIPRTION
            Test different AgglomerativeClustering parameters
            :param:<features, kstart, kend, kstep; JSON, list, list>: featurelist / end, start, and step vals for model training
            :return: error_list, k_list
            :rtype: list, list
    '''
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
    '''
        DESCIPRTION
        Plot AgglomerativeClustering values with their corresponding error
        :param:<error_list, k_list; list, list>: values with gaussian model error and model arguments
        :return: none
        :rtype: none
    '''
    #plot the error levels found in the different tests to be later analyzed



    plt.plot(k_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../agglo-testing/agglo_analysis.png")

def k_means_plot(error_list, k_list):
    '''
        DESCIPRTION
        Plot k means values with their corresponding error
        :param:<error_list, k_list; list, list>: values with gaussian model error and model arguments
        :return: none
        :rtype: none
    '''
    #plot the error levels found in the different tests to be later analyzed

    plt.plot(k_list, error_list)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')

    #save plot to images
    plt.savefig("../k-means-testing/clusters_analysis.png")

def comp(features, kstart, kend, kstep):
    '''
            DESCIPRTION
            Plot gausian values with their corresponding error
            :param:<features, kstart, kend, kstep; JSON, list, list>: featurelist / end, start, and step vals for model training
            :return: none
            :rtype: none
    '''

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
        #check for proper amount of labels
        if len(pred) <=1:
            continue
        if len(pred) <= 1 or len(pred) < k:
            continue
        error_list.append(cluster.bic(np.array(features)))
        copy_list.append(k)

    #plot vals with error
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
    list = sampling(20, features, 'smallest avg distance')
    print("asdfd")
    print(list)

    comp(features, 1, 100, 1)





main()
