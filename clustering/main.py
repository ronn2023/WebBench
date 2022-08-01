
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
import numpy as np
import scipy as sp
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import csv
import glob
import pandas as pd


WEBSITE_NAMES = []


def loadJSON(dynamic_url):
    '''
            DESCIPRTION
            loads json with website info
            :param:<url, String>: path of json file
            :return: JSON variable
            :rtype: JSON
    '''

    df = open(dynamic_url, "r")

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

def buildFeatureSet(df, apis, browser_events, sf):
    '''
            DESCIPRTION
            Build Feature list from dynamic feature JSON and static feature CSV files
            :param:<df, apis, browser_events, sf; JSON, list, list, CSV>: website data JSON, list of known API's and browser events, static feature data
            :return: 2D feature list
            :rtype: 2D list
    '''

    # load static feature data from CSV file
    static_info = {}
    with open(sf, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            static_info[row[0]] = row[1:]


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
    static = []
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
        file_path = "/Users/ronnatarajan/Desktop/WebBench/testing/full/" + site_file

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

        if len(num) == 0 or not len(num) == 4:
            continue

        # ----------------------------------------------------------- #
        # ***     Get num events for GPU analysis                 *** #
        # ----------------------------------------------------------- #
        # arr.append(df[site_name]['GPU']['num_events'])


        try:
            web_static_list = static_info["https://" + site_name + "/"]
        except:
            continue

        static_row = []
        for val in web_static_list:
            static_row.append(float(val))

        static.append(static_row)
        features.append(arr)


    print(len(features))
    print(len(static))



    #normalize data: MAX NORMALIZATION
    for col in range(len(features[0])):
        arr = []
        for row in range(len(features)):
            arr.append(features[row][col])
        for row in range(len(features)):
            if max(arr) == 0:
                features[row][col] =0
            else:
                features[row][col] = features[row][col] / max(arr)
    for col in range(len(static[0])):
        arr = []
        for row in range(len(static)):
            arr.append(static[row][col])
        for row in range(len(static)):
            if max(arr) == 0:
                static[row][col] =0
            else:
                static[row][col] = static[row][col] / max(arr)


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
    return features, static
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
def test_pca(features):
    cluster = KMeans(n_clusters=10, random_state=0).fit(features)

    # PCA on orig. dataset
    # Xred will have only 2 columns, the first two princ. comps.
    # evals has shape (4,) and evecs (4,4). We need all eigenvalues
    # to determine the portion of variance
    Xred, evals, evecs, numcomp = dim_red_pca(SN)

    xlab = '1. PC - ExpVar = {:.2f} %'.format(evals[0]/sum(evals)*100) # determine variance portion
    ylab = '2. PC - ExpVar = {:.2f} %'.format(evals[1]/sum(evals)*100)
    # plot the clusters, each set separately
    plt.figure()
    ax = plt.gca()
    scatterHs = []
    clr = ['r', 'b', 'k']
    for cluster in set(labels_):
        scatterHs.append(ax.scatter(Xred[labels_ == cluster, 0], Xred[labels_ == cluster, 1],
                       color=clr[cluster], label='Cluster {}'.format(cluster)))
    plt.legend(handles=scatterHs,loc=4)
    plt.setp(ax, title='Principle Components', xlabel=xlab, ylabel=ylab)
    # plot also the eigenvectors for deriving the influence of each feature
    fig, ax = plt.subplots(2,1)

    for i in range(numcomp):
        ax[i].bar([x for x in range(len(features))],evecs[i])
        plt.setp(ax[i], title="First and Second Component's Eigenvectors ", ylabel='Weight')
        ax[i].bar([x for x in range(len(features))],evecs[i])
        plt.setp(ax[1], xlabel='Features', ylabel='Weight')


def dim_red_pca(X, d=0, corr=False):
    r"""
    Performs principal component analysis.

    Parameters
    ----------
    X : array, (n, d)
        Original observations (n observations, d features)

    d : int
        Number of principal components (default is ``0`` => all components).

    corr : bool
        If true, the PCA is performed based on the correlation matrix.

    Notes
    -----
    Always all eigenvalues and eigenvectors are returned,
    independently of the desired number of components ``d``.

    Returns
    -------
    Xred : array, (n, m or d)
        Reduced data matrix

    e_values : array, (m)
        The eigenvalues, sorted in descending manner.

    e_vectors : array, (n, m)
        The eigenvectors, sorted corresponding to eigenvalues.

    """
    X = np.array(X)
    # Center to average
    X_ = X-X.mean(0)
    # Compute correlation / covarianz matrix
    if corr:
        CO = np.corrcoef(X_.T)
    else:
        CO = np.cov(X_.T)
    # Compute eigenvalues and eigenvectors
    e_values, e_vectors = sp.linalg.eigh(CO)

    # Sort the eigenvalues and the eigenvectors descending
    idx = np.argsort(e_values)[::-1]
    e_vectors = e_vectors[:, idx]
    e_values = e_values[idx]
    # Get the number of desired dimensions
    d_e_vecs = e_vectors
    if d > 0:
        d_e_vecs = e_vectors[:, :d]

    else:
        d = None
    # Map principal components to original data
    LIN = np.dot(d_e_vecs, np.dot(d_e_vecs.T, X_.T)).T
    return LIN[:, :d], e_values, e_vectors, len(d_e_vecs)


def sampling(testval, numclust, budget, features, selector):
    '''
            DESCIPRTION
            Sampling method
            :param:<budget, features, selector; int, 2D list, string>: desired num websites, 2D list of features, selector for sampling techniques
            :return: smaller_web
            :rtype: list
    '''

    # clustering object
    # cluster = DBSCAN(eps =1.2, min_samples = 2).fit(features)
    cluster = KMeans(n_clusters=numclust, random_state=0).fit(features)
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

        # with open('../clustering/' + testval, 'w') as f:
        #
        #     # create the csv writer
        #     writer = csv.writer(f)
        #     header = ['Closest WEBSITE NAME', 'Centroid Data']
        #     writer.writerow(header)
        #
        #     arr = []
        #     map = {}
        #
        #     for i in centroid_of_cluster:
        #         arr.append(i.tolist())
        #
        #     for j in range(len(arr[0])):
        #         copy = []
        #         for i in range(len(arr)):
        #             copy.append(str(arr[i][j]))
        #         str_line = copy
        #         writer.writerow(str_line)


        total = budget
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
            print(str(num))
            for p in points_of_cluster[c]:
                feature_index_P= features.index(p.tolist())

                for cp in points_of_cluster[c]:
                    if cp.tolist() == p.tolist():
                        continue
                    list = points_of_cluster[c].tolist()

                    sum += math.dist(p.tolist(), cp.tolist())

                distances[feature_index_P] = float(sum) / float((len(points_of_cluster[c])))

            val = int(math.ceil(num))
            if val > total:
                val = total - val
            for i in range(int(math.ceil(num))):

                best_web= min(distances, key=distances.get)
                smaller_web.append(WEBSITE_NAMES[best_web])
                del distances[best_web]
        return smaller_web


def write_json(filename, json_dict):
    with open(filename,"w") as f:
        json.dump(json_dict, f, indent=4)

    df = pd.DataFrame.from_dict(json_dict, orient="index")
    df.to_csv("/Users/ronnatarajan/Desktop/WebBench/testing/complexity_metrics.csv")


def caclRequestsData(requests, speedIndex):
    # Num of Req before SpeedIndex
    # Transfer Size before SpeedIndex

    # Num of Req after SpeedIndex
    # Transfer Size after SpeedIndex
    data = {}
    data["RequestsBeforeSI"] = 0
    data["TransferBeforeSI"] = 0
    data["RequestsAfterSI"]  = 0
    data["TransferAfterSI"]  = 0
    for eachItem in requests:
        if "endTime" in eachItem:
            if eachItem["endTime"] < speedIndex:
                data["RequestsBeforeSI"] += 1
                data["TransferBeforeSI"] += eachItem["transferSize"]
            else:
                data["RequestsAfterSI"] += 1
                data["TransferAfterSI"] += eachItem["transferSize"]
        elif "startTime" in eachItem:
            if eachItem["startTime"] < speedIndex:
                data["RequestsBeforeSI"] += 1
                data["TransferBeforeSI"] += eachItem["transferSize"]
            else:
                data["RequestsAfterSI"] += 1
                data["TransferAfterSI"] += eachItem["transferSize"]
    return data

def buildStaticJSON():
    count = 0
    err   = 0
    urls = []
    mega_data = {}
    for path in  glob.iglob('/Users/ronnatarajan/Desktop/WebBench/testing/full/*.json'):
        try:
            if path == "all_urls.json":
                continue
            else:
                with open(path,"r") as f:
                    data = json.load(f)
                    count += 1
                    if data["audits"]["metrics"]["details"]["items"][0]["observedLoad"] < 1000:
                        # print(data["requestedUrl"])
                        err += 1
                    else:
                        # pass
                        # print(data["audits"]["metrics"]["details"]["items"][0]["observedLoad"]/1000.0,data["audits"]["speed-index"]["numericValue"]/1000.0, data["audits"]["interactive"]["numericValue"]/1000.0)
                        # print(data["audits"]["metrics"]["details"]["items"][0]["observedLoad"]/1000.0,data["audits"]["speed-index"]["numericValue"]/1000.0, data["audits"]["interactive"]["numericValue"]/1000.0,path)
                        # urls.append(data["requestedUrl"])
                        mega_data[data["requestedUrl"]] = complexity(data)

        except Exception as e:
            print("Error in Parsing:",e)
    # print("Total Sites:",count)
    # print("Erred Sites:",err)
    # print("Erred Sites %age:",err/count*100.0)
    write_json("/Users/ronnatarajan/Desktop/WebBench/testing/complecity_metrics.json", mega_data)



def complexity(data):
    metrics = {}
    metrics["total_requests"]   = list(filter(lambda x: x["label"] == "Total" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["js_requests"]      = list(filter(lambda x: x["label"] == "Script" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["img_requests"]     = list(filter(lambda x: x["label"] == "Image" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["doc_requests"]     = list(filter(lambda x: x["label"] == "Document" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["css_requests"]     = list(filter(lambda x: x["label"] == "Stylesheet" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["font_requests"]    = list(filter(lambda x: x["label"] == "Font" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["media_requests"]   = list(filter(lambda x: x["label"] == "Media" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["other_requests"]   = list(filter(lambda x: x["label"] == "Other" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["3rdparty_requests"]= list(filter(lambda x: x["label"] == "Third-party" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["requestCount"]
    metrics["total_size"]       = list(filter(lambda x: x["label"] == "Total" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["js_size"]          = list(filter(lambda x: x["label"] == "Script" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["img_size"]         = list(filter(lambda x: x["label"] == "Image" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["doc_size"]         = list(filter(lambda x: x["label"] == "Document" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["css_size"]         = list(filter(lambda x: x["label"] == "Stylesheet" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["font_size"]        = list(filter(lambda x: x["label"] == "Font" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["media_size"]       = list(filter(lambda x: x["label"] == "Media" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["other_size"]       = list(filter(lambda x: x["label"] == "Other" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0
    metrics["3rdparty_size"]    = list(filter(lambda x: x["label"] == "Third-party" ,data["audits"]["resource-summary"]["details"]["items"]))[0]["transferSize"]/1000.0

    metrics["firstContentfulPaint"]         = data["audits"]["metrics"]["details"]["items"][0]["firstContentfulPaint"]
    metrics["firstMeaningfulPaint"]         = data["audits"]["metrics"]["details"]["items"][0]["firstMeaningfulPaint"]
    metrics["largestContentfulPaint"]       = data["audits"]["metrics"]["details"]["items"][0]["largestContentfulPaint"]
    metrics["totalCumulativeLayoutShift"]   = data["audits"]["metrics"]["details"]["items"][0]["totalCumulativeLayoutShift"]
    metrics["speedIndex"]                   = data["audits"]["metrics"]["details"]["items"][0]["speedIndex"]
    metrics["interactive"]                  = data["audits"]["metrics"]["details"]["items"][0]["interactive"]
    metrics["observedLoad"]                 = data["audits"]["metrics"]["details"]["items"][0]["observedLoad"]
    metrics["observedDomContentLoaded"]     = data["audits"]["metrics"]["details"]["items"][0]["observedDomContentLoaded"]

    metrics.update(caclRequestsData(data["audits"]["network-requests"]["details"]["items"], data["audits"]["metrics"]["details"]["items"][0]["observedSpeedIndex"] ))
    return metrics


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
    '''Uncomment Below Code to build Static Data File's '''
    #buildStaticJSON()


    # df = loadJSON("/Users/ronnatarajan/Desktop/WebBench/testing/smaller_webList/web_list.json")
    df = loadJSON("/Users/ronnatarajan/Desktop/WebBench/testing/web_bench_stats_random_100.json")
    #
    apis, browser_events = gatherUsedAPIs(df)
    sf = "/Users/ronnatarajan/Desktop/WebBench/testing/complexity_metrics.csv"
    features, static = buildFeatureSet(df, apis, browser_events, sf)
    # # error_list, k_list = trainKMeans(features, 1, 10, 1)
    # # k_means_plot(error_list, k_list) # comes up with 5 suggested clusters
    #
    # static_error_list, static_k_list = trainKMeans(static, 1, 10, 1)
    # k_means_plot(static_error_list, static_k_list) # comes up with 5 suggested clusters but less error


    #
    # arr = []
    # for i in range(len(features)):
    #     arr.append(features[i] + static[i])
    #
    # list = sampling("both",5, 5, arr, 'smallest avg distance')

    # cpy_list = sampling("dynamic",5,5,features, 'smallest avg distance')
    # print(list)
    # print("asdfd")
    # print(list)
    #
    # comp(features, 1, 100, 1)

    # LIN, e_values, e_vectors, numcomp = dim_red_pca(features)



main()
