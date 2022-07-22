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
import os



# "/Users/ronnatarajan/Desktop/WebBench-Practice/data/web_bench_stats_random_100.json"
# **** name of true testing file ***** #
# web_bench_stats_random_100.json
def loadJSON(url):
    df = open(url, "r")

    #JSON file loaded into manipulatable variable
    df = json.loads(df.read())
    return df

def runBrowsertime(df):
    x = {}

    os.system("cd /Users/ronnatarajan/Desktop/WebBench")
    for site in df:

        resfile = "../data/" + site.replace("/","_").replace(".","__") + ".json"
        os.system("docker run --shm-size=1gb -v /Users/ronnatarajan/Desktop/WebBench/data:/data lighthouse-test lighthouse --output json  --port=9222 --preset=desktop http://" +site + " > " + resfile)

    # docker run -v /Users/ronnatarajan/Desktop/WebBench-Practice/docker/data:/data lighthouse-test browsertime --pageLoadStrategy normal --resultDir browsertime_results --output testing.json https://www.google.com



    #os.system('docker run --shm-size=1gb -v /Users/ronnatarajan/Documents/WebBench-Practice/docker/data lighthouse-test lighthouse --output json --port=9222 --preset=desktop ' + site +  " > " + resfile)
    #os.system(docker run --shm-size=1gb -v /Users/ronnatarajan/Documents/WebBench-Practice/docker/data lighthouse-test lighthouse --output json --port=9222 --preset=desktop https://www.google.com  > google_testing.json)

# ** working lighthouse statement
# ** docker run --shm-size=1gb -v /Users/ronnatarajan/Documents/WebBench-Practice/docker/data lighthouse-test lighthouse
# ** --output=json --output-path=results.json --port=9222 --preset=desktop https://www.google.com

# docker run -v /Users/ronnatarajan/Desktop/webmedic-dev/src/amp_analysis/src/docker/data:/data lighthouse-test browsertime --pageLoadStrategy normal --resultDir data --output testing.json https://www.google.com


def main():

    df = loadJSON("/Users/ronnatarajan/Desktop/WebBench/testing/smaller_webList/web_list.json")
    runBrowsertime(df)

main()
