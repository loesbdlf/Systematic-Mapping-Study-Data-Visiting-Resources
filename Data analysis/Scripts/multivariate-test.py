import pandas as pd
from pandas import Series
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
import skbio.stats.distance as skbio
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def debug(string: str):
    if verbose:
        print(string)

def permdisp(distance_matrix: skbio.DistanceMatrix, labels: list[int]):
    result = skbio.permdisp(distance_matrix, labels, permutations=5000)
    debug('\n' + result.to_string())

    p = result["p-value"]
    print(p)
    if p > 0.05:
        print(bcolors.OKGREEN + f"Passes PERMDISP test with p: {p}" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + f"Fails PERMDISP test with p: {p}. Statistically different variances." + bcolors.ENDC)

def permanova(distance_matrix: skbio.DistanceMatrix, labels: list[int]):
    result = skbio.permanova(distance_matrix, labels, permutations=5000)
    debug('\n' + result.to_string())

    p = result["p-value"]
    if p <= 0.05:
        print(bcolors.OKGREEN + f"Passes PERMANOVA test with p: {p}" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + f"Fails PERMANOVA test with p: {p}. Statistically not significantly different." + bcolors.ENDC)

layered = [255, 384, 530, 165, 697, 25, 485, 156, 157, 82, 8]
mesh = [360, 582, 124, 682, 657, 174, 563, 9, 57, 209, 82, 166]
one_to_many = [288, 174, 142, 592, 454, 582,  217, 682, 124, 563, 9, 57, 160, 209, 11]
middleman_node = [549, 743, 659, 530, 442, 583, 621, 11]
cloud_storage = [384, 8, 530, 255, 165, 25, 583, 549, 697, 485, 156, 157]
centralized_storage = [90, 621, 549, 530, 25, 697, 255, 442, 485]

distributed = [360, 582, 124, 682, 657, 174, 563, 9, 57, 209, 82, 166, 288, 174, 142, 592, 454, 582,  217, 682, 124, 563, 9, 57, 160, 209, 11]
centralized = [549, 743, 659, 530, 442, 583, 621, 11, 90, 621, 549, 530, 25, 697, 255, 442, 485]

verbose = True
df = pd.read_csv("extracted-data.csv", sep=";")
categories = ["Researcher autonomy", "Data distribution", "Data visibility", "Anonymity of data", "Output control_20", "Trust and control distribution", "Auditability and traceability"]
data = df[categories].to_numpy()
labels = [1 if s in layered else 0 for s in df["Covidence #"]]
ids = [str(s) for s in df["Covidence #"]]

square_matrix = squareform(pdist(data, metric="cityblock"))
distance_matrix = skbio.DistanceMatrix(square_matrix, ids=ids)

permdisp(distance_matrix=distance_matrix, labels=labels)
permanova(distance_matrix=distance_matrix, labels=labels)
