import pandas as pd
from pandas import Series
from scipy import stats
import statsmodels.stats.api as sms
import statsmodels.stats.multitest as mt

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

def levene(incl: Series, excl: Series) -> float:
    debug("\n"+ bcolors.OKBLUE + "Testing homogeneity of variance" + bcolors.ENDC)
    stat, p = stats.levene(
        incl,
        excl
        )   

    debug(f"Levene statistic: {stat}")
    debug(f"p-value: {p}")

    if p > 0.05:
        print(f"Passes Levene's test with p: {p}")
    else:
        print("Fails Levene's test. Variances are statistically different.")

    return p

def mann_whitney_u(incl: Series, excl: Series, incl_mean: float, excl_mean: float) -> tuple[float, float]:
    alternative_hypothesis = "less" if incl_mean < excl_mean else "greater"
    debug("\n" + bcolors.OKBLUE + f"Testing statistical significance using Mann-Whitney U test. Alternative hypothesis: {alternative_hypothesis}" + bcolors.ENDC)
    u_stat, p = stats.mannwhitneyu(
        incl,
        excl,
        alternative='two-sided'
    )

    debug(f"U = {u_stat}")
    debug(f"p = {p}")

    if p <= 0.05:
        debug(bcolors.OKGREEN + f"Passes Mann-Whitney U test with p: {p}" + bcolors.ENDC)
    else:
        debug(bcolors.FAIL + f"Fails Mann-Whitney U test with p: {p}. No statistically significant difference." + bcolors.ENDC)
    return u_stat, p

def debug(string: str):
    if verbose:
        print(string)

df = pd.read_csv("extracted-data.csv", sep=";")

# Selection for analysis
layered = [255, 384, 530, 165, 697, 25, 485, 156, 157, 82, 8]
mesh = [360, 582, 124, 682, 657, 174, 563, 9, 57, 209, 82, 166]
one_to_many = [288, 174, 142, 592, 454, 582,  217, 682, 124, 563, 9, 57, 160, 209, 11]
middleman_node = [549, 743, 659, 530, 442, 583, 621, 11]
cloud_storage = [384, 8, 530, 255, 165, 25, 583, 549, 697, 485, 156, 157]
centralized_storage = [90, 621, 549, 530, 25, 697, 255, 442, 485]

distributed = [360, 582, 124, 682, 657, 174, 563, 9, 57, 209, 82, 166, 288, 174, 142, 592, 454, 582,  217, 682, 124, 563, 9, 57, 160, 209, 11]
centralized = [549, 743, 659, 530, 442, 583, 621, 11, 90, 621, 549, 530, 25, 697, 255, 442, 485]

categories = ["Researcher autonomy", "Data distribution", "Data visibility", "Anonymity of data", "Output control_20", "Trust and control distribution", "Auditability and traceability"]
verbose = False 

# Running analysis
results = []

for category in categories:
    debug("\n" + bcolors.OKBLUE + f"Running all tests for category: {bcolors.HEADER} {category}" + bcolors.ENDC)
    
    all = df[category]
    all_mean = all.mean()
    incl = df.loc[df["Covidence #"].isin(layered)][category]
    incl_mean = incl.mean()
    excl = df.loc[~df["Covidence #"].isin(layered)][category]
    excl_mean = excl.mean()

    debug(f"Mean of included: {incl_mean}, mean of excluded: {excl_mean}")

    _ = levene(incl=incl, excl=excl)

    u, p = mann_whitney_u(
        incl=incl,
        excl=excl,
        incl_mean=incl_mean,
        excl_mean=excl_mean
    )

    results.append((category, incl_mean, excl_mean, u, p))

# Correction
reject, pvals_corrected, alphacSidak, alphacBonf = mt.multipletests(
    [p for (_,_,_,_,p) in results],
    alpha=0.05,
    method="fdr_bh"
)

print(
    f"{bcolors.OKBLUE}{'Category':<35}{'Mean':>12}{'Rest mean':>12}{'U':>12}{'Raw p':>12}{'Holm p':>12}{'Significant':>12}{bcolors.ENDC}"
)
significant = False
for i, (category, i_mean, e_mean, u, raw_p) in enumerate(results):
    adj_p = pvals_corrected[i]
    if adj_p <= 0.05:
        colour = bcolors.OKGREEN 
        significant = True
        sig = True
    else:
        colour = ""
        sig = False

    print(
        f"{colour}{category:<35} "
        f"{i_mean:>12.6f}"
        f"{e_mean:>12.6f}"
        f"{u:>12.6f}"
        f"{raw_p:>12.6f} "
        f"{adj_p:>12.6f} "
        f"{str(sig):>12}{bcolors.ENDC}"
    )

if not significant:
    print(f"{bcolors.FAIL}No statistically significant differences.{bcolors.ENDC}")
