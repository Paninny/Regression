import pandas as pd
import numpy as np
import scipy.stats as sst
import statsmodels.formula.api as smf


readexe = pd.read_excel("Data/Chap5EMP.xlsx")
print(readexe)

#average value of Birthweight for all mothers
meanallmothers = np.mean(readexe["Birthweight"])
print("------meanallmotherss-------")
print(meanallmothers)
print("-----meanallmotherse-------")

#meansmoker
meanbygroup = readexe.groupby("Smoker")[["Birthweight"]].mean()
print("------meanbygroups-------")
print(meanbygroup)
print("------meanbygroupe-------")

#smokermean
smokermean = meanbygroup.iat[1,0]
print("-------smokermeans--------")
print(smokermean)
print("-------smokermeane--------")

#nonsmokermean
nonsmokermean = meanbygroup.iat[0,0]
print("-------nonsmokermeans--------")
print(nonsmokermean)
print("-------nonsmokermeane--------")

#difference
difference = nonsmokermean-smokermean
print("----------differs-----------")
print(difference)
print("-----------differe------------")

#stdbygroup
stdbygroup = readexe.groupby("Smoker")[["Birthweight"]].std()
print("-----------------stds-----------------")
print(stdbygroup)
print("-----------------stde-----------------")

#stdsm
stdsm = stdbygroup.iat[1,0]
print("-----------------stdsms-----------------")
print(stdsm)
print("-----------------stdsme-----------------")

#stdNONsm
stdNONsm = stdbygroup.iat[0,0]
print("-----------------stdNONsms-----------------")
print(stdNONsm)
print("-----------------stdNONsme-----------------")

#groupbysize
groupsize = readexe.groupby("Smoker").size()
print("-----------------groupsizes-----------------")
print(groupsize)
print("-----------------groupsizee-----------------")

#stderror
stderror = np.sqrt(stdsm**2/42+stdNONsm**2/58)
print("-----------------stderror-----------------")
print(stderror)
print("-----------------stderror-----------------")

#confidenceinterval
lbddiff = difference - sst.norm.ppf(1-0.05/2)*stderror
ubddiff = difference + sst.norm.ppf(1-0.05/2)*stderror
print("-----------------confidenceintervals-----------------")
print(lbddiff)
print(ubddiff)
print("-----------------confidenceintervale-----------------")

#firstregression
firstregression = smf.ols(formula="Birthweight~Smoker",data=readexe)
print("---------------------firstregressions-------------------")
print(firstregression.fit().summary())
print("--------------------firstregressione---------------------")

#secondregression
secondregression = smf.ols(formula="Birthweight~Smoker",data=readexe)
print("---------------------secondregressions-----------------")
print(secondregression.fit(cov_type="HC1").summary())
print("--------------------secondregressione-----------------")




