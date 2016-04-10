import locale

import datetime
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# What you need to do:
#  1 Analyze min and max values of Istanbul Stock Market indexes using block maxima method (don’t forget to reconstruct all gaps)
#  2 Analyze min and max values of Istanbul Stock Market indexes using block POT method (don’t forget to prove chosen threshold)
#  3 Fit appropriate distribution for extreme values (GEV, GPD)
#  4 Find curves for VaR and ES criteria for various quantiles (0.9 – 0.999)

def month_converter(month):
    split = str(month).split("-")
    month = split[1]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return datetime.datetime(day=int(split[0]), month=months.index(month) + 1, year=int("20"+split[2]))

locale.setlocale(locale.LC_NUMERIC, '')
data = pd.read_csv('Istanbul_StockMarket_12-14.csv', delimiter=';', names=["date", "value"])
data['value'] = data['value'].str.replace(',', '.').astype('float')
data['date'] = data['date'].apply(month_converter)

plt.figure(figsize=(20, 10))
plt.title('Istanbul_StockMarket')
plt.xlabel('Date')
plt.ylabel('Value')
plt.plot(data['date'].values, data['value'].values, label="")
# plt.show()

maxs=[]

groupby = data.groupby(data['date'].map(lambda x: (x.year, x.month)))
for group in groupby:
    # print(list(map(lambda x:x[1],group[1].values)))
    print(""+str(group[0]) + " "+ str(max(map(lambda x:x[1],group[1].values))))
    maxs.append(max(map(lambda x:x[1],group[1].values)))


# N = 25
# groups = []
# for i in range(N):
#     groups.append([])
#
# print(data.size)
#
# i = 0
# for value in data.values:
#     if len(groups[i]) < N:
#         groups[i].append(value[1])
#     else:
#         i += 1
#
# print(groups)
#
#

