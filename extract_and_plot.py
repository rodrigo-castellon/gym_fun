from __future__ import print_function
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import scipy.stats as stats
import pylab as pl

file = io.open("data3.txt","r")
new_file = io.open("timesteps.txt","w")
lines = file.readlines()

#Find all instances of "Episode n finished after n timesteps"
instances = []
count = 0
for i in range(len(lines)):
    ii = lines[i]
    try:
        if ii.index('Episode') == 0:
            instances.append(ii)
    except:
        continue

#Now instances holds all episodes and timesteps
for j in range(len(instances)):
    instances[j] = instances[j].split()
timesteps = []
for k in range(len(instances)):
    timesteps.append(instances[k][4])


X = timesteps
X = [int(X[i]) for i in range(len(X))]


y = [i for i in range(len(X))]
# calc the trendline
z = np.polyfit(y, X, 1)
p = np.poly1d(z)
plt.plot(y,p(y),"r--")
# the line equation:
print("y=%.6fx+(%.6f)"%(z[0],z[1]))

x = plt.plot(X,linewidth=0.2)
plt.show(x)
print("Total Episodes: %s " % len(X))

X = sorted(X)

fit = stats.norm.pdf(X, np.mean(X), np.std(X))  #this is a fitting indeed
pl.plot(X,fit,'-o')
dist1 = pl.hist(X,normed=True)      #use this to draw histogram of your data
pl.show(dist1)


hmean = np.mean(X)
hstd = np.std(X)
pdf = stats.norm.pdf(X, hmean, hstd)
dist2 = plt.plot(X, pdf) # including h here is crucial
plt.show(dist2)
