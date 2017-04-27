
# Demyanchuk is learning
import scipy as sp
import matplotlib.pyplot as plt

PATH = "/Users/alexeydemyanchuk/Machine_Learning/BuildingMachineLearningSystemsWithPython/"

# function to find approx error
def error(f, x, y):
    return sp.sum((f(x)-y)**2)

# data x: hours; y: hits per hour
data = sp.genfromtxt(PATH+"ch01/data/web_traffic.tsv", delimiter="\t")
# create vectors from data
x = data[:,0]
y = data[:,1]
# cleaning data from NaN
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# plot the (x,y) points with dots of size 10
plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")

plt.xticks([w*7*24 for w in range(10)],
               ['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
# draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-', color='0.75')

# find a function which fit into x and y data (straight line in this case)
# sp.polyfit HELPS!
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print(residuals)
print("Model parameters: %s" % fp1)

# create a model function from the model parameters
# sp.poly1d HELPS!
f1 = sp.poly1d(fp1)
print("Error for polynomial order 1 model is {}".format(error(f1, x, y)))

# add model to plot
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")
plt.show()
