
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

inflection = int(3.5*7*24) # calculate the inflection point in hours
xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("Error inflection=%f" % (fa_error + fb_error))

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



# add model to plot
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, fa(fx), linewidth=4)
plt.plot(fx, fb(fx),color='blue', linestyle="-.",linewidth=4)
plt.legend(["d=%i" % fa.order,"d=%i" % fb.order], loc="upper left")


plt.show()
