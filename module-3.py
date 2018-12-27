import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D
#--------------module-3 project work-------------------------#
df=pd.read_csv('2008.csv')
#q1 and q2
print "percentage of money to be losse by AA trips are :"
money=ss.norm.cdf(x=20,loc=15,scale=3)
print 100.00-money*100

print "probability of flights diverted 10 out of 6"
diverted=float(ss.binom.pmf(6,10,.35))
print diverted
#q3 linear regression of Arrival Delay vs Departure Delay of flights
new_df=df[['DepDelay','Distance','UniqueCarrier','ArrDelay']].interpolate()
print "Linear Regression Chart for ArrDelay VS Dep delay "
sns.lmplot(x='DepDelay',y='ArrDelay',fit_reg=True,data=df,hue='UniqueCarrier',col='UniqueCarrier',ci=95 )
plt.ylim(0,1600)
plt.xlim(0,1600)
plt.xlabel('Arrdelay')
plt.ylabel('Depdelay')
plt.title('Arrdelay VS Depdelay flights')
plt.show()
#q4 Confidence interval value for this regression line for each unique carrier
print new_df
print "Confidence interval best regression line for Unique Carrier"
model=ols("ArrDelay ~ UniqueCarrier+DepDelay",new_df).fit()
print model.summary()
#multivariate analysis

print new_df
model1=ols("ArrDelay ~ C(UniqueCarrier) + DepDelay + Distance",data=new_df).fit()
print model1.summary()
print "plotting of chart multi regressions one by one of every carrier"

g=new_df.groupby('UniqueCarrier')
g1=g.get_group('AA')
# multi 3d plotting
X1 = g1[['DepDelay', 'Distance','UniqueCarrier']]
y1 = g1['ArrDelay']
#  grid for 3d plot
xx1, xx2 = np.meshgrid(np.linspace(X1.DepDelay.min(), X1.DepDelay.max(), 50),
                       np.linspace(X1.Distance.min(), X1.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z1 = model1.params[0] +model1.params[1] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf1 = ax.plot_surface(xx1, xx2, Z1,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y1 - model1.predict(X1)
ax.scatter(X1[resid >= 0].DepDelay, X1[resid >= 0].Distance, y1[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X1[resid < 0].DepDelay, X1[resid < 0].Distance, y1[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of AA carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf1 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#2nd plot
g2=g.get_group('AQ')
# multi 3d plotting
X2 = g2[['DepDelay', 'Distance','UniqueCarrier']]
y2 = g2['ArrDelay']
#  grid for 3d plot
xx12, xx22 = np.meshgrid(np.linspace(X2.DepDelay.min(), X2.DepDelay.max(), 50),
                       np.linspace(X2.Distance.min(), X2.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z2 = model1.params[0] +model1.params[2] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf2 = ax.plot_surface(xx12, xx22, Z2,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y2 - model1.predict(X2)
ax.scatter(X2[resid >= 0].DepDelay, X2[resid >= 0].Distance, y2[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X2[resid < 0].DepDelay, X2[resid < 0].Distance, y2[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of AQ carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf2 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#3rd plot
g3=g.get_group('AS')
# multi 3d plotting
X3 = g3[['DepDelay', 'Distance','UniqueCarrier']]
y3 = g3['ArrDelay']
#  grid for 3d plot
xx13, xx23 = np.meshgrid(np.linspace(X3.DepDelay.min(), X3.DepDelay.max(), 50),
                       np.linspace(X3.Distance.min(), X3.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z3 = model1.params[0] +model1.params[3] + model1.params[20] * xx13 + model1.params[21] * xx23

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf3 = ax.plot_surface(xx13, xx23, Z3,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y3 - model1.predict(X3)
ax.scatter(X3[resid >= 0].DepDelay, X3[resid >= 0].Distance, y3[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X3[resid < 0].DepDelay, X3[resid < 0].Distance, y3[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of AS carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf3 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#4th plot
g4=g.get_group('B6')
# multi 3d plotting
X4 = g4[['DepDelay', 'Distance','UniqueCarrier']]
y4 = g4['ArrDelay']
#  grid for 3d plot
xx14, xx24 = np.meshgrid(np.linspace(X1.DepDelay.min(), X1.DepDelay.max(), 50),
                       np.linspace(X1.Distance.min(), X1.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z4 = model1.params[0] +model1.params[4] + model1.params[20] * xx14 + model1.params[21] * xx24

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf4 = ax.plot_surface(xx14, xx24, Z4,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y4 - model1.predict(X4)
ax.scatter(X4[resid >= 0].DepDelay, X4[resid >= 0].Distance, y4[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X4[resid < 0].DepDelay, X4[resid < 0].Distance, y4[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of B6 carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf4 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#5th plot
g5=g.get_group('CO')
# multi 3d plotting
X5 = g5[['DepDelay', 'Distance','UniqueCarrier']]
y5 = g5['ArrDelay']
#  grid for 3d plot
xx15, xx25 = np.meshgrid(np.linspace(X5.DepDelay.min(), X5.DepDelay.max(), 50),
                       np.linspace(X5.Distance.min(), X5.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z5 = model1.params[0] +model1.params[5] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf5 = ax.plot_surface(xx15, xx25, Z5,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y5 - model1.predict(X5)
ax.scatter(X5[resid >= 0].DepDelay, X5[resid >= 0].Distance, y5[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X5[resid < 0].DepDelay, X5[resid < 0].Distance, y5[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of CO carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf5 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
# 6th plot
g6=g.get_group('DL')
# multi 3d plotting
X6 = g6[['DepDelay', 'Distance','UniqueCarrier']]
y6 = g6['ArrDelay']
#  grid for 3d plot
xx16, xx26 = np.meshgrid(np.linspace(X6.DepDelay.min(), X6.DepDelay.max(), 50),
                       np.linspace(X6.Distance.min(), X6.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z6 = model1.params[0] +model1.params[6] + model1.params[20] * xx16 + model1.params[21] * xx26

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf6 = ax.plot_surface(xx16, xx26, Z6,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y6 - model1.predict(X6)
ax.scatter(X6[resid >= 0].DepDelay, X6[resid >= 0].Distance, y6[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X6[resid < 0].DepDelay, X6[resid < 0].Distance, y6[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of DL carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf6 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#7th plot
g7=g.get_group('EV')
# multi 3d plotting
X7 = g7[['DepDelay', 'Distance','UniqueCarrier']]
y7 = g7['ArrDelay']
#  grid for 3d plot
xx17, xx27 = np.meshgrid(np.linspace(X7.DepDelay.min(), X7.DepDelay.max(), 50),
                       np.linspace(X7.Distance.min(), X7.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z7 = model1.params[0] +model1.params[7] + model1.params[20] * xx17 + model1.params[21] * xx27

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf7 = ax.plot_surface(xx17, xx27, Z7,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y7 - model1.predict(X7)
ax.scatter(X7[resid >= 0].DepDelay, X7[resid >= 0].Distance, y7[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X7[resid < 0].DepDelay, X7[resid < 0].Distance, y7[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of EV carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf7 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#8th plot
g8=g.get_group('F9')
# multi 3d plotting
X8 = g8[['DepDelay', 'Distance','UniqueCarrier']]
y8 = g8['ArrDelay']
#  grid for 3d plot
xx18, xx28 = np.meshgrid(np.linspace(X8.DepDelay.min(), X8.DepDelay.max(), 50),
                       np.linspace(X8.Distance.min(), X8.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z8 = model1.params[0] +model1.params[8] + model1.params[20] * xx18 + model1.params[21] * xx28

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf8 = ax.plot_surface(xx18, xx28, Z8,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y8 - model1.predict(X8)
ax.scatter(X8[resid >= 0].DepDelay, X8[resid >= 0].Distance, y8[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X8[resid < 0].DepDelay, X8[resid < 0].Distance, y8[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of F9 carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf8 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()

#10th plot
g10=g.get_group('FL')
# multi 3d plotting
X10 = g10[['DepDelay', 'Distance','UniqueCarrier']]
y10 = g10['ArrDelay']
#  grid for 3d plot
xx110, xx210 = np.meshgrid(np.linspace(X10.DepDelay.min(), X10.DepDelay.max(), 50),
                       np.linspace(X10.Distance.min(), X10.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z10 = model1.params[0] +model1.params[9] + model1.params[20] * xx110 + model1.params[21] * xx210

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf10 = ax.plot_surface(xx110, xx210, Z10,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y10 - model1.predict(X10)
ax.scatter(X10[resid >= 0].DepDelay, X10[resid >= 0].Distance, y10[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X10[resid < 0].DepDelay, X10[resid < 0].Distance, y10[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of HA carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf10 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#11th plot
g11=g.get_group('HA')
# multi 3d plotting
X11 = g11[['DepDelay', 'Distance','UniqueCarrier']]
y11 = g11['ArrDelay']
#  grid for 3d plot
xx111, xx211 = np.meshgrid(np.linspace(X11.DepDelay.min(), X11.DepDelay.max(), 50),
                       np.linspace(X11.Distance.min(), X11.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z11= model1.params[0] +model1.params[10] + model1.params[20] * xx111 + model1.params[21] * xx211

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf11 = ax.plot_surface(xx111, xx211, Z11,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y11 - model1.predict(X11)
ax.scatter(X11[resid >= 0].DepDelay, X11[resid >= 0].Distance, y11[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X11[resid < 0].DepDelay, X11[resid < 0].Distance, y11[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of MQ carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf11 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#12th plot
g12=g.get_group('MQ')
# multi 3d plotting
X12 = g12[['DepDelay', 'Distance','UniqueCarrier']]
y12 = g12['ArrDelay']
#  grid for 3d plot
xx112, xx212 = np.meshgrid(np.linspace(X12.DepDelay.min(), X12.DepDelay.max(), 50),
                       np.linspace(X12.Distance.min(), X12.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z12 = model1.params[0] +model1.params[11] + model1.params[20] * xx112 + model1.params[21] * xx212

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf12 = ax.plot_surface(xx112, xx212, Z12,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y12 - model1.predict(X12)
ax.scatter(X12[resid >= 0].DepDelay, X12[resid >= 0].Distance, y12[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X12[resid < 0].DepDelay, X12[resid < 0].Distance, y12[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of NW carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf12 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#13th plot
g13=g.get_group('NW')
# multi 3d plotting
X13 = g13[['DepDelay', 'Distance','UniqueCarrier']]
y13 = g13['ArrDelay']
#  grid for 3d plot
xx113, xx213 = np.meshgrid(np.linspace(X13.DepDelay.min(), X13.DepDelay.max(), 50),
                       np.linspace(X13.Distance.min(), X13.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z13 = model1.params[0] +model1.params[12] + model1.params[20] * xx113 + model1.params[21] * xx213

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf13 = ax.plot_surface(xx113, xx213, Z13,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y13 - model1.predict(X13)
ax.scatter(X13[resid >= 0].DepDelay, X13[resid >= 0].Distance, y13[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X13[resid < 0].DepDelay, X13[resid < 0].Distance, y13[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of OH carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf13 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#14 th plot
g14=g.get_group('OH')
# multi 3d plotting
X14 = g14[['DepDelay', 'Distance','UniqueCarrier']]
y14 = g14['ArrDelay']
#  grid for 3d plot
xx114, xx214 = np.meshgrid(np.linspace(X14.DepDelay.min(), X14.DepDelay.max(), 50),
                       np.linspace(X14.Distance.min(), X14.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z14 = model1.params[0] +model1.params[13] + model1.params[20] * xx114 + model1.params[21] * xx214

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf14 = ax.plot_surface(xx114, xx214, Z14,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y14 - model1.predict(X14)
ax.scatter(X14[resid >= 0].DepDelay, X14[resid >= 0].Distance, y14[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X14[resid < 0].DepDelay, X14[resid < 0].Distance, y14[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of OO carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf14 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#15 th plot
g15=g.get_group('OO')
# multi 3d plotting
X15 = g15[['DepDelay', 'Distance','UniqueCarrier']]
y15 = g15['ArrDelay']
#  grid for 3d plot
xx115, xx215 = np.meshgrid(np.linspace(X15.DepDelay.min(), X15.DepDelay.max(), 50),
                       np.linspace(X15.Distance.min(), X15.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z15 = model1.params[0] +model1.params[14] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf15 = ax.plot_surface(xx115, xx215, Z15,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y15 - model1.predict(X15)
ax.scatter(X15[resid >= 0].DepDelay, X15[resid >= 0].Distance, y15[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X15[resid < 0].DepDelay, X15[resid < 0].Distance, y15[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of UA carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf15 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#16 th plot
g16=g.get_group('UA')
# multi 3d plotting
X16 = g16[['DepDelay', 'Distance','UniqueCarrier']]
y16 = g16['ArrDelay']
#  grid for 3d plot
xx116, xx216 = np.meshgrid(np.linspace(X16.DepDelay.min(), X16.DepDelay.max(), 50),
                       np.linspace(X16.Distance.min(), X16.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z16 = model1.params[0] +model1.params[15] + model1.params[20] * xx116 + model1.params[21] * xx216

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf16 = ax.plot_surface(xx116, xx216, Z16,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y16 - model1.predict(X16)
ax.scatter(X16[resid >= 0].DepDelay, X16[resid >= 0].Distance, y16[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X16[resid < 0].DepDelay, X16[resid < 0].Distance, y16[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of US carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf16 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#17 th plot
g17=g.get_group('US')
# multi 3d plotting
X17 = g17[['DepDelay', 'Distance','UniqueCarrier']]
y17 = g17['ArrDelay']
#  grid for 3d plot
xx117, xx217 = np.meshgrid(np.linspace(X17.DepDelay.min(), X17.DepDelay.max(), 50),
                       np.linspace(X17.Distance.min(), X17.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z17 = model1.params[0] +model1.params[16] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf17 = ax.plot_surface(xx117, xx217, Z17,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y17 - model1.predict(X17)
ax.scatter(X17[resid >= 0].DepDelay, X17[resid >= 0].Distance, y17[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X17[resid < 0].DepDelay, X17[resid < 0].Distance, y17[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of WN carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf17 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#18 th plot
g18=g.get_group('WN')
# multi 3d plotting
X18 = g18[['DepDelay', 'Distance','UniqueCarrier']]
y18= g18['ArrDelay']
#  grid for 3d plot
xx118, xx218 = np.meshgrid(np.linspace(X18.DepDelay.min(), X18.DepDelay.max(), 50),
                       np.linspace(X18.Distance.min(), X18.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z18 = model1.params[0] +model1.params[17] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf18 = ax.plot_surface(xx118, xx218, Z18,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y18 - model1.predict(X18)
ax.scatter(X18[resid >= 0].DepDelay, X18[resid >= 0].Distance, y18[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X18[resid < 0].DepDelay, X18[resid < 0].Distance, y18[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of WN carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf18 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#19 th plot
g19=g.get_group('XE')
# multi 3d plotting
X19 = g19[['DepDelay', 'Distance','UniqueCarrier']]
y19 = g19['ArrDelay']
#  grid for 3d plot
xx119, xx219 = np.meshgrid(np.linspace(X19.DepDelay.min(), X19.DepDelay.max(), 50),
                       np.linspace(X19.Distance.min(), X19.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z19 = model1.params[0] +model1.params[18] + model1.params[20] * xx1 + model1.params[21] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf19 = ax.plot_surface(xx119, xx219, Z19,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y19 - model1.predict(X19)
ax.scatter(X19[resid >= 0].DepDelay, X19[resid >= 0].Distance, y19[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X19[resid < 0].DepDelay, X19[resid < 0].Distance, y19[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of XE carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf19 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()
#20th plot
g20=g.get_group('YV')
# multi 3d plotting
X20 = g20[['DepDelay', 'Distance','UniqueCarrier']]
y20 = g20['ArrDelay']
#  grid for 3d plot
xx2020, xx2021 = np.meshgrid(np.linspace(X20.DepDelay.min(), X20.DepDelay.max(), 50),
                       np.linspace(X20.Distance.min(), X20.Distance.max(), 50))
# plot the hyperplane by evaluating the parameters on the grid
Z9 = model1.params[0] +model1.params[19] + model1.params[20] * xx2020 + model1.params[21] * xx2021

# create matplotlib 3d axes
fig = plt.figure(figsize=(15, 8))
ax = Axes3D(fig, azim=-150, elev=20)

# plot hyperplane
surf20 = ax.plot_surface(xx2020, xx2021, Z9,cmap=plt.cm.RdBu_r , alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y20 - model1.predict(X20)
ax.scatter(X20[resid >= 0].DepDelay, X20[resid >= 0].Distance, y20[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X20[resid < 0].DepDelay, X20[resid < 0].Distance, y20[resid < 0], color='black', alpha=1.0)
# set axis labels
ax.set_title("ArrDelays Departure Distance of FL carrier")
ax.set_xlabel('DepDelay',color='red')
ax.set_ylabel('ArrDelay',color='blue')
ax.set_zlabel('Distance',color='green')
ax.set_facecolor(color='grey')
fig.colorbar(surf20 , shrink=0.5 , aspect=5)
plt.legend()
plt.show()