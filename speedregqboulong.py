import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date,timedelta
from scipy import io as io
from scipy.signal import detrend as detrend
from netCDF4 import Dataset
import matplotlib.patheffects as pe
mpl.rcParams['contour.negative_linestyle'] = 'dashed'
mpl.rcParams.update({'font.size': 13})

'''speedregqboulong.py loads standard RMM index and the Oliver and Thomson 
reconstructed index, averages values to monthly, and then finds their 
association with 70 hPa zonal wind from 5N to 5S previously obtained from
the ERA5 reanalysis, and the 70-200 hPa vertical shear of the zonal wind
from the same reanalysis but also downloaded separately.'''

rmm=np.loadtxt('/roundylab_rit/roundy/era5/rmmrecon.txt',delimiter=',',skiprows=1) #Reconstructed RMM index
rmmstandard=np.loadtxt('/roundylab_rit/roundy/rmm.74toRealtime.txt')#Standard RMM index used after 1979

#Create a Fractional year time series for plotting data by year
rmmyear=np.array([rmm[n,0]+(date(int(rmm[n,0]),int(rmm[n,1]),int(rmm[n,2]))-date(int(rmm[n,0]),1,1)).days/366 for n in np.arange(rmm.shape[0]).astype(int)]) 

#Reminder, the two indexes swap the last two columns of phase and amplitude
rmmamp=rmm[:,5]  
rmmampsm=rmmamp.copy()
maxlag=10
for t in np.arange(maxlag,rmmamp.shape[0]-maxlag):
     rmmamp[t]=rmmampsm[t-maxlag:t+maxlag].mean()

#Create year and month number indexes
yearind=np.zeros((2027-1940)*12)
monthind=np.zeros((2027-1940)*12)
for year in np.arange(1940,2027):
     yearind[(year-1940)*12:(year-1940+1)*12]=year+(np.arange(12)/12)
     monthind[(year-1940)*12:(year-1940+1)*12]=(np.arange(12))+1

#Load 200 hPa zonal wind data from ERA5 monthly. Data are 5N to 5S. 
D=Dataset('/roundylab_rit/roundy/era5/u200monthly1940.nc','r')
uarray200=D.variables['u'][:].squeeze()
uarray200=uarray200.mean(axis=1) #Average in Latitude

#Block average the quarter degree longitude data to 1 degree
x=np.arange(0,1440,4)
uarray200=(uarray200[:,x]+uarray200[:,x+1]+uarray200[:,x+2]+uarray200[:,x+3])/4

#Load the similar u70 monthly data and do the same averaging
D=Dataset('/roundylab_rit/roundy/era5/u70monthly.nc','r')
uarray70=D.variables['u'][:].squeeze()
uarray70=uarray70.mean(axis=1)
uarray70=(uarray70[:,x]+uarray70[:,x+1]+uarray70[:,x+2]+uarray70[:,x+3])/4

#Load and similarly process 50 hPa zonal wind data similarly (these weren't used)

D=Dataset('/roundylab_rit/roundy/era5/u50monthly.nc','r')
uarray50=D.variables['u'][:].squeeze()
uarray50=uarray50.mean(axis=1)
uarray50=(uarray50[:,x]+uarray50[:,x+1]+uarray50[:,x+2]+uarray50[:,x+3])/4

#Create monthly RMM amplitude timeseries
rmmmonthly=np.zeros_like(uarray70.mean(axis=1))
for year in np.arange(1940,2025):
     for month in np.arange(12):
          if year<1979:
               Imonth=np.where(np.logical_and(rmm[:,0]==year,rmm[:,1]==month+1))[0]
               rmmmonthly[(year-1940)*12+month]=rmm[Imonth,5].mean()
          else:
               Imonth=np.where(np.logical_and(rmmstandard[:,0]==year,rmmstandard[:,1]==month+1))[0]
               rmmmonthly[(year-1940)*12+month]=rmmstandard[Imonth,6].mean()




yearind=yearind[:uarray200.shape[0]] #trim data for consistency
monthind=monthind[:uarray200.shape[0]]


uarray200=np.hstack((uarray200[:,180:],uarray200[:,:180])) #Swap longitudes so array begins at 0E for tropospheric data


#Calculate linear through cubic trend
uarray200sm=uarray200.copy()
t=np.arange(uarray200.shape[0])
X=np.ones((uarray200.shape[0],4))
X[:,1]=t
X[:,2]=t**2
X[:,3]=t**3
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(uarray200))
trend=X.dot(C)

#Estimate seasonal cycle
harmonicnum=4  #Number of harmonics for seasonal cycle extraction. 
def harmbuild(N,harmonicnum):
     """Constructs matrix of Fourier harmonics for Fourner regresssion."""
     t=np.arange(N)
     period=12
     j=1
     cycle=np.ones((N,2*harmonicnum+1))
     for i in np.arange(1,harmonicnum+1):
         cycle[:,j]=np.sin(i*2*np.pi*t/period)
         cycle[:,j+1]=np.cos(i*2*np.pi*t/period)
         j=j+2
     return cycle

X=harmbuild(uarray200.shape[0],harmonicnum)
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(uarray200-trend))
cycle200=X.dot(C)

#Plot time series form of the zonal wind and shear data
N3=int(uarray200.shape[0]/4)
shear=(uarray70[:,50:80]-uarray200[:,50:80]).mean(axis=1) #Shear across the tropopause over the Indian Ocean

plt.figure(figsize=(8.5,11))
for i in np.arange(4):
     plt.subplot(4,1,i+1)
     plt.plot(yearind[i*N3:(i+1)*N3],uarray70[i*N3:(i+1)*N3,50:80].mean(axis=1),'r',label='u70')
     plt.plot(yearind[i*N3:(i+1)*N3],trend[i*N3:(i+1)*N3,50:80].mean(axis=1),'r',linestyle=':')
     plt.plot(yearind[i*N3:(i+1)*N3],uarray200[i*N3:(i+1)*N3,50:80].mean(axis=1),'b',label='u200')
     #plt.plot(yearind[i*N3:(i+1)*N3],trend[i*N3:(i+1)*N3,50:80].mean(axis=1)+biennial[i*N3:(i+1)*N3,50:80].mean(axis=1),'b',linestyle='-.')
     plt.plot(yearind[i*N3:(i+1)*N3],trend[i*N3:(i+1)*N3,100:130].mean(axis=1)+cycle200[i*N3:(i+1)*N3,100:130].mean(axis=1),'b',linestyle='--')

     plt.plot(yearind[i*N3:(i+1)*N3],shear[i*N3:(i+1)*N3],'k',linewidth=3,label='Shear')
     plt.plot(yearind[i*N3:(i+1)*N3],rmmmonthly[i*N3:(i+1)*N3]*8,'c',linewidth=3,label='RMM')
     if i==0:
          plt.title('u200, u70, 70-200 hPa \nVertical Shear of Zonal Wind, and RMM Amplitude')
          plt.legend(loc='upper right')
     plt.grid(True)

     plt.ylabel('m/s')


plt.savefig('/pr11/roundy/public_html/exam1.png')


#Create Figure 2, showing the wind data in a year x month contour format
u70mean=uarray70[:,50:80].mean(axis=1)
u200mean=uarray200[:,50:80].mean(axis=1)
shearmonth=np.zeros((2025-1940,12))
u70month=np.zeros((2025-1940,12))
u200month=np.zeros((2025-1940,12))
rmmmonth=np.zeros((2025-1940,12))
yearindint=np.floor(yearind)
for year in np.arange(1940,2025):
     for month in np.arange(12):
          I=np.where(np.logical_and(yearindint==year,monthind==month+1))[0]
          shearmonth[year-1940,month]=shear[(year-1940)*12+month]
          u70month[year-1940,month]=u70mean[(year-1940)*12+month]
          u200month[year-1940,month]=u200mean[(year-1940)*12+month]
          rmmmonth[year-1940,month]=rmmmonthly[(year-1940)*12+month]


fig=plt.figure(figsize=(8.5,11))
years=np.arange(1940,2025)
months=np.arange(1,13)
V=np.arange(-36,37,4)
Vrmm=np.arange(1.5,5,.5)

plt.subplot(1,2,1)
plt.contourf(months,years,shearmonth,V,cmap='bwr')
cbar=plt.colorbar()
cbar.ax.set_visible(False)
plt.contour(months,years,u70month,V,colors='k')
plt.contour(months,years,rmmmonth,Vrmm,vmin=1,vmax=3,cmap='cool',linewidth=2,path_effects=[pe.TickedStroke(spacing=1,length=2,angle=90)])
plt.axis((1,12,1940,1986))
plt.xticks(np.arange(1,12,2))
plt.xlabel('Month')
plt.subplot(1,2,2)
plt.contourf(months,years,shearmonth,V,cmap='bwr')
plt.colorbar()
plt.contour(months,years,u70month,V,colors='k')
plt.contour(months,years,rmmmonth,Vrmm,vmin=1,vmap=3,cmap='cool',linewidth=2,path_effects=[pe.TickedStroke(spacing=1,length=2,angle=90)])


plt.xticks(np.arange(1,12,2))
plt.xlabel('Month')
plt.axis((1,12,1986,2025))
fig.suptitle('Monthly Shear (Shading),\n u70 (Black Contours), and MJO Activity (Cyan)', fontsize=16, fontweight='bold')
plt.savefig('/pr11/roundy/public_html/exam30.png')

#Code below finds the various correlations presented in the paper

def std(x):
     '''standard deviation for data centered on zero instead of subtracting the mean'''
     return(np.sqrt(np.mean(x*x)))



def correlate(x,y,u200):
    '''Calculates the Correlation coefficient without subtracting the mean, for cases in which the displacement from 
    zero is intended to be maintained in the correlation.'''
    #x=x-x.mean()
    y=y-y.mean()
    x=x/std(x)
    y=y/np.std(y)
    corrs=np.zeros(12)
    corrsbs=np.zeros((12,10000))
    I200=np.logical_and(u200<0,u200>-17)
    I200=np.ones_like(u200)
    x=x*I200
    for month in np.arange(12):
         xx=x[:,month]
         yy=y[:,month]
         corrs[month]=xx.dot(yy)/xx.shape[0]
         for bsnum in np.arange(10000):
             Irand=np.random.randint(0,xx.shape[0],xx.shape[0]) 
             corrsbs[month,bsnum]=xx[Irand].dot(yy[Irand])/xx.shape[0]
    corrsbs=np.sort(corrsbs,axis=1)[:,[250,9750]]
    return corrs,corrsbs

actcors,actcorsbs=correlate(shearmonth,rmmmonth,u200month)
actcorsu70,actcorsu70bs=correlate(u70month,rmmmonth,u200month)


plt.figure()
plt.plot(months,actcors,label='Shear',color='b')
plt.plot(months,actcorsbs,linestyle='--',color='b')
plt.plot(months,actcorsu70,label='u70',color='r')
plt.plot(months,actcorsu70bs,linestyle='--',color='r')
plt.xlabel('Month')
plt.ylabel('Correlation')
plt.grid(True)
plt.legend()
plt.title('Monthly Correlation Between MJO Activity\n and u70 and 70-200 hPa Shear')
plt.savefig('/pr11/roundy/public_html/exam31.png')


I1980=np.where(yearind>=1980)[0][0]
uind=uarray200[:,50:80].mean(axis=1)
print('Pre 1980 then post')
print(np.sum(np.logical_and(shear[:I1980]<-5,uind[:I1980]<-5))/I1980)
print(np.sum(np.logical_and(shear[I1980:]<-5,uind[I1980:]<-5))/(shear.shape[0]-I1980))

nn=np.where(rmmmonthly==0)[0][0]

Idjfpre1980=np.where(np.logical_or(monthind[:I1980]<3,monthind[:I1980]==12))[0]
Idjfpost1980=np.where(np.logical_or(monthind[I1980:nn]<3,monthind[I1980:nn]==12)+I1980)[0]

print('shear pre 1980, post, u70 pre, post')
print(np.corrcoef(rmmmonthly[:I1980],shear[:I1980])[0,1])
print(np.corrcoef(rmmmonthly[I1980:nn],shear[I1980:nn])[0,1])
print(np.corrcoef(rmmmonthly[:I1980],uarray70[:I1980,50:80].mean(axis=1))[0,1])
print(np.corrcoef(rmmmonthly[I1980:nn],uarray70[I1980:nn,50:80].mean(axis=1))[0,1])
print(np.corrcoef(rmmmonthly[:I1980],uarray50[:I1980,50:80].mean(axis=1))[0,1])
print(np.corrcoef(rmmmonthly[I1980:nn],uarray50[I1980:nn,50:80].mean(axis=1))[0,1])
print('uarray70 correlation DJF')
print(rmmmonthly.shape)
print(np.corrcoef(rmmmonthly[Idjfpre1980],uarray70[Idjfpre1980,50:80].mean(axis=1))[0,1])
print(np.corrcoef(rmmmonthly[Idjfpost1980],uarray70[Idjfpost1980,50:80].mean(axis=1))[0,1])
print(np.corrcoef(rmmmonthly[Idjfpre1980],uarray50[Idjfpre1980,50:80].mean(axis=1))[0,1])
print(np.corrcoef(rmmmonthly[Idjfpost1980],uarray50[Idjfpost1980,50:80].mean(axis=1))[0,1])




