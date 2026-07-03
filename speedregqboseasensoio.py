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
mpl.rcParams['contour.negative_linestyle'] = 'dashed'
mpl.rcParams.update({'font.size': 13})

'''Code for the daily data calculations of "The Role of Stratosphere-
Troposphere Vertical Shear of the Zonal Wind in the QBO-MJO relationship"'''


RMM=np.loadtxt('/roundylab_rit/roundy/RMMindex.txt') #BOM RMM index
print(RMM.shape)
RMMamp=RMM[:,6] #Amplitude series
I=np.where(RMMamp>6)[0] #set missing data values to zero
RMMamp[I]=0

cutter=(date(1974,6,1)-date(1974,1,1)).days  #Used to trim arrays to the same as the RMM index start date

#Section below opens NOAA extended reconstructed SST data, but these 
#were not used in the paper. 
F=Dataset('/roundylab_rit/roundy/era5/sst.mnmean.nc','r')
sstlat=F['lat'][:]
sstlon=F['lon'][:]
time=F['time'][:]
sst=F['sst'][:]
print(time.min())
print(sst.max())
d0=date(1800,1,1)
yearf=np.array([(d0+timedelta(days=day)).year + ((d0+timedelta(days=day)).month-1)/12 for day in time])
year=np.array([(d0+timedelta(days=day)).year for day in time])
month=np.array([(d0+timedelta(days=day)).month for day in time])

Isstkeep=np.where(np.logical_and(year>=1974,year<=2020))[0]
sst=sst[Isstkeep,:,:]
year=year[Isstkeep]
yearf=yearf[Isstkeep]
month=month[Isstkeep]
Ilat=np.where(np.abs(sstlat)<=5)[0]
Ilon=np.where(np.logical_and(sstlon>=360-170,sstlon<=360-120))[0]
sst=sst[:,Ilat,:].mean(axis=1)
nino34=sst[:,Ilon].mean(axis=1)
plt.figure()
plt.plot(yearf,nino34)
plt.savefig('/pr11/roundy/public_html/exam.png')

#Load the daily ERA5 u200 data (these data were originally 15N to 15S, which gets subset

Y=np.arange(-15,16)
Iy=np.where(np.abs(Y)<=5)[0]
uarray200=np.load('/roundylab_rit/roundy/era5/u200.npy')
uarray200=uarray200[:,Iy,:].mean(axis=1)
uarray70=np.load('/roundylab_rit/roundy/era5/u5n5s1974_70.npy')

uarray2001974=np.load('/roundylab_rit/roundy/era5/u5n5s1974_200.npy')
uarray200=np.vstack((uarray2001974,uarray200))
uarray70=uarray70[:uarray200.shape[0],:]

#Create date index arrays
d0=date(1974,1,1)
drmm=date(1974,6,1)
d1=date(2020,12,31)

dates=np.array([d0+timedelta(days=int(n)) for n in np.arange(uarray70.shape[0])])
datesmjo=np.array([drmm+timedelta(days=int(n)) for n in np.arange((d1-drmm).days)])
monthseries=np.array([d.month for d in dates])
Imonth=np.where(monthseries==12)[0]
yearseries=np.array([d.year for d in dates])
yearseriesmjo=np.array([d.year for d in datesmjo])

#Create timeseries of fractional year
yearseriesfrac=np.zeros_like(yearseries).astype(float)
yearseriesfracmonth=np.zeros_like(yearseries).astype(float)
yearseriesfracmjo=np.zeros_like(yearseries).astype(float)

for d in np.arange(dates.shape[0]):
     yearseriesfrac[d]=yearseries[d]+(dates[d]-date(dates[d].year-1,12,31)).days/366
     yearseriesfracmonth[d]=yearseries[d]+(dates[d].month-1)/12
for d in np.arange(datesmjo.shape[0]):
     yearseriesfracmjo[d]=yearseriesmjo[d]+(datesmjo[d]-date(datesmjo[d].year-1,12,31)).days/366

nino34dly=np.zeros(dates.shape[0])

for t in np.arange(nino34.shape[0]):
     I=np.where(np.logical_and(yearseries==year[t],monthseries==month[t]))[0]
     nino34dly[I]=nino34[t]

slidermonth=np.arange(0,14)
slidermonth[0]=12
slidermonth[13]=1

#Load the QBO 70 hPa zonal wind
qbo=np.load('/roundylab_rit/roundy/era5/u5n5sglob70.npy')#[:,16,:]
qbo=qbo.mean(axis=1)
qbosm=qbo.copy()
smoother=50 #101-day centered moving average to smooth out subseasonal variability
for t in np.arange(smoother,qbosm.shape[0]-smoother):
     qbo[t]=qbosm[t-smoother:t+smoother].mean()
qbo[:smoother]=0
qbo[qbo.shape[0]-smoother:]=0
qbodly=qbo.copy()

N=uarray200.shape[0]
Y=np.arange(-15,16,1)
Iy=np.where(np.abs(Y)<=5)[0]

#Flip ERA5 data arrays in longitude so that they begin at 0E
uarray200=np.concatenate((uarray200[:,180:],uarray200[:,:180]),axis=1)
uarray70=np.concatenate((uarray70[:,180:],uarray70[:,:180]),axis=1)

uarray200sm=uarray200.copy()
maxlag=30

#Find the linear through cubic trend
t=np.arange(uarray200.shape[0])
X=np.ones((uarray200.shape[0],4))
X[:,1]=t
X[:,2]=t**2
X[:,3]=t**3
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(uarray200))
trend=X.dot(C)

#Seasonal cycle estimator algorithm
harmonicnum=4  #Number of harmonics for seasonal cycle extraction. 
def harmbuild(N,harmonicnum):
     """Constructs matrix of Fourier harmonics for Fourner regresssion."""
     t=np.arange(N)
     period=365.25
     j=1
     cycle=np.ones((N,2*harmonicnum+1))
     for i in np.arange(1,harmonicnum+1):
         cycle[:,j]=np.sin(i*2*np.pi*t/period)
         cycle[:,j+1]=np.cos(i*2*np.pi*t/period)
         j=j+2
     return cycle

def lowpass(array):
     '''120-day low pass filter'''
     if array.ndim==1:
         array=np.expand_dims(array,1)
     N=array.shape[0]
     freqs=np.arange(N)/N
     periods=1./freqs
     I=np.where(periods>120)[0]
     fftarray=np.fft.fft(array,axis=0)
     filter=np.zeros_like(fftarray)
     filter[I,:]=fftarray[I,:]
     filter[-I,:]=fftarray[-I,:]
     filter[0,:]=fftarray[0,:]
     output=np.fft.ifft(filter,axis=0).real
     return(output.squeeze())
def bandpass(array):
     '''1.5-year to 50 year band pass filter, effectively a lowpass filter, I kept this format because I tried other filters to assess sensitivity'''
     N=array.shape[0]
     freqs=np.arange(N)/N
     periods=1./freqs
     I=np.where(np.logical_and(periods>365*1.5,periods<365*50))[0]
     fftarray=np.fft.fft(array,axis=0)
     filter=np.zeros_like(fftarray)
     filter[I,:]=fftarray[I,:]
     filter[-I,:]=fftarray[-I,:]
     filter[0,:]=0
     output=np.fft.ifft(filter,axis=0).real
     return(output)
X=harmbuild(uarray200.shape[0],harmonicnum)
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(uarray200-trend))
cycle200=X.dot(C)
uarray200small=lowpass(uarray200)
uarray200sm=uarray200small
uarray70sm=lowpass(uarray70)
interannual=bandpass(uarray200sm-cycle200-trend)

#vertical shear across the tropopause index over Indian Ocean
shear=uarray70sm[:,50:80].mean(axis=1)-(uarray200small[:,50:80]).mean(axis=1)

N=uarray70sm.shape[0]
N3=int(N/3)
RMMamp=lowpass(RMMamp)
RMMamppl=np.hstack((np.zeros(cutter),RMMamp))

#Create Figure 3: 
plt.figure(figsize=(8.5,11))
for i in np.arange(3):
     plt.subplot(3,1,i+1)
     plt.plot(dates[i*N3:(i+1)*N3],uarray70sm[i*N3:(i+1)*N3,50:80].mean(axis=1),'r',label='u70')
     plt.plot(dates[i*N3:(i+1)*N3],trend[i*N3:(i+1)*N3,50:80].mean(axis=1),'b',linestyle=':')
     plt.plot(dates[i*N3:(i+1)*N3],uarray200small[i*N3:(i+1)*N3,50:80].mean(axis=1),'b',label='u200')
     plt.plot(dates[i*N3:(i+1)*N3],trend[i*N3:(i+1)*N3,50:80].mean(axis=1)+interannual[i*N3:(i+1)*N3,50:80].mean(axis=1),'b',linestyle='-.')
     plt.plot(dates[i*N3:(i+1)*N3],trend[i*N3:(i+1)*N3,50:80].mean(axis=1)+cycle200[i*N3:(i+1)*N3,100:130].mean(axis=1),'b',linestyle='--')

     plt.plot(dates[i*N3:(i+1)*N3],shear[i*N3:(i+1)*N3],'k',linewidth=3,label='Shear')
     plt.plot(dates[i*N3:(i+1)*N3],RMMamppl[i*N3:(i+1)*N3]*6,'c',linewidth=3,label='RMM')
     if i==0:
          plt.title('Frequency Components of u200, u70, and 70-200 hPa \nVertical Shear of Zonal Wind, Indian Basin')
          plt.legend(loc='upper right')
     plt.grid(True)
     
     plt.ylabel('m/s')




print('The correlation coefficient is: ')
print(np.corrcoef(uarray70sm[:,50:80].mean(axis=1)-(interannual[:,50:80]+cycle200[:,50:80]+trend[:,50:80]).mean(axis=1),shear)[0,1])
print('RMMamppl.shape')
print(RMMamppl.shape)
print('The correlation coefficient between RMM amplitude and u70 is: ')
print(np.corrcoef(uarray70sm[:,50:80].mean(axis=1),RMMamppl[:uarray70sm.shape[0]])[0,1])
print('The correlation coefficient between RMM amplitude and shear is: ')
print(np.corrcoef(shear,RMMamppl[:shear.shape[0]])[0,1])
print('\n\n\n\n')

plt.grid(True)
plt.savefig('/pr11/roundy/public_html/examsheario.png')
print('Shear easterly amplitude: ')
cutter=(date(1974,6,1)-date(1974,1,1)).days
plt.figure(figsize=(8,5))
plt.plot(cycle200[:365,50:80].mean(axis=1)+trend[:,50:80].mean(axis=1).mean())
plt.plot([0,365],[-5,-5],'k')
plt.plot([0,365],[-15,-15],'k')
plt.xlabel('Day of Year')
plt.ylabel('m/s')
plt.title('Seasonal Cycle of Indian Basin 200 hPa Zonal Wind')
plt.axis([0,365,-22,0])
plt.grid(True)
plt.savefig('/pr11/roundy/public_html/examioseas.png')

a=np.percentile(shear,5)
b=np.percentile(shear,95)
a70=np.percentile(uarray70sm[:,50:80].mean(axis=1),5)
b70=np.percentile(uarray70sm[:,50:80].mean(axis=1),95)
shearnegamp=np.zeros(1000)
shearposamp=np.zeros(1000)
u70negamp=np.zeros(1000)
u70posamp=np.zeros(1000)
u70negampdjf=np.zeros(1000)
u70posampdjf=np.zeros(1000)
A=shear[cutter:]
B=uarray70sm[cutter:,50:80].mean(axis=1)
for bsnum in np.arange(1000):
     Irand=np.random.randint(0,A.shape[0],A.shape[0])
     I=np.where(A[Irand]<a)[0]
     shearnegamp[bsnum]=RMMamp[Irand[I]].mean()
     I=np.where(B[Irand]<a70)[0]
     u70negamp[bsnum]=RMMamp[Irand[I]].mean()

     I=np.where(np.logical_and(B[Irand]<a70,np.logical_or(monthseries[:B.shape[0]]<3,monthseries[:B.shape[0]]==12)))[0]
     u70negampdjf[bsnum]=RMMamp[Irand[I]].mean()

     I=np.where(A[Irand]>b)[0]
     shearposamp[bsnum]=RMMamp[Irand[I]].mean()
     I=np.where(B[Irand]>b70)[0]
     u70posamp[bsnum]=RMMamp[Irand[I]].mean()
     
     I=np.where(np.logical_and(B[Irand]>b70,np.logical_or(monthseries[:B.shape[0]]<3,monthseries[:B.shape[0]]==12)))[0]
     u70posampdjf[bsnum]=RMMamp[Irand[I]].mean()


#Create Figure 4
plt.figure()
plt.hist(shearnegamp,20,facecolor='black')
plt.hist(shearposamp,20,histtype='step',edgecolor='black')
plt.hist(u70negampdjf,20,facecolor='red')
plt.hist(u70posampdjf,20,histtype='step',edgecolor='red')
plt.hist(u70negamp,20,facecolor='blue')
plt.hist(u70posamp,20,histtype='step',edgecolor='blue')
plt.title('Histogram of RMM Amplitude by 70-200 hPa\n Vertical Shear and u70')
plt.grid(True)
plt.xlabel('RMM Amplitude')
plt.ylabel('Counts')
plt.savefig('/pr11/roundy/public_html/exam11.png')

easterlypermonth=np.zeros((2020-1974)*12)

plt.clf()
plt.plot(dates,shear)
plt.savefig('/pr11/roundy/public_html/examshear1io.png')

print('The 5th percentile is '+str(a))
A=shear[cutter:]
B=uarray70sm[cutter:,50:80].mean(axis=1)
for bsnum in np.arange(1000):
     Irand=np.random.randint(0,A.shape[0],A.shape[0])
     I=np.where(A[Irand]<a)[0]
     shearnegamp[bsnum]=RMMamp[Irand[I]].mean()
     I=np.where(B[Irand]<a70)[0]
     u70negamp[bsnum]=RMMamp[Irand[I]].mean()

     I=np.where(np.logical_and(B[Irand]<a70,np.logical_or(monthseries[:B.shape[0]]<3,monthseries[:B.shape[0]]==12)))[0]
     u70negampdjf[bsnum]=RMMamp[Irand[I]].mean()

     I=np.where(A[Irand]>b)[0]
     shearposamp[bsnum]=RMMamp[Irand[I]].mean()
     I=np.where(B[Irand]>b70)[0]
     u70posamp[bsnum]=RMMamp[Irand[I]].mean()
     
     I=np.where(np.logical_and(B[Irand]>b70,np.logical_or(monthseries[:B.shape[0]]<3,monthseries[:B.shape[0]]==12)))[0]
     u70posampdjf[bsnum]=RMMamp[Irand[I]].mean()



plt.figure()
plt.hist(shearnegamp,20,facecolor='black')
plt.hist(shearposamp,20,histtype='step',edgecolor='black')
plt.hist(u70negampdjf,20,facecolor='red')
plt.hist(u70posampdjf,20,histtype='step',edgecolor='red')
plt.hist(u70negamp,20,facecolor='blue')
plt.hist(u70posamp,20,histtype='step',edgecolor='blue')
plt.title('Histogram of RMM Amplitude by 70-200 hPa\n Vertical Shear and u70')
plt.grid(True)
plt.xlabel('RMM Amplitude')
plt.ylabel('Counts')
plt.savefig('/pr11/roundy/public_html/exam12.png')

easterlypermonth=np.zeros((2020-1974)*12)

plt.clf()
plt.plot(dates,shear)
plt.savefig('/pr11/roundy/public_html/examshear1io.png')
print('examshear1.png printed')

print('The 5th percentile is '+str(a))
for year in np.arange(1974,2020):
     for month in np.arange(12):
          Imonth=np.where(np.logical_and(yearseries==year,monthseries==month+1))[0]
          easterlypermonth[(year-1974)*12+month]=np.sum((shear[Imonth])<a)


plt.figure(figsize=(10,6))
yearindexes=np.arange(1979,2020)
yearseriesfracind=np.unique(yearseriesfracmonth)
plt.bar(yearseriesfracind[:easterlypermonth.shape[0]],easterlypermonth)
plt.plot(yearseriesfracmjo[:RMMamp.shape[0]],RMMamp*40,'k')
plt.axis([1974,2020,0,100])
plt.title('Number of 70-100 hPa Easterly Shear Days')
plt.ylabel('Number per Year')
plt.xlabel('Year')
plt.grid(True)
plt.savefig('/pr11/roundy/public_html/exameasterlyshear.png')



I=np.where(monthseries<13)[0]
p25=np.quantile(qbodly[I],.2)
p75=np.quantile(qbodly[I],.8)
arrayIOqboe=np.zeros((12,4))
arrayIOqbow=np.zeros((12,4))
ensolevels=np.zeros(2)
for monthi in np.arange(12):
     Imonth=np.where(np.logical_or(monthseries==slidermonth[monthi+1],np.logical_or(monthseries==slidermonth[monthi],monthseries==slidermonth[monthi+2])))[0]
     Imonthi=np.logical_or(monthseries==slidermonth[monthi+1],np.logical_or(monthseries==slidermonth[monthi],monthseries==slidermonth[monthi+2]))
     ensolevels=np.percentile(nino34dly[Imonth],[30,70])

     Ie=np.where(np.logical_and(qbodly<p25,np.logical_and(Imonthi,nino34dly<ensolevels[0])))[0]
     Iw=np.where(np.logical_and(qbodly>p75,np.logical_and(Imonthi,nino34dly<ensolevels[0])))[0]
     arrayIOqboe[monthi,0]=np.mean(uarraydiff[Ie,50:80])
     arrayIOqbow[monthi,0]=np.mean(uarraydiff[Iw,50:80])
     
     Ie=np.where(np.logical_and(qbodly<p25,np.logical_and(Imonthi,np.logical_and(nino34dly>ensolevels[0],nino34dly<ensolevels[1]))))[0]
     Iw=np.where(np.logical_and(qbodly>p75,np.logical_and(Imonthi,np.logical_and(nino34dly>ensolevels[0],nino34dly<ensolevels[1]))))[0]
     arrayIOqboe[monthi,1]=np.mean(uarraydiff[Ie,50:80])
     arrayIOqbow[monthi,1]=np.mean(uarraydiff[Iw,50:80])
     
     Ie=np.where(np.logical_and(qbodly<p25,np.logical_and(Imonthi,nino34dly>ensolevels[1])))[0]
     Iw=np.where(np.logical_and(qbodly>p75,np.logical_and(Imonthi,nino34dly>ensolevels[1])))[0]
     arrayIOqboe[monthi,2]=np.mean(uarraydiff[Ie,50:80])
     arrayIOqbow[monthi,2]=np.mean(uarraydiff[Iw,50:80])

     Ie=np.where(np.logical_and(qbodly<p25,Imonthi))[0]
     Iw=np.where(np.logical_and(qbodly>p75,Imonthi))[0]
     arrayIOqboe[monthi,3]=np.mean(uarraydiff[Ie,50:80])
     arrayIOqbow[monthi,3]=np.mean(uarraydiff[Iw,50:80])

#Create Figure 6

fig,ax=plt.subplots(figsize=(5,6),dpi=200)
V=np.arange(-30,31,1)
monthinds=np.arange(1,13)
#for monthi in np.arange(1,13):
#    monthinds[monthi-1,:]=monthi+1
#for ensoi in np.arange(3):
ensoinds=np.arange(4)
im=ax.pcolormesh(ensoinds,monthinds,arrayIOqboe,cmap='bwr',shading='nearest',vmin=-20,vmax=20)
plt.colorbar(im,label='70 hPa - 200 hPa Shear')
boundaries = [-0.5,0.5,1.5,2.5,3.5]
for x in boundaries:
     ax.axvline(x,color='black',linewidth=2,alpha= 0.9)

ax.grid(False)
labels=["La Nina","Neutral","El Nino","All"]
centers = [0.0,1.0,2.0,3.0]
for x, label, in zip(centers,labels):
     ax.text(x, -0.15, label,
             ha='center', va='top',
	     fontsize=11,fontweight='bold',
	     transform=ax.get_xaxis_transform())

ax.set_xlim(-0.5,3.5)
ax.set_ylim(-0.5,arrayIOqboe.shape[0] - 0.5)
ax.set_xticks([])
ax.set_xlabel('ENSO phase',labelpad=20)
ax.set_ylabel('Month')
ax.set_yticks(np.arange(1,13,1))
ax.set_ylim(1,12)
ax.set_title('70hPa-200 hPa Zonal Wind Shear, QBOe')
plt.tight_layout()
plt.savefig('/pr11/roundy/public_html/exam21io.png',dpi=300,bbox_inches='tight',facecolor='white')


plt.clf()
fig,ax=plt.subplots(figsize=(5,6),dpi=200)
V=np.arange(-30,31,1)
monthinds=np.arange(1,13)
#for monthi in np.arange(1,13):
#    monthinds[monthi-1,:]=monthi+1
#for ensoi in np.arange(3):
ensoinds=np.arange(4)
im=ax.pcolormesh(ensoinds,monthinds,arrayIOqbow,cmap='bwr',shading='nearest',vmin=-20,vmax=20)
plt.colorbar(im,label='70 hPa - 200 hPa Shear')
boundaries = [-0.5,0.5,1.5,2.5,3.5]
for x in boundaries:
     ax.axvline(x,color='black',linewidth=2,alpha= 0.9)

ax.grid(False)

labels=["La Nina","Neutral","El Nino","All"]
centers = [0.0,1.0,2.0,3.0]
for x, label, in zip(centers,labels):
     ax.text(x, -0.15, label,
             ha='center', va='top',
	     fontsize=11,fontweight='bold',
	     transform=ax.get_xaxis_transform())

ax.set_xlim(-0.5,3.5)
ax.set_ylim(-0.5,arrayIOqbow.shape[0] - 0.5)
ax.set_xticks([])
ax.set_xlabel('ENSO phase',labelpad=20)
ax.set_ylabel('Month')
ax.set_yticks(np.arange(1,13,1))
ax.set_ylim(1,12)
ax.set_title('70hPa-200 hPa Zonal Wind Shear, QBOw')
plt.tight_layout()
plt.savefig('/pr11/roundy/public_html/exam22io.png',dpi=300,bbox_inches='tight',facecolor='white')



