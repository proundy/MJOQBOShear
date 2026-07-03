import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date,timedelta
from scipy import io as io
from scipy.signal import detrend as detrend
mpl.rcParams['contour.negative_linestyle'] = 'dashed'
mpl.rcParams.update({'font.size': 13})
RMM=np.loadtxt('/roundylab_rit/roundy/RMMindex.txt')

"""Code for Figures 1 and 7 of Roundy 2026 The Role of Stratosphere-Troposphere Vertical Shear of the Zonal Wind in 
the QBO-MJO Relationship.  You will need to have previously downloaded the ERA5 zonal wind and geopotential datasets.
This code loads those previously downloaded datasets from the copernicus climate data server. Linked code in the paper
shows how this is done. You can modify the script to load your own files, format should be time x longitude, with values 
averaged to 1 degree and daily resolution (modify the meridional averaging components below to suit your data."""


uarray=np.load('/roundylab_rit/roundy/era5/u100.npy')
warray=np.load('/roundylab_rit/roundy/era5/w100.npy')#vertical wind data are loaded and processed here but not used in this paper

d0=date(1979,1,1)  #My full vertical cross section data begin in 1979.
d1=date(2020,12,31)


dates=np.array([d0+timedelta(days=int(n)) for n in np.arange((d1-d0).days)])
monthseries=np.array([d.month for d in dates])

I=np.where(monthseries==12)[0]
monthseries[I]=0 #For convenience, set December to zero, allows a single test to extract values from March and earlier. 
I=np.where(monthseries==11)[0]
monthseries[I]=-1 #For convenience, set November to -1, same reason as above

qbo=np.load('/roundylab_rit/roundy/era5/u5n5sglob70.npy')#[:,16,:]
u200=np.load('/roundylab_rit/roundy/era5/u200.npy')#[:,16,:]
qbo=np.concatenate((qbo[:,180:],qbo[:,:180]),axis=1)
u200=np.concatenate((u200[:,180:],u200[:,:180]),axis=1)
Y=np.arange(-15,16) #These data are subset already to 15N to 15S
Iy=np.where(np.abs(Y)<=5)[0] #Select 5N to 5S
u200=u200[:,Iy,:].mean(axis=1) #Take meridional average
qbo=qbo-u200  #The variable QBO is here assigned the value of the 70-200 hPa vertical shear of the zonal wind
qbo=qbo[:,50:90].mean(axis=1)
qbosm=qbo.copy()

#Take 101-day centered moving average
smoother=50
for t in np.arange(smoother,qbosm.shape[0]-smoother):
     qbo[t]=qbosm[t-smoother:t+smoother].mean()
qbo[:smoother]=0
qbo[qbo.shape[0]-smoother:]=0
qbodly=qbo.copy()



plt.figure()
plt.plot(qbodly)
plt.savefig('/pr11/roundy/public_html/exam.png')
I=np.where(monthseries<4)[0]
p1=np.quantile(qbodly[I],.2)
p9=np.quantile(qbodly[I],.8)

Ie=np.where(np.logical_and(qbodly<p1,monthseries<4))[0]
Iw=np.where(np.logical_and(qbodly>p9,monthseries<4))[0]


N=uarray.shape[0]

uarray=np.concatenate((uarray[:,:,180:],uarray[:,:,:180]),axis=2)
warray=np.concatenate((warray[:,:,180:],warray[:,:,:180]),axis=2)

cutter=(date(1979,1,1)-date(1974,6,1)).days
RMM=RMM[cutter:,:]
baseindex=RMM[:,4:5]
OLRarray=np.load('/roundylab_rit/roundy/projections/olrbig.npy') #OLR data are loaded here but not used in the figures for this paper
OLRarray=OLRarray[cutter:,:,:]

Yolr=np.arange(-30,32.5,2.5)

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
Y=np.arange(-15,16,1)
Iy=np.where(np.abs(Y)<=10)[0]


def lowpass(array):
     '''100-day low pass Fourier filter.'''
     N=array.shape[0]
     freqs=np.arange(N)/N
     periods=1./freqs
     I=np.where(periods>100)[0]
     fftarray=np.fft.fft(array,axis=0)
     filter=np.zeros_like(fftarray)
     filter[I,:]=fftarray[I,:]
     filter[-I,:]=fftarray[-I,:]
     filter[0,:]=fftarray[0,:]
     output=np.fft.ifft(filter,axis=0).real
     return(output)




uarray=uarray[:,Iy,:].mean(axis=1)
warray=warray[:,Iy,:].mean(axis=1)
A=uarray.copy()
X=harmbuild(A.shape[0],harmonicnum)
X=X.squeeze()
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(A))
A=A-X.dot(C)
Yolr=np.arange(-30,32.5,2.5)
Iyolr=np.where(np.abs(Yolr)<=10)[0]
OLRarray=OLRarray[:,Iyolr,:].mean(axis=1)
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(OLRarray[:uarray.shape[0],:]))
OLRarray=OLRarray[:uarray.shape[0],:]-X.dot(C)


cut=(date(2020,12,31)-date(1979,1,1)).days
plt.figure(figsize=(10,30))
plt.contourf(A,20)
plt.savefig('/pr11/roundy/public_html/exam10.png')
plt.close()

signs=np.array((-1,1))

def randphase(data):
     fftout=np.fft.fft(data)
     power=(fftout*np.conj(fftout)).real
     randfft=np.zeros_like(fftout)
     for f in np.arange(1,int(power.shape[0]/2)):
          realval=(np.random.random()*power[f])
          imagval=power[f]-realval
          randfft[f]=np.sqrt(realval)*np.random.choice(signs)+np.sqrt(imagval)*np.random.choice(signs)*1j
          randfft[-f]=np.conj(randfft[f])
     return np.fft.ifft(randfft).real

#Create 1000 indexes for the QBO with randomized Fourier phases for null distribution in significance test
randqbo=np.zeros((qbo.shape[0],1000))
for i in np.arange(1000):
     randqbo[:,i]=randphase(qbo)


levs=np.array([300,250,225,200,175,150,125,100])
levsstrat=np.array([70,50]) # I had to extend my previous troposphere only datasets to include the stratosphere
levsall=np.array([300,250,225,200,175,150,125,100,70,50])
levsallmap=np.array([300,250,225,200,175,150,125,100,70,60,50])
levsallstr=np.array(['300','250','225','200','175','150','125','100','70',' ','50'])

levssub=np.array([300,250,225,175,150,125]) #Subset for labels
Tarray=np.zeros((N,levs.shape[0]))
uarray=np.zeros((N,360,levs.shape[0]))
zarray=np.zeros((N,360,levs.shape[0]))
warray=np.zeros((N,levs.shape[0]))
for levnum in np.arange(levs.shape[0]):
     temp=np.load('/roundylab_rit/roundy/era5/u'+str(levs[levnum])+'.npy')
     temp=np.concatenate((temp[:,:,180:],temp[:,:,:180]),axis=2)
     temp=temp[:,Iy,:].mean(axis=1)
     uarray[:,:,levnum]=temp[:,:]
     
     temp=np.load('/roundylab_rit/roundy/era5/ght'+str(levs[levnum])+'.npy')/9.8
     temp=np.concatenate((temp[:,:,180:],temp[:,:,:180]),axis=2)
     temp=temp[:,Iy,:].mean(axis=1)
     zarray[:,:,levnum]=temp[:,:]
     temp=np.load('/roundylab_rit/roundy/era5/w'+str(levs[levnum])+'.npy')
     temp=temp[:,Iy,:].mean(axis=1)
     warray[:,levnum]=temp[:,70]
     if levs[levnum]==100 or levs[levnum]==200: 
          temp=np.load('/roundylab_rit/roundy/era5/T'+str(levs[levnum])+'.npy')
          temp=np.concatenate((temp[:,:,180:],temp[:,:,:180]),axis=2)
          temp=temp[:,Iy,:].mean(axis=1)
          Tarray[:,levnum]=temp[:,70]

for lev in levssub:
     I=np.where(levs==lev)[0][0]
     temp=np.load('/roundylab_rit/roundy/era5/T70e_'+str(lev)+'.npy')
     Tarray[:,I]=temp[:,Iy].mean(axis=1)

#create stratospheric data arrays and fill them
uarraystrat=np.zeros((N,360,levsstrat.shape[0]))
zarraystrat=np.zeros((N,360,levsstrat.shape[0]))
warraystrat=np.zeros((N,levsstrat.shape[0]))
Tarraystrat=np.zeros((N,levsstrat.shape[0]))
for levnum in np.arange(levsstrat.shape[0]):
     if levnum==0:
          temp=np.load('/roundylab_rit/roundy/era5/u5n5sglob70.npy')
          uarraystrat[:,:,levnum]=temp[:,:]
          temp=np.load('/roundylab_rit/roundy/era5/ght5n5sglob70.npy')/9.8
          zarraystrat[:,:,levnum]=temp[:,:]
     else:
          temp=np.load('/roundylab_rit/roundy/era5/u5n5sglob50.npy')
          uarraystrat[:,:,levnum]=temp[:,:]
          temp=np.load('/roundylab_rit/roundy/era5/ght5n5sglob50.npy')/9.8
          zarraystrat[:,:,levnum]=temp[:,:]

     
     temp=np.load('/roundylab_rit/roundy/era5/T70e_'+str(levsstrat[levnum])+'.npy')
     Tarraystrat[:,levnum]=temp[:,Iy].mean(axis=1)
     
    
     temp=np.load('/roundylab_rit/roundy/era5/w70e_'+str(levsstrat[levnum])+'.npy')
     warraystrat[:,levnum]=temp[:,Iy].mean(axis=1)


uarray=np.concatenate([uarray,uarraystrat],axis=2)
zarray=np.concatenate([zarray,zarraystrat],axis=2)

uarraylow=lowpass(uarray)
zarraylow=lowpass(zarray)


#Remove mean and seasonal cycle.
L=uarray.shape[0]
C=np.linalg.inv(X[:L,:].T.dot(X[:L,:])).dot((X[:L,:].T.dot(uarray.transpose(1,0,2))).transpose(1,0,2))
uarray=uarray-(X[:L,:].dot(C.transpose(1,0,2)))


C=np.linalg.inv(X[:L,:].T.dot(X[:L,:])).dot(X[:L,:].T.dot(zarray.transpose(1,0,2)).transpose(1,0,2))
zarray=zarray-X[:L,:].dot(C.transpose(1,0,2))


uarray=uarray-lowpass(uarray) #Remove the low frequency background condition so fractured low frequency variability doesn't project to subseasonal signal
zarray=zarray-lowpass(zarray)
warray=warray-lowpass(warray)

maxlag=35 #number of lead times in days to retain

compulow=np.zeros((2*maxlag+1,uarray.shape[1]))
compzlow=np.zeros((2*maxlag+1,uarray.shape[1]))


def regressor(X,Y,I):
     output=np.zeros((2*maxlag+1,Y.shape[1],Y.shape[2]))
     #I=np.arange(maxlag,Y.shape[0]-maxlag)
     Iin=np.where(np.logical_and(I>maxlag,I<Y.shape[0]-maxlag))[0]
     I=I[Iin]
     for lag in np.arange(-maxlag,maxlag+1):
          C=np.linalg.inv(X[I,:].T.dot(X[I,:])).dot((X[I,:].T.dot(Y[I+lag,:].transpose(1,0,2)).transpose(1,0,2)))
          output[lag+maxlag,:,:]=-2*np.std(X,axis=0).dot(C.transpose(1,0,2))
          #output[lag+maxlag,:,:]=-2*np.std(X,axis=0).dot(C)
     return output
lags=np.arange(-maxlag,maxlag+1)

def regressorbs(X,Y,rmm1=baseindex[:,0]):
     '''Repeat regression but in Monte-Carlo mode, using scrambled low frequency background shear or QBO index.''' 
     bnum=1000 #run 1000 times
     outpute=np.zeros((2*maxlag+1,Y.shape[1],Y.shape[2],bnum))
     outputw=np.zeros((2*maxlag+1,Y.shape[1],Y.shape[2],bnum))
     I=np.arange(Y.shape[0])
     Iin=np.where(np.logical_and(I>maxlag,I<Y.shape[0]-maxlag))[0]
     I=I[Iin]
     XX=np.expand_dims(rmm1,1)
     for bsnum in np.arange(bnum):
          print(bsnum)
          p1=np.quantile(X[:,bsnum],.2)
          p9=np.quantile(X[:,bsnum],.8)
          Ie=I[np.where(np.logical_and(X[I,bsnum]<p1,monthseries[I]<4))[0]]
          Iw=I[np.where(np.logical_and(X[I,bsnum]>p9,monthseries[I]<4))[0]]
          for lag in np.arange(-maxlag,maxlag+1):
               C=np.linalg.inv(XX[Ie,:].T.dot(XX[Ie,:])).dot((XX[Ie,:].T.dot(Y[Ie+lag,:,:].transpose(1,0,2)).transpose(1,0,2)))
               outpute[lag+maxlag,:,:,bsnum]=-2*np.std(XX,axis=0).dot(C.transpose(1,0,2))
               C=np.linalg.inv(XX[Iw,:].T.dot(XX[Iw,:])).dot((XX[Iw,:].T.dot(Y[Iw+lag,:,:].transpose(1,0,2)).transpose(1,0,2)))
               outputw[lag+maxlag,:,:,bsnum]=-2*np.std(XX,axis=0).dot(C.transpose(1,0,2))
     return outpute,outputw

x=np.arange(360)
xolr=np.arange(0,360,2.5)
plt.figure(figsize=(5,10))
Vu=np.arange(-30,31,1)
Vz=np.arange(-15,16,1)
Ilag=np.arange(1,2*maxlag)
plt.figure(9,figsize=(10,5))
plt.figure(10,figsize=(10,5))
plt.figure(11,figsize=(10,5))
plt.figure(12,figsize=(10,5))
plt.figure(13,figsize=(10,5))
plt.figure(14,figsize=(10,5))
plt.figure(15,figsize=(10,5))
plt.figure(16,figsize=(10,5))
letters=np.array(('a. ','b. ','c. ','d. ','e. ','f.','g. '))

lags=np.arange(-maxlag,maxlag+1)
x=np.arange(360)

#indexes for quiver plot
xx=np.zeros((2*maxlag+1,levsall.shape[0]))
tt=np.zeros((2*maxlag+1,levsall.shape[0]))
for t in np.arange(xx.shape[0]):
     xx[t,:]=levsall
for xi in np.arange(tt.shape[1]):
     tt[:,xi]=lags

Imonth=np.where(monthseries<3)[0]

def lowcomposite(array,Ie=Ie,Iw=Iw):
     compositee=np.zeros((array.shape[1],array.shape[2]))
     compositew=np.zeros((array.shape[1],array.shape[2]))
     
     compositee[:,:]=np.mean(array[Ie,:,:],axis=0)
     compositew[:,:]=np.mean(array[Iw,:,:],axis=0)

     return compositee,compositew


VT=np.arange(-1.2,1.4,.2)
plt.figure(5,figsize=(8,4))
Vw=np.linspace(-.002,.002,10)
f=0
Vu=np.arange(-20,21,1)
Vz=np.arange(-24,27,3)
print(f)
reguw=regressor(baseindex,uarray,Iw)
regzw=regressor(baseindex,zarray,Iw)
regue=regressor(baseindex,uarray,Ie)

regze=regressor(baseindex,zarray,Ie)

#regzbse,regzbsw=regressorbs(randqbo,zarray)
#np.savez('bsoutputz.npz',regzbse=regzbse,regzbsw=regzbsw)
F=np.load('bsoutputz.npz')
regzbse=F['regzbse']
regzbsw=F['regzbsw']
regzdiffbs=np.sort((regzbse-regzbsw),axis=3)[:,:,:,[25,975]]



#regubse,regubsw=regressorbs(randqbo,uarray)
#np.savez('bsoutputu.npz',regubse=regubse,regubsw=regubsw)
F=np.load('bsoutputu.npz')
regubse=F['regubse']
regubsw=F['regubsw']

regudiffbs=np.sort((regubse-regubsw),axis=3)[:,:,:,[25,975]]

print(regubse.shape)
print(regudiffbs.max())
print(regudiffbs.min())
print(regzdiffbs.max())
print(regzdiffbs.min())


'''
regzebs=regressorbs(baseindex,zarray,Ie)
regzwbs=regressorbs(baseindex,zarray,Iw)
'''


#reguebs=regressorbs(baseindex,uarray,Ie)
#reguwbs=regressorbs(baseindex,uarray,Iw)



'''
regzwbs=np.sort(regzwbs,axis=2)[:,:,[5,995]]
regzebs=np.sort(regzebs,axis=2)[:,:,[5,995]]
testmap=np.logical_or(regzebs[:,:,1]<regzwbs[:,:,0],regzebs[:,:,0]>regzwbs[:,:,1])
'''

compositeue,compositeuw=lowcomposite(uarraylow)

diffu=regue-reguw
usig=np.logical_or(diffu<regudiffbs[:,:,:,0],diffu>regudiffbs[:,:,:,1])
diffz=regze-regzw
zsig=np.logical_or(diffz<regzdiffbs[:,:,:,0],diffz>regzdiffbs[:,:,:,1])
plt.figure()
plt.contour(usig[:,50,:].squeeze().T,levels=[0.5],colors='k')
plt.savefig('/pr11/roundy/public_html/exam15.png')

plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
V=np.arange(-28,30,2)
lons=np.arange(360)


plt.contourf(lons,levsall,compositeue.T,V,cmap='bwr')
plt.xticks(np.arange(0,360,45),'')
plt.gca().invert_yaxis()
plt.gca().set_yscale('log')
plt.yticks(levsallmap,levsallstr)
plt.title('a. QBOe Mean Zonal Wind')
plt.ylabel('Pressure (hPa)')
plt.colorbar()
plt.subplot(2,1,2)
plt.contourf(lons,levsall,compositeuw.T,V,cmap='bwr')
plt.gca().invert_yaxis()
plt.gca().set_yscale('log')
plt.yticks(levsallmap,levsallstr)
plt.xticks(np.arange(0,360,45))
plt.title('b. QBOw Mean Zonal Wind')
plt.ylabel('Pressure (hPa)')
plt.xlabel('Longitude (Degrees East)')
plt.colorbar()
plt.savefig('/pr11/roundy/public_html/exam10.png')
plt.savefig('/pr11/roundy/public_html/exammeanxsectshear.eps',format='eps')
lons=np.arange(40,220,30)
plt.figure(figsize=(8.5,15))
letters=['a. ','b.','c. ','d. ','e. ','f. ','g. ','h. ','i. ','j. ','k. ','l. ']
for x in np.arange(lons.shape[0]*2):
     if x<6:
          print(lons[x])
     else:
          print(lons[x-6])
     if x%2==0:
          plt.subplot(6,2,int(x+1))
          plt.contourf(lags,levsall,regze[:,lons[int(x/2)],:].T,Vz,cmap='bwr')
          plt.colorbar()
          plt.contour(lags,levsall,usig.squeeze()[:,lons[int(x/2)],:].T,[0.5],colors='r')
          plt.contour(lags,levsall,regue[:,lons[int(x/2)],:].T,Vu[Vu>0],colors='k')
          plt.contour(lags,levsall,regue[:,lons[int(x/2)],:].T,Vu[Vu<0],colors='k',linestyle='--')
          plt.title(letters[int(x/2)]+'QBOe, '+str(lons[int(x/2)])+'E')
          plt.ylabel('hPa')
          if int(x/2)+1==6:
               plt.xticks(np.arange(-30,40,10))
               plt.xlabel('Time Lag (Days)')
          else: 
               plt.xticks(np.arange(-30,40,10),'')
          plt.gca().set_yscale('log')
          plt.yticks(levsallmap,levsallstr)
          plt.axis((-30,30,50,300))
          plt.gca().invert_yaxis()
      
     if x%2==1:
          plt.subplot(6,2,int(x+1))
          plt.contourf(lags,levsall,regzw[:,lons[int(x/2)],:].T,Vz,cmap='bwr')
          plt.colorbar()
          plt.contour(lags,levsall,usig.squeeze()[:,lons[int(x/2)],:].T,[0.5],colors='r')
          plt.contour(lags,levsall,reguw[:,lons[int(x/2)],:].T,Vu[Vu>0],colors='k')
          plt.contour(lags,levsall,reguw[:,lons[int(x/2)],:].T,Vu[Vu<0],colors='k',linestyle='--')
          plt.title(letters[int((x-1)/2)+6]+'QBOw, '+str(lons[int((x-1)/2)])+'E')
          if (x-1)/2+1==6:
               plt.xlabel('Time Lag (Days)')
               plt.xticks(np.arange(-30,40,10))
          else:
               plt.xticks(np.arange(-30,40,10),'')

          plt.gca().set_yscale('log')

          plt.yticks(levsallmap,'')
          plt.axis((-30,30,50,300))
          plt.gca().invert_yaxis()
     
plt.savefig('/pr11/roundy/public_html/examzuermmglobshear.png')
plt.savefig('/pr11/roundy/public_html/examzuermmglobshear.eps',format='eps')

