import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date,timedelta
from scipy import io as io
from scipy.signal import detrend as detrend

'''Finds Figures 8 and 9 of Roundy (2026, Climate, on the MJO/QBO relationship.'''

mpl.rcParams['contour.negative_linestyle'] = 'dashed'
mpl.rcParams.update({'font.size': 13})

#Use data from 1979 forward to 2020
cutter=(date(1979,1,1)-date(1974,6,1)).days
RMM=np.loadtxt('/roundylab_rit/roundy/RMMindex.txt')[cutter:,:] #BOM RMM data
Yolr=np.arange(-30,32.5,2.5) #My OLR data subset is 30S to 30N
Iy=np.where(np.abs(Yolr)<=7.5)[0]#Average signals across the equator

olr=np.load('/roundylab_rit/roundy/projections/olrbig.npy') #OLR data array
olr=olr[:,Iy,:].mean(axis=1)
olrsm=olr.copy()
for t in np.arange(1,olr.shape[0]-1):
     olr[t,:]=olrsm[t-1:t+1,:].mean(axis=0)

Xolr=np.arange(0,360,2.5)
Idiff=np.arange(1,olr.shape[0]-1)
olrdiff=np.zeros_like(olr)
olrdiff[Idiff,:]=olr[Idiff+1,:]-olr[Idiff-1,:]
olr=olr[cutter:,:]
olrdiff=olrdiff[cutter:,:]
Xera=np.arange(360)

baseindex=RMM[:,4:5]

uarray=np.load('/roundylab_rit/roundy/era5/u5n5sglob70.npy') #Load 70 hPa u wind data, from 5N to 5S
zarray=np.load('/roundylab_rit/roundy/era5/ght5n5sglob70.npy')#geopotential data
d0=date(1979,1,1)
d1=date(2020,12,31)
olr=olr[:uarray.shape[0],:]
olrdiff=olrdiff[:uarray.shape[0],:]

def olrretract(uarray,olr=olr,olrdiff=olrdiff):
    '''Regresses out the OLR associated signal from the 70 hPa zonal wind.'''
    Im=np.where(qbo<0)[0]
    Ip=np.where(qbo>=0)[0]
    uadjusted=np.zeros_like(uarray)
    Xqbo=np.ones((uarray.shape[0],2))
    Xqbo[:,1]=qbo
    for x in np.arange(360):
         Ix=np.argmin(np.abs(x-Xolr))
         XX=np.ones((uarray.shape[0],3))
         XX[:,1]=olr[:,Ix]
         XX[:,2]=olrdiff[:,Ix]
	 
         varolr=olr[:,Ix]**2
         covolr=olr[:,Ix]*uarray[:,x]
         Yvar=np.ones((varolr.shape[0],1))
         Yvar[:,0]=varolr
         Ycov=np.ones((covolr.shape[0],1))
         Ycov[:,0]=covolr
         Cvar=np.linalg.inv(Xqbo.T.dot(Xqbo)).dot(Xqbo.T.dot(Yvar))
         varseries=Xqbo.dot(Cvar)
         Ccov=np.linalg.inv(Xqbo.T.dot(Xqbo)).dot(Xqbo.T.dot(Ycov))
         covseries=Xqbo.dot(Ccov)
         slopeseries=covseries.squeeze()/varseries.squeeze()
         
         varolr=olrdiff[:,Ix]**2
         covolr=olrdiff[:,Ix]*uarray[:,x]
         Yvar=np.ones((varolr.shape[0],1))
         Yvar[:,0]=varolr
         Ycov=np.ones((covolr.shape[0],1))
         Ycov[:,0]=covolr
         Cvar=np.linalg.inv(Xqbo.T.dot(Xqbo)).dot(Xqbo.T.dot(Yvar))
         varseries=Xqbo.dot(Cvar)
         Ccov=np.linalg.inv(Xqbo.T.dot(Xqbo)).dot(Xqbo.T.dot(Ycov))
         covseries=Xqbo.dot(Ccov)
         slopeseriesdiff=covseries.squeeze()/varseries.squeeze()

         uadjusted[:,x]=uarray[:,x]-olr[:,Ix]*slopeseries.squeeze()-olrdiff[:,Ix]*slopeseriesdiff.squeeze()
    return uadjusted

dates=np.array([d0+timedelta(days=int(n)) for n in np.arange((d1-d0).days)])
monthseries=np.array([d.month for d in dates])

I=np.where(monthseries==12)[0]
monthseries[I]=0 #For convenience, set December to zero
I=np.where(monthseries==11)[0]
monthseries[I]=-1 #For convenience, set November to -1

qbo=np.load('/roundylab_rit/roundy/era5/u5n5sglob70.npy')#[:,16,:]
qbo=qbo.mean(axis=1)
qbosm=qbo.copy()
#Do 101-day centered moving average to smooth out subseasonal variability
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
p25=np.quantile(qbodly[I],.2)
p75=np.quantile(qbodly[I],.8)

Ie=np.where(np.logical_and(qbodly<p25,monthseries<4))[0]
print(Ie.shape)
Iw=np.where(np.logical_and(qbodly>p75,monthseries<4))[0]
print(Iw.shape)
print('qboe mean is '+str(qbo[Ie].mean()))
print('qbow mean is '+str(qbo[Iw].mean()))
print('qbo mean is '+str(qbo.mean()))


N=uarray.shape[0]

def speedline(ybase,S,color='k'):
     '''Draw a line with slop determined by given speed S.'''
     x2=359.
     for y1 in np.arange(0,700,20):
          y2=x2/S*2*np.pi*6.37e6/(86400*360)+y1
          plt.plot([0,359],[dates[y1+ybase],dates[int(y2+ybase)]],color=color)


uarray=np.concatenate((uarray[:,180:],uarray[:,:180]),axis=1)
zarray=np.concatenate((zarray[:,180:],zarray[:,:180]),axis=1)


uarrayall=uarray.copy()

def cycrem(array):
     '''Remove seasonal cycle including 4 harmonics.'''
     harmonicnum=4
     days=array.shape[0]
     cycle = np.ones((days, harmonicnum*2+1), dtype = 'float64')
     j = 1
     t=np.arange(days)
     period = 365.25
     for i in np.arange(1,harmonicnum+1):
         cycle[:,j] = np.sin(i*2*np.pi*t/period)
         cycle[:,j+1] = np.cos(i*2*np.pi*t/period)
         j = j + 2
     
     X=cycle
     C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(array))
     anomaly=array-X.dot(C)

     return anomaly
zarray=cycrem(zarray)
uarrayall=cycrem(uarrayall)


uarrayminusolr=olrretract(uarrayall)

print('olrretract done')



def filter(array):

     '''Fourier filter for 5-100 days.'''
     periods=array.shape[0]/np.arange(array.shape[0])
     fftarray=np.fft.fft(array,axis=0)
     filterarray=np.zeros_like(fftarray)
     I=np.where(np.logical_and(periods>=5,periods<=100))[0]
     filterarray[I,:]=fftarray[I,:]
     filterarray[-I,:]=fftarray[-I,:]
     filterarray=np.fft.ifft(filterarray,axis=0).real
     return filterarray

zarraysub=filter(zarray)
uarraysub=filter(uarrayall)
uarrayminusolrsub=filter(uarrayminusolr)

plt.figure(figsize=[8.5,40])
X=np.arange(360)
plt.contourf(X,dates[4500:5200],uarraysub[4500:5200,:],20,cmap='bwr')
plt.contour(X,dates[4500:5200],zarraysub[4500:5200,:],20,colors='k')
speedline(4500,18,color='k')
speedline(4500,5,color='r')
plt.savefig('/pr11/roundy/public_html/exam1.png')
plt.clf()
X=np.arange(360)
plt.contourf(X,dates[4500:5200],uarrayminusolrsub[4500:5200,:],20,cmap='bwr')
plt.contour(X,dates[4500:5200],zarraysub[4500:5200,:],20,colors='k')
speedline(4500,18,color='k')
speedline(4500,5,color='r')
plt.savefig('/pr11/roundy/public_html/exam2.png')


uarray[:,160:]=0
uarray[:,:30]=0
def compbs(X,Y):
     '''Composite bootstrap test...''' 
     bnum=1000
     outpute=np.zeros((Y.shape[1],Y.shape[2],bnum))
     outputw=np.zeros((Y.shape[1],Y.shape[2],bnum))
     for bsnum in np.arange(1000):
         print(bsnum)
         p25=np.quantile(X[:,bsnum],.2)
         p75=np.quantile(X[:,bsnum],.8)
         Ie=np.where(np.logical_and(X[I,bsnum]<p25,monthseries[I]<4))[0]
         Iw=np.where(np.logical_and(X[I,bsnum]>p75,monthseries[I]<4))[0]
         
         outpute[:,:,bsnum]=Y[Ie,:,:].mean(axis=0)
         outputw[:,:,bsnum]=Y[Iw,:,:].mean(axis=0)
     diff=outpute-outputw
     output=np.sort(diff,axis=2)
     output=output[:,:,[5,995]]

     return output



def specmake(array):
     '''Create spectra of many overlapping time segments of longitude-time data.'''
     N=99
     half=int((N-1)/2)
     t=np.arange(99)
     spectra=np.zeros((array.shape[0],N,360))
     cosbell=np.expand_dims((1-np.cos(2*np.pi*t/N))/2,1)
     cosbell=np.tile(cosbell,360) 
     for i in np.arange(half,array.shape[0]-half-1):

          segment=detrend(array[i-half:i+half+1,:],axis=0)*cosbell
          fftseg=np.fft.fftshift(np.fft.fft2(segment))
          spectra[i,:,:]=(fftseg*np.conj(fftseg)).real
     return spectra


uspec=specmake(uarray)
uspecmolr=specmake(uarrayminusolr)

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
randqbo=np.zeros((qbo.shape[0],1000))
for i in np.arange(1000):
     randqbo[:,i]=randphase(qbo)



uspecmean=np.mean(uspec,axis=0)
'''I run the functions, then store the output so that I don't need to run it more than once. Uncomment these to run for the first time:'''
#specbs=compbs(randqbo,uspec)
#specbsmolr=compbs(randqbo,uspecmolr)
#np.save('specbs.npy',specbs)
#np.save('specbsmolr.npy',specbs)

specbs=np.load('specbs.npy')
specbsmolr=np.load('specbsmolr.npy')
uspecmeanw=np.mean(uspec[Iw,:,:],axis=0)
uspecmeane=np.mean(uspec[Ie,:,:],axis=0)
uspecolrmean=np.mean(uspecmolr[:,:,:],axis=0)
uspecmolrw=np.mean(uspecmolr[Iw,:,:],axis=0)
uspecmolre=np.mean(uspecmolr[Ie,:,:],axis=0)
usig=np.logical_or(uspecmolre-uspecmolrw<specbs[:,:,0],uspecmolre-uspecmolrw>specbsmolr[:,:,1])


'''Plot the power spectrum data:'''

freqs=np.arange(-98/2,98/2+1)/99.
X=-np.arange(-180,180)

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.contourf(X,freqs,np.log(uspecmean),20,cmap='bwr')
plt.ylabel('Frequency (Cycles per Day)')
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.axis((-15,15,0,0.4))
plt.title('a. 70 hPa u Spectrum')
plt.grid(True)
plt.colorbar()
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.axis((-15,15,0,0.4))
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')
plt.subplot(2,2,2)
plt.contourf(X,freqs,np.log(uspecmeanw),20,cmap='bwr')
plt.axis((-15,15,0,0.4))
plt.title('b. 70 hPa u Spectrum, QBOw')
plt.grid(True)
plt.colorbar()
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')
plt.subplot(2,2,3)
plt.contourf(X,freqs,np.log(uspecmeane),20,cmap='bwr')
plt.ylabel('Frequency (Cycles per Day)')
plt.xlabel('Westward Wavenumber  Eastward')
plt.axis((-15,15,0,0.4))
plt.title('c. 70 hPa u Spectrum, QBOe')
plt.grid(True)
plt.colorbar()
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')


uspecdiff=np.log(uspecmeane)-np.log(uspecmeanw)
plt.subplot(2,2,4)
plt.contourf(X,freqs,uspecdiff,20,cmap='bwr')
plt.xlabel('Westward Wavenumber  Eastward')
plt.axis((-15,15,0,0.4))
plt.title('d. 70 hPa u Spectrum, QBOe-QBOw')
plt.grid(True)
plt.colorbar()
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.savefig('/pr11/roundy/public_html/exam3paper.png')
plt.savefig('/pr11/roundy/public_html/exam3paper.eps',format='eps')



plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.contourf(X,freqs,np.log(uspecolrmean),20,cmap='bwr')
plt.ylabel('Frequency (Cycles per Day)')
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.axis((-15,15,0,0.4))
plt.title('a. 70 hPa u Spectrum')
plt.grid(True)
plt.colorbar()
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.axis((-15,15,0,0.4))
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')
plt.subplot(2,2,2)
plt.contourf(X,freqs,np.log(uspecmolrw),20,cmap='bwr')
plt.axis((-15,15,0,0.4))
plt.title('b. 70 hPa u Spectrum, QBOw')
plt.grid(True)
plt.colorbar()
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')
plt.subplot(2,2,3)
plt.contourf(X,freqs,np.log(uspecmolre),20,cmap='bwr')
plt.ylabel('Frequency (Cycles per Day)')
plt.xlabel('Westward Wavenumber  Eastward')
plt.axis((-15,15,0,0.4))
plt.title('c. 70 hPa u Spectrum, QBOe')
plt.grid(True)
plt.colorbar()
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')


uspecdiff=np.log(uspecmolre)-np.log(uspecmolrw)
plt.subplot(2,2,4)
plt.contourf(X,freqs,uspecdiff,20,cmap='bwr')
plt.xlabel('Westward Wavenumber  Eastward')
plt.axis((-15,15,0,0.4))
plt.title('d. 70 hPa u Spectrum, QBOe-QBOw')
plt.grid(True)
plt.colorbar()
plt.plot((0,10),(0,.4),linewidth=3,color='k')
plt.text(9,0.32,'18.5 m/s')
plt.plot((0,10),(0,.11),linewidth=3,color='r')
plt.text(7,0.07,'5 m/s')
plt.contour(X,freqs,usig,[0.5],colors='r')
plt.savefig('/pr11/roundy/public_html/exam3paperolr.png')
plt.savefig('/pr11/roundy/public_html/exam3paperolr.eps',format='eps')


