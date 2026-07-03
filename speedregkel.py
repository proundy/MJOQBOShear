import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date
from scipy import io as io
from mpl_toolkits.basemap import Basemap
from scipy.signal import detrend as detrend
mpl.rcParams['contour.negative_linestyle'] = 'solid'
mpl.rcParams.update({'font.size': 13})
RMM=np.loadtxt('/roundylab_rit/roundy/RMMindex.txt')



file1=io.loadmat('/roundylab_rit/roundy/aroniadata/olr/Ldata/L0') #Load equatorial OLR anomaly data.
Aeq=file1['A']/5
file1=io.loadmat('/roundylab_rit/roundy/aroniadata/olr/Ldata/L25') #Load equatorial OLR anomaly data.
Aeq=file1['A']/5+Aeq
file1=io.loadmat('/roundylab_rit/roundy/aroniadata/olr/Ldata/Lm25') #Load equatorial OLR anomaly data.
Aeq=file1['A']/5+Aeq
file1=io.loadmat('/roundylab_rit/roundy/aroniadata/olr/Ldata/Lm5') #Load equatorial OLR anomaly data.
Aeq=file1['A']/5+Aeq
file1=io.loadmat('/roundylab_rit/roundy/aroniadata/olr/Ldata/L5') #Load equatorial OLR anomaly data.
A=file1['A']/5+Aeq


A=np.hstack((A,A,A)) #Repeat the data going around the world 3 times. 

Xvals=np.arange(60,185,5)

def indmake(A,speed):
    
     Xvals=np.arange(60,185,5)
     speedindexes=np.zeros((A.shape[0],Xvals.shape[0]))
     Re=6.371e6
     t=np.arange(-100,101)
     psi=np.zeros((201,144*3))
     fbx=10000.  #Sets the zonal envelope width.
     fbt=1000.   #Sets the temporal envelope width.
     fcx=4/360.  #Sets the central zonal wavenumber of the wavelet. 

     fct=(fcx)*(360./(2*np.pi*Re))*speed*86400   #Setting the phase speed sets the frequency. 
     for xx in np.arange(Xvals.shape[0]):
          x=np.arange(144*3)*2.5-(Xvals[xx]+360)  #Center on 80E (treat that as the zero longitude.
          for xxx in np.arange(144*3):
            #for  tt in np.arange(201):
              psi[t+100,xxx]=((np.pi*fbx)**(-.5))*((np.pi*fbt)**(-.5))*np.cos(2*np.pi*(fcx*x[xxx]-fct*t))*np.exp(-x[xxx]**2/fbx)*np.exp(-(t**2)/fbt)

          for tind in np.arange(101,np.shape(A)[0]-100):
              Aseg=detrend(A[tind-100:tind+101,:],axis=0)
              speedindexes[tind,xx]=np.sum(Aseg*psi)
     return speedindexes
     '''
speedindexes17=indmake(A,17.)
print 'speedindexes7 made'
speedindexes10=indmake(A,10.)
'''
#speedindexes18=indmake(A,18.)
#np.save('speedindexes18.npy',speedindexes18)
#np.save('speedindexes10.npy',speedindexes10)
speedindexes18=np.load('speedindexes18.npy')
speedindexes10=np.load('speedindexes10.npy')
Iout=np.where(np.logical_and(np.logical_or(RMM[:,1]<3,RMM[:,1]>11),RMM[:,6]<10))[0]
I=np.where(Iout<12000)[0]
Iout=Iout[I]

Llist=['Lm30','Lm275','Lm250','Lm225','Lm20','Lm175','Lm15','Lm125','Lm10','Lm75','Lm5','Lm25','L0','L25','L5','L75','L10','L125','L15','L175','L20','L225','L250','L275','L30','L325','L35','L375','L40','L425','L45','L475','L50','L525','L55','L575','L60']

OLRarray=np.zeros((25,12000,144))
htarray=np.zeros((37,12000,144))
uarray=np.zeros((37,12000,144))
uarraybk=np.zeros((37,12000,144))
varraybk=np.zeros((37,12000,144))
varray=np.zeros((37,12000,144))
d0=date(1974,1,1)
d1=date(1974,6,1)
hcutter=(d1-d0).days
'''
for y in np.arange(37):
    if y<25:
        file='/aronia1/roundy/olr/Ldata/%s.mat'%Llist[y]
        filed=io.loadmat(file)
        A=filed['A'][:12000,:]
        OLRarray[y,:A.shape[0],:]=A
    file='/aronia2/roundy/height/data/h200/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][hcutter:,:][:12000,:]
    htarray[y,:A.shape[0],:]=A
    
    file='/aronia2/roundy/winds/uwnd/h200/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][hcutter:,:][:12000,:]
    uarray[y,:A.shape[0],:]=A
    file='/aronia2/roundy/winds/uwnd/h200/cycles/C%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['cycles'][hcutter:,:][:12000,:]
    uarraybk[y,:A.shape[0],:]=A.squeeze()
    file='/aronia2/roundy/winds/vwnd/h200/Ldata/%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['A'][hcutter:,:][:12000,:]
    varray[y,:A.shape[0],:]=A
    file='/aronia2/roundy/winds/vwnd/h200/cycles/C%s.mat'%Llist[y]
    filed=io.loadmat(file)
    A=filed['cycles'][hcutter:,:][:12000,:]
    varraybk[y,:A.shape[0],:]=A.squeeze()
np.save('data/varray200bk.npy',varraybk)
np.save('data/varray200.npy',varray)
np.save('data/uarray200.npy',uarray)
np.save('data/uarray200bk.npy',uarraybk)
np.save('data/OLRarray.npy',OLRarray)
np.save('data/htarray200.npy',htarray)
'''
OLRarray=np.load('data/OLRarray.npy')
htarray=np.load('data/htarray200.npy')
varray=np.load('data/varray200.npy')
varraybk=varray+varraybk
uarray=np.load('data/uarray200.npy')
uarraybk=np.load('data/uarray200bk.npy')
uarraybk=uarray+uarraybk

Ix=np.arange(-1,144)
Ix[0]=143
Ix=np.append(Ix,0)
Iix=np.arange(1,145)


ubkfft=np.fft.fft(uarraybk,axis=1)
n=uarraybk.shape[1]/np.arange(uarraybk.shape[1]).astype(float)
I=np.where(n>30)[0]
ubkfil=np.zeros_like(ubkfft)
ubkfil[:,I,:]=ubkfft[:,I,:]
ubkfil[:,-I[1:],:]=ubkfft[:,-I[1:],:]
ubk=np.fft.ifft(ubkfil,axis=1).real


uhffft=np.fft.fft(uarray,axis=1)
n=uarray.shape[1]/np.arange(uarray.shape[1]).astype(float)
I=np.where(n<30)[0]
uhffil=np.zeros_like(uhffft)
uhffil[:,I,:]=uhffft[:,I,:]
uhffil[:,-I,:]=uhffft[:,-I,:]
uhf=np.fft.ifft(uhffil,axis=1).real

Iyy=np.arange(1,36)
adv=np.zeros_like(uarraybk)
adv[:,:,Ix[Iix]-1]=-ubk[:,:,Ix[Iix]-1]*(uhf[:,:,Ix[Iix+1]]-uhf[:,:,Ix[Iix-1]])
Re=6.38e6
lats=np.arange(-30,62.5,2.5)*2*np.pi/360
dx=2*np.pi*Re*np.cos(lats)/144*2
dy=2*np.pi*Re/144*2
for y in np.arange(lats.shape[0]):
     adv[y,:,:]=adv[y,:,:]/dx[y]

#adv[Iyy,:,:]=adv[Iyy,:,:]-varraybk[Iyy,:,:]*(uarraybk[Iyy+1,:,:]-uarraybk[Iyy-1])/dy


t=np.arange(12000)
nharm=4
X=np.ones((12000,2*nharm+1))
for n in np.arange(1,nharm+1):
    X[:,2*n-1]=np.cos(2*np.pi*n*t/365.25) 
    X[:,2*n]=np.sin(2*np.pi*n*t/365.25) 

C=np.linalg.inv(X.T.dot(X)).dot((X.T.dot(htarray)).transpose(1,0,2))
htarray=htarray-X.dot(C.transpose(1,0,2)).transpose(1,0,2)
xgrid=np.arange(0,360,2.5)
ygrid=np.arange(-30,62.5,2.5)
Iy=np.where(np.abs(ygrid)<=10)[0]
I0=np.where(ygrid==0)[0]
I230=np.where(xgrid==230)[0]
for y in np.arange(37):
    if y==0:
        Xgrid=xgrid.reshape(1,144)
    else:
        Xgrid=np.vstack((Xgrid,xgrid.reshape(1,144)))
for x in np.arange(144):
    if x==0:
      Ygrid=ygrid.reshape(37,1)
    else:
        Ygrid=np.hstack((Ygrid,ygrid.reshape(37,1)))

def reg(x,y):
    C=(LA.inv(x.T.dot(x)).dot(x.T).dot(y)).transpose(1,0,2)
    a=x.squeeze()
    a=np.expand_dims(-2*np.std(a),1)
    return np.squeeze(a.dot(C))


def randmake(datarg):
     fftout=np.fft.fft(datarg)
     power=fftout*np.conj(fftout)
     randfft=np.zeros(datarg.shape[0]).astype('complex')
     signs=np.array([-1,1])
     for n in np.arange(int(power.shape[0]/2)):
          realsq=np.random.random()*power[n]
          imagsq=power[n]-realsq
          realpt=np.random.choice(signs)*np.sqrt(realsq)
          imagpt=np.random.choice(signs)*np.sqrt(imagsq)
          randfft[n]=realpt+imagpt*1j
          randfft[-n]=realpt-imagpt*1j
     return np.fft.ifft(randfft).real

randcomp=np.zeros((37,144,1000))
adv1comprand=np.zeros((37,144))
adv3comprand=np.zeros((37,144))
adv5comprand=np.zeros((37,144))
adv7comprand=np.zeros((37,144))

my_cmap=plt.cm.get_cmap('jet',25)
my_vals=my_cmap(np.arange(25))
my_vals[12]=[1,1,1,.3]
my_cmap=mpl.colors.LinearSegmentedColormap.from_list("my_cmap",my_vals)
plt.figure(num=1,figsize=(11,11))

lags=np.arange(1)
def randreg(speedindexes,adv):
    randcomp=np.zeros((adv.shape[0],adv.shape[2],1000))
    for i in np.arange(1000):
        print i
        X=np.expand_dims(randmake(speedindexes).squeeze(),1)
        randcomp[:,:,i]=reg(X,adv)
    return randcomp
for n in [5]:
     base=n
     for lag in np.arange(lags.shape[0]): 
            advcomprand18=randreg(speedindexes18[Iout,n],adv[:,Iout+lags[lag],:])
            advcomprand10=randreg(speedindexes10[Iout,n],adv[:,Iout+lags[lag],:])
            advcomprand18=np.sort(advcomprand18,axis=2)	    
            advcomprand10=np.sort(advcomprand10,axis=2)	    
	    temp=np.mean(np.abs(advcomprand10[:,:,(50,950)]),axis=2)

	    
	    X=speedindexes18[Iout,n:n+1]
	    compositeheight18=reg(X,htarray[:,Iout+lags[lag],:])
	    compositeolr18=reg(X,OLRarray[:,Iout+lags[lag],:])
	    compositeu18=reg(X,uarray[:,Iout+lags[lag],:])
	    compositev18=reg(X,varray[:,Iout+lags[lag],:])
	    compositeadv18=reg(X,adv[:,Iout+lags[lag],:])
	    
	    X=speedindexes10[Iout,n:n+1]
	    compositeheight10=reg(X,htarray[:,Iout+lags[lag],:])
	    compositeolr10=reg(X,OLRarray[:,Iout+lags[lag],:])
	    compositeu10=reg(X,uarray[:,Iout+lags[lag],:])
	    compositev10=reg(X,varray[:,Iout+lags[lag],:])
	    compositeadv10=reg(X,adv[:,Iout+lags[lag],:])

	    sigsm=compositeadv10-temp
            sigbg=compositeadv10+temp
	    
	    sig18=np.logical_or(np.less(compositeadv18,sigsm),np.greater(compositeadv18,sigbg))


	    plt.clf()

	    Xgrid,Ygrid=np.meshgrid(xgrid,ygrid)
	    plt.subplot(2,1,1)
	    map=Basemap(projection='cyl',llcrnrlat=-30,urcrnrlat=30,llcrnrlon=40,urcrnrlon=120,resolution='l')
	    Xgrid1,Ygrid1=map(Xgrid,Ygrid)
	    map.drawcoastlines(linewidth=0.25)
	    map.drawmeridians(np.arange(0,360,30))
	    map.drawparallels(np.arange(-90,90,30))
	    #my_cmap.set_middle('w')
	     
	    map.contourf(Xgrid[:25,:],Ygrid[:25,:],compositeolr10[:,:],np.arange(-20,21,1),vmin=-20,vmax=20,cmap=my_cmap)
	    plt.colorbar(shrink=0.5)
	    map.contour(Xgrid1,Ygrid1,compositeadv10[:,:],np.arange(.1,1,.1)*5e-5,colors='r')
	    map.contour(Xgrid1,Ygrid1,compositeadv10[:,:],np.arange(-1,0,.1)*5e-5,colors='b')
	    map.contour(Xgrid1,Ygrid1,sig18,1,colors='k',linewidth=2)
	    
	    map.quiver(Xgrid1[:25,:144:2],Ygrid1[:25,:144:2],compositeu10[:25,0:144:2],compositev10[:25,0:144:2])
	    s=n+1
	 #   plt.tight_layout()
	    plt.title('Regressed Kelvin Structure, 200 hPa Height  %dE Speed = 10 m/s, lag = %d'%(Xvals[n],lags[lag]))
	    
	    
	    plt.subplot(2,1,2)
	    map=Basemap(projection='cyl',llcrnrlat=-30,urcrnrlat=30,llcrnrlon=40,urcrnrlon=120,resolution='l')
	    Xgrid1,Ygrid1=map(Xgrid,Ygrid)
	    map.drawcoastlines(linewidth=0.25)
	    map.drawmeridians(np.arange(0,360,30))
	    map.drawparallels(np.arange(-90,90,30))
	    #my_cmap.set_middle('w')
	     
	    map.contourf(Xgrid1[:25,:],Ygrid1[:25,:],compositeolr18[:,:],np.arange(-20,21,1),vmin=-20,vmax=20,cmap=my_cmap)
	    plt.colorbar(shrink=0.5)
	    map.contour(Xgrid1,Ygrid1,compositeadv18[:,:],np.arange(.1,1,.1)*5e-5,colors='r')
	    map.contour(Xgrid1,Ygrid1,compositeadv18[:,:],np.arange(-1,0,.1)*5e-5,colors='b')
	    map.contour(Xgrid1,Ygrid1,sig18,1,colors='k',linewidth=2)
	    
	    map.quiver(Xgrid1[:25,:144:2],Ygrid1[:25,:144:2],compositeu18[:25,0:144:2],compositev18[:25,0:144:2])
	    #plt.tight_layout()
	    plt.title('Regressed Kelvin Structure, 200 hPa Height  %dE Speed = 18 m/s, lag = %d'%(Xvals[n],lags[lag]))
	    
            plt.savefig('/pr11/roundy/public_html/speedfigs/figadvkel1018_80.png')

