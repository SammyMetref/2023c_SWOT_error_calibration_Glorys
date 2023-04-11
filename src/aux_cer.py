import xarray as xr
import numpy as np
import sys
import pyinterp
import pyinterp.fill
import matplotlib.pylab as plt
from scipy import interpolate
sys.path.append('..')  



def interp2d(ds,name_vars,lon_out,lat_out):

    ds = ds.assign_coords(
                 {name_vars['lon']:(ds[name_vars['lon']] % 360),
                  name_vars['lat']:ds[name_vars['lat']]})

    if ds[name_vars['var']].shape[0]!=ds[name_vars['lat']].shape[0]:
        ds[name_vars['var']] = ds[name_vars['var']].transpose()

    if len(ds[name_vars['lon']].shape)==2:
        dlon = (ds[name_vars['lon']][:,1:].values - ds[name_vars['lon']][:,:-1].values).max()
        dlat = (ds[name_vars['lat']][1:,:].values - ds[name_vars['lat']][:-1,:].values).max()
         

        ds = ds.where((ds[name_vars['lon']]<=np.nanmax(lon_out)+dlon) &\
                      (ds[name_vars['lon']]>=np.nanmin(lon_out)-dlon) &\
                      (ds[name_vars['lat']]<=np.nanmax(lat_out)+dlat) &\
                      (ds[name_vars['lat']]>=np.nanmin(lat_out)-dlat),drop=True)

        lon_sel = ds[name_vars['lon']].values
        lat_sel = ds[name_vars['lat']].values

    else:
        dlon = (ds[name_vars['lon']][1:].values - ds[name_vars['lon']][:-1].values).max()
        dlat = (ds[name_vars['lat']][1:].values - ds[name_vars['lat']][:-1].values).max()
 

        ds = ds.where((ds[name_vars['lon']]<=np.nanmax(lon_out)+dlon) &\
                      (ds[name_vars['lon']]>=np.nanmin(lon_out)-dlon) &\
                      (ds[name_vars['lat']]<=np.nanmax(lat_out)+dlat) &\
                      (ds[name_vars['lat']]>=np.nanmin(lat_out)-dlat),drop=True)

        lon_sel,lat_sel = np.meshgrid(
            ds[name_vars['lon']].values,
            ds[name_vars['lat']].values)

    var_sel = ds[name_vars['var']].values


    # Interpolate to state grid 
    var_out = interpolate.griddata((lon_sel.ravel(),lat_sel.ravel()),
                   var_sel.ravel(),
                   (lon_out.ravel(),lat_out.ravel())).reshape((lat_out.shape))

    return var_out


def run_CER_method(swt,Nens=10):
    

    ssha = swt.dset['ssh_err'].values
    
    ssha_orig = ssha 
    
    if np.shape(ssha)[1]!=62:  
        print('Warning: Issue with across track size')
        ssha0 = np.zeros([np.shape(ssha)[0],62]) + np.nan
        if np.nanmin(swt.dset['x_ac'].values)!=-70000: ssha0[:,-np.shape(ssha)[1]:] = ssha
        if np.nanmax(swt.dset['x_ac'].values)!=70000: ssha0[:,:np.shape(ssha)[1]] = ssha
        ssha = ssha0

        
    if 'ssh_oi' in list(swt.dset.keys()):
        
        ssh_ens_proj = np.zeros([Nens,np.shape(ssha)[0],np.shape(ssha)[1]]) + np.nan
        
        for iens in range(Nens):
            ssh_ens_proj[iens,:,:] = swt.dset['ssh_oi'].values + np.random.normal(0,0.01)#,np.shape(ssha))
    
    else:
        # Retrieve and project ensemble
        ens_path='../inputs/dc_SWOTcalibGS_maps/dc_SWOTcalibGS_maps_20deg.nc'

        print('Retrieving and projecting ensemble')
        ssh_ens_proj = interp_ens_to_track(ens_path,swt,Nens)


        if np.shape(ssh_ens_proj)[2]!=62:  
            ssh_ens_proj0 = np.zeros([np.shape(ssh_ens_proj)[0],np.shape(ssh_ens_proj)[1],62]) + np.nan

            if np.nanmin(swt.dset['x_ac'].values)!=-70000: ssh_ens_proj0[:,:,-np.shape(ssh_ens_proj)[2]:] = ssh_ens_proj
            if np.nanmax(swt.dset['x_ac'].values)!=70000: ssh_ens_proj0[:,:,:np.shape(ssh_ens_proj)[2]] = ssh_ens_proj
            ssh_ens_proj = ssh_ens_proj0

    # Detrend ensemble
    n_gap = 0

    ssh_ens_detrend = np.zeros_like(ssh_ens_proj)

    for iens in range(Nens):
        #print(iens)
        ssh_ens_detrend[iens,:,:], aa1,bb1,cc1,ee11,ee12,ff11,ff12 = obs_detrendswot0(ssh_ens_proj[iens,:,:], n_gap, removealpha0=False,boxsize = 100) 
        


    print('Detrending ensemble')

    # Detrend SWOT data 
    ssha_detrend, aa1,bb1,cc1,ee11,ee12,ff11,ff12 = obs_detrendswot0(ssha, n_gap, removealpha0=False,boxsize = 100)

    # Prepare data for ETKF
    sshed = ssh_ens_detrend
    sshep = ssh_ens_proj
     

    obs = ssha_detrend[~np.isnan(ssha_detrend)] 
    nobs = np.shape(obs)[0]
    ensobs = np.zeros([nobs,Nens])
    ens = np.zeros([nobs,Nens])
    
    ensana0 = np.zeros_like(ssha_detrend)+np.nan 
    
    for iens in range(Nens): 
        ensobs[:,iens] =  sshed[iens,:,:][~np.isnan(ssha_detrend)] 
        ens[:,iens] = sshep[iens,:,:][~np.isnan(ssha_detrend)]

    ens[np.isnan(ens)] = 0
    ensobs[np.isnan(ensobs)] = 0


    # Run ETKF
    sig_obs = 0.0001
    noise_obs = np.random.normal(0,sig_obs,np.shape(obs))
    
    
    print('Running EnKF')
    ensana = etkfana(ens,ensobs,obs+noise_obs,sig_obs**2*np.ones_like(ensobs),Nens)
    
    ensana = np.mean(ensana,1)
    ensana0[~np.isnan(ssha_detrend)]  = ensana
    ensana = np.reshape(ensana0,(np.shape(ssha)[0],np.shape(ssha)[1]))

    ensanagap = ssha*np.nan
    ensanagap[:,:int(np.shape(ssha)[1]/2-n_gap/2)] = ensana[:,:int(np.shape(ssha)[1]/2-n_gap/2)]
    ensanagap[:,int(np.shape(ssha)[1]/2+n_gap/2):] = ensana[:,int(np.shape(ssha)[1]/2+n_gap/2):]
    
    
    if np.shape(ssha_orig)[1]!=62:  
        if np.nanmin(swt.dset['x_ac'].values)!=-70000: ensanagap = ensanagap[:,-np.shape(ssha_orig)[1]:]
        if np.nanmax(swt.dset['x_ac'].values)!=70000: ensanagap = ensanagap[:,:np.shape(ssha_orig)[1]]


    return ensanagap


def interp_ens_to_track(ens_path,swt,Nens=10,nselect=60):
 
    ssha = swt.dset['ssh_err'].values
    ssha_t = swt.dset['ssh_true'].values
    lon = swt.dset['lon'].values%360
    lat = swt.dset['lat'].values
    
    i_swt = 20#int( (np.array(np.mean(swt.dset.time), dtype='datetime64[D]')-np.datetime64('2012-10-01') )/ np.timedelta64(1, "D") )

    ssh_ens_proj = np.zeros([Nens,np.shape(ssha)[0],np.shape(ssha)[1]])

    ds_ens = xr.open_dataset(ens_path)
    
    x = ds_ens.lon[:].values
    y = ds_ens.lat[:].values
    if np.size(np.shape(x))<1:
        xx, yy = np.meshgrid(x, y)
    else: 
        xx,yy = x,y

    ssh_ens = np.zeros([Nens,np.shape(ds_ens.lon)[0],np.shape(ds_ens.lat)[0]])

    ii_ens = 0
    
    if nselect==1:
        random_members = [i_swt]
    else:
        random_members = np.random.randint(max(0,i_swt-int(nselect/2)),min(ds_ens.time.size,i_swt+int(nselect/2)),Nens)
    
    for iens in random_members:#np.shape(ds_ens.ssh)[0]

        #print(iens)

        ssh0 = np.array(ds_ens.gssh[iens,:,:].values) 
        ssh_ens[ii_ens,:,:] = ssh0 

        ssh_ens_proj[ii_ens,:,:] =interp2d(ds_ens.isel({'time':iens}),{'lon':'lon','lat':'lat','var':'gssh'},lon% 360,lat)

        ii_ens += 1
        
    return ssh_ens_proj



import scipy.optimize as so

def obs_detrendswot0(ssh_swotgrid0, n_gap, removealpha0=False,boxsize = 100000):
    """
    NAME 
        obs_detrendswot

    DESCRIPTION 
        Create the projection T (defined in Metref et al. 2019) of ssh_swotgrid0, an array of SSH in SWOT grid.
        
        Args: 
            ssh_swotgrid0 (array of float): SSH on the SWOT grid.  
            n_gap (integer): Swath qap size (in gridpoints)
  

        Returns: 
            ssh_detrended0 (array of float): T(SSH) on the SWOT grid.

    """    
    # Size of averaging box for regression coefficient (only works if meanafter == True)
    #boxsize = 500 
    # Lenght of the swath
    dimtime = np.shape(ssh_swotgrid0)[0]
    # Width of the swath
    dimnc = np.shape(ssh_swotgrid0)[1]

    ssh_detrended0 = np.zeros_like(ssh_swotgrid0)
    ssh_detrended0[:,:] = ssh_swotgrid0[:,:]
    
    # Initialization of the regression coefficients
    a1=0
    b1=0 
    c1=0 
    e11=0
    e12=0 
    f11=0
    f12=0 
    
    # Two options: computing the regression coefficients on the mean(ssh_acrosstrack) with meanafter==False or averaging the regression coefficients computed at each ssh_acrosstrack with meanafter==True (!! a lot slower !!)
    meanafter =  False # True
    meanbefore = True # True # False # 
    
    #if boxsize > dimtime: 
    #    meanbefore = False # True # False # 
 
    # Calculate the regression coefficients 

    
    if meanbefore :  
        
        ssh_masked = np.ma.masked_where(ssh_swotgrid0>999.,ssh_swotgrid0) 
        ssh_masked = np.ma.masked_where(ssh_masked<-999.,ssh_masked)
        aa1=np.zeros(dimtime) 
        bb1=np.zeros(dimtime) 
        cc1=np.zeros(dimtime) 
        ee11=np.zeros(dimtime)
        ee12=np.zeros(dimtime)
        ff11=np.zeros(dimtime)
        ff12=np.zeros(dimtime)
        i_along_init = 0
        i_along_fin = 1
        for i_along in range(dimtime):
            ssh_across = np.ma.mean(ssh_masked[max(0,int(i_along-boxsize/2)):min(dimtime,int(i_along+boxsize/2)),:],0)
            #print(max(0,int(i_along-boxsize/2)))
            #print(min(dimtime,int(i_along+boxsize/2))) 
            #print(np.shape(ssh_across))
            if np.shape(ssh_across)[0]==dimnc:  
                if i_along_init == 0:
                    i_along_init = i_along
                nn = np.shape(ssh_across)[0] 
                if nn != 0: 
                    x_across = np.arange(nn)-int(nn/2)
                #x_across = np.where(ssh_swotgrid0[i_along,:]<999.)[0]-int(np.shape(ssh_swotgrid0)[1]/2)  
                x_across[x_across<0]=x_across[x_across<0]-n_gap/2
                x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2
                i_along_fin = i_along


                def linreg3(params): 
                    return np.sum( ( ssh_across-(params[0]+params[1]*x_across + params[2]*x_across**2 +np.append(params[3]+params[4]*x_across[x_across<=0],params[5]+params[6]*x_across[x_across>0],axis=0) ) )**2 ) 

                params = np.array([a1,b1,c1,e11,e12,f11,f12])
                coefopt = so.minimize(linreg3, params, method = "BFGS") 
                aa1[i_along], bb1[i_along], cc1[i_along], ee11[i_along], ee12[i_along], ff11[i_along], ff12[i_along] = coefopt['x'][0], coefopt['x'][1], coefopt['x'][2], coefopt['x'][3], coefopt['x'][4], coefopt['x'][5], coefopt['x'][6] 
                a1=aa1[i_along]
                b1=bb1[i_along] 
                c1=cc1[i_along] 
                e11=ee11[i_along]
                e12=ee12[i_along]
                f11=ff11[i_along]
                f12=ff12[i_along]    
    else :   
            ssh_masked = np.ma.masked_where(ssh_swotgrid0>999.,ssh_swotgrid0)
            ssh_masked = np.ma.masked_where(ssh_masked<-999.,ssh_masked) 
            ssh_across = np.nanmean(ssh_masked,0) 

            if np.shape(ssh_across)[0]==dimnc:  
                nn = np.shape(ssh_across)[0] 
                if nn != 0: 
                    x_across = np.arange(nn)-int(nn/2) #np.where(ssh_swotgrid0[int(dimtime/2),:]<999.)[0]-int(np.shape(ssh_swotgrid0)[1]/2)   
                    x_across[x_across<0]=x_across[x_across<0]-n_gap/2+1 
                    x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2   

                    if nn == np.shape(x_across)[0]: 

                        def linreg3(params): 
                            return np.sum( ( ssh_across-(params[0]+params[1]*x_across + params[2]*x_across**2 +np.append(params[3]+params[4]*x_across[x_across<=0],params[5]+params[6]*x_across[x_across>0],axis=0) ) )**2 ) 

                        params = np.array([a1,b1,c1,e11,e12,f11,f12])
                        coefopt = so.minimize(linreg3, params, method = "BFGS") 
                        a1,b1,c1,e11,e12,f11,f12 = coefopt['x'][0], coefopt['x'][1], coefopt['x'][2], coefopt['x'][3], coefopt['x'][4], coefopt['x'][5], coefopt['x'][6]  

    
    # Detrend ssh_acrosstrack using the regression coefficients calculated above
    for i_along in range(dimtime): 
        ssh_across = ssh_swotgrid0[i_along,ssh_swotgrid0[i_along,:]<999.]
        if True:#np.sum(ssh_across)==dimnc:#!=0:   
            if meanafter :
                if int(max(i_along_init,i_along-boxsize/2)) < int(min(i_along+boxsize/2,i_along_fin)):
                    a1 = np.mean(aa1[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    b1 = np.mean(bb1[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))]) 
                    c1 = np.mean(cc1[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))]) 
                    e11 = np.mean(ee11[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    e12 = np.mean(ee12[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    f11 = np.mean(ff11[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    f12 = np.mean(ff12[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))]) 
                else:
                    if i_along-boxsize/2 >= i_along_fin: 
                        # Check obs at 2012-10-02 06:00:00 
                        a1 = np.mean(aa1[int(i_along_fin)])
                        b1 = np.mean(bb1[int(i_along_fin)]) 
                        c1 = np.mean(cc1[int(i_along_fin)]) 
                        e11 = np.mean(ee11[int(i_along_fin)])
                        e12 = np.mean(ee12[int(i_along_fin)])
                        f11 = np.mean(ff11[int(i_along_fin)])
                        f12 = np.mean(ff12[int(i_along_fin)])
                    if i_along_init >= i_along+boxsize/2: 
                        # Check obs at 2012-10-05 15:00:00 
                        a1 = np.mean(aa1[int(i_along_init)])
                        b1 = np.mean(bb1[int(i_along_init)]) 
                        c1 = np.mean(cc1[int(i_along_init)]) 
                        e11 = np.mean(ee11[int(i_along_init)])
                        e12 = np.mean(ee12[int(i_along_init)])
                        f11 = np.mean(ff11[int(i_along_init)])
                        f12 = np.mean(ff12[int(i_along_init)])
            elif meanbefore: 
                a1 = aa1[i_along]
                b1 = bb1[i_along]
                c1 = cc1[i_along]
                e11 = ee11[i_along]
                e12 = ee12[i_along]
                f11 = ff11[i_along]
                f12 = ff12[i_along]  
            else: 
                aa1=np.zeros(dimtime) + a1 
                bb1=np.zeros(dimtime) + b1 
                cc1=np.zeros(dimtime) + c1 
                ee11=np.zeros(dimtime) + e11
                ee12=np.zeros(dimtime) + e12
                ff11=np.zeros(dimtime) + f11
                ff12=np.zeros(dimtime) + f12 
               

            nn = np.shape(ssh_across)[0] 
            x_across = np.where(ssh_swotgrid0[i_along,:]<999.)[0]-int(np.shape(ssh_swotgrid0)[1]/2) 
            x_across[x_across<0]=x_across[x_across<0]-n_gap/2
            x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2
            
            x_across_sized = x_across[ssh_across<999.]  
            if removealpha0 :
                ssh_detrended0[i_along,ssh_swotgrid0[i_along,:]<999.] = ssh_across[ssh_across<999.] - (a1+b1*x_across_sized + c1*x_across_sized**2+np.append(e11+e12*x_across_sized[x_across_sized<=0],f11+f12*x_across_sized[x_across_sized>0],axis=0) ) 
            else :
                ssh_detrended0[i_along,ssh_swotgrid0[i_along,:]<999.] = ssh_across[ssh_across<999.] - (b1*x_across_sized + c1*x_across_sized**2+np.append(e11+e12*x_across_sized[x_across_sized<=0],f11+f12*x_across_sized[x_across_sized>0],axis=0) )  
   
    
    return  ssh_detrended0, aa1,bb1,cc1,ee11,ee12,ff11,ff12
 
    
    
    
    #! /usr/bin/python # -*- coding: iso-8859-1 -*-

import numpy as npy
import copy

##-----------------------------------------------------------------
## Function etkfana
## Perform ETKF analysis
##-----------------------------------------------------------------

def etkfana(xx_in,nobs_in,muobs_in,sigobs_in,nmem):

    """Perform ETKF analysis"""

# xx_in: input forecast ensemble
# nobs_in: index of the observed variable in xx_in[nobs_in,:]
# muobs_in: observation mean
# sigobs_in: observation variance. Obs in Gaussian for now.

    xxa=npy.zeros_like(xx_in)
    nm=npy.shape(xx_in)[1]
    nvar=npy.shape(xx_in)[0]
    nobs=npy.shape(nobs_in)[0]
 
    Zk = npy.zeros([nvar,nmem],)
    A  = npy.zeros([nobs,nmem],)
  
    for ivar in range(nvar) :
        Zk[ivar,:] = (xx_in[ivar,:]-npy.mean(xx_in,1)[ivar]*npy.ones([nmem],))/npy.sqrt(nmem-1) # Anomaly matrix 
 

    A = npy.dot(npy.transpose(Zk[:,:]/npy.sqrt(sigobs_in[:,:])),Zk[:,:]/npy.sqrt(sigobs_in[:,:]))  

    
    [Gamma,C] = npy.linalg.eigh(A)
    
    for imem in range(nmem) : 
        Gamma[imem] = max(Gamma[imem],0.)   

# Increment on the anomalies   
    SQRTinvIpGamma = npy.zeros([nmem],) 
    SQRTinvIpGamma[:] = 1./npy.sqrt(1.+Gamma[:]) 
    T = npy.dot(C,npy.dot(npy.diag(SQRTinvIpGamma),npy.transpose(C)))      
    Za = npy.dot(Zk,T)  

#    print npy.dot(Za*sqrt(nmem-1),npy.transpose(Za*sqrt(nmem-1)))
 
# Increment on the mean       

    eta_o = npy.dot(npy.transpose(Zk[:,:]),(muobs_in-npy.mean(nobs_in,1))/sigobs_in[:,0])    
    wa = npy.dot(npy.transpose(C),eta_o)
    invIpGamma = npy.zeros([nmem],) 
    invIpGamma[:] = 1./(1.+Gamma[:])
    wa = npy.dot(wa,npy.diag(invIpGamma))
    wa = npy.dot(C,wa)  #    wa = C (1+Gamma)m1 C^T (H Zk)^T Rm1 innov
   
   
    xxamean = npy.mean(xx_in,1) + npy.dot(Zk,wa)   

     
    for ivar in range(nvar) :
        xxa[ivar,:] = xxamean[ivar]*npy.ones([nmem],)+Za[ivar,:]*npy.sqrt(nmem-1)  

 
 

    return xxa
