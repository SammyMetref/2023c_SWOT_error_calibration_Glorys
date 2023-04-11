import xarray as xr
import numpy as np
import sys
import pyinterp
import pyinterp.fill
import matplotlib.pylab as plt
from src.aux_cer import *
from src.aux_proj import *
sys.path.append('..')  

class SwotTrack(object):

    def __init__(self, fname=None, dset=None,ilmax=None,minlon=None,lonlatbox=None):
        """ constructeur """
        if fname is not None:
            self._fname = fname
            self._dset = xr.open_dataset(self.filename)
        elif dset is not None:
            self._fname = None
            self._dset = dset
        else:
            raise Exception('either fname or dset should be provided')
             
        if ilmax is not None: 
            self._dset = self._dset.isel(time=slice(0,ilmax))
        if lonlatbox is not None: 
            
            outofbox = False
             
            if np.any(self._dset.lon>lonlatbox['min_lon']): 
                self._dset = self._dset.where(np.max(self._dset.lon,1)>lonlatbox['min_lon'],drop=True)
            else: 
                outofbox = True
            if np.any(self._dset.lon<lonlatbox['max_lon']): 
                self._dset = self._dset.where(np.min(self._dset.lon,1)<lonlatbox['max_lon'],drop=True)
            else: 
                outofbox = True
            if np.any(self._dset.lat>lonlatbox['min_lat']): 
                self._dset = self._dset.where(np.max(self._dset.lat,1)>lonlatbox['min_lat'],drop=True)
            else: 
                outofbox = True
            if np.any(self._dset.lat<lonlatbox['max_lat']): 
                self._dset = self._dset.where(np.min(self._dset.lat,1)<lonlatbox['max_lat'],drop=True)
            else: 
                outofbox = True
                  
            if outofbox : 
                self._dset = self._dset.sel(time=slice(0,0)) 
                            

        self._nadir_mask = None
        
    
    def compute_geos_current(self, invar, outvar):
        
        def coriolis_parameter(lat):  
            """Compute the Coriolis parameter for the given latitude:
                ``f = 2*omega*sin(lat)``, where omega is the angular velocity 
                of the Earth.
    
                Parameters
                ----------
                lat : array
                  Latitude [degrees].
            """
            omega = 7.2921159e-05  # angular velocity of the Earth [rad/s]
            fc = 2*omega*np.sin(lat*np.pi/180.)
            # avoid zero near equator, bound fc by min val as 1.e-8
            fc = np.sign(fc)*np.maximum(np.abs(fc), 1.e-8)
    
            return fc
        
        ds = self._dset
        
        
        dx = 2000 # m
        dy = 2000 # m
        gravity = 9.81
        f_coriolis = coriolis_parameter(ds.lat.values)
        ref_gx, ref_gy = gravity/f_coriolis*np.gradient(ds[invar], dx, edge_order=2)
        geos_current = np.sqrt(ref_gx**2 + ref_gy**2)
        
        self.fc = f_coriolis
         
        self.__enrich_dataset(outvar, geos_current)
        self.__enrich_dataset(outvar + '_y', ref_gx)
        self.__enrich_dataset(outvar + '_x', -ref_gy) 
         
        
    def compute_relative_vorticity(self, invar_x, invar_y, outvar):
        
        ds = self._dset
        
        dx = 2000 # m
        dy = 2000 # m
        
        du_dx, du_dy = np.gradient(ds[invar_x], dx, edge_order=2)
        dv_dx, dv_dy = np.gradient(ds[invar_y], dx, edge_order=2)
        
        ksi = (dv_dx - du_dy)/self.fc
        
        self.__enrich_dataset(outvar, ksi)
        
        
    def display_demo_target(self):
        
        ds = self._dset
        #ds = ds.isel(num_lines=slice(2400, 3000), drop=True)
        ds['time'] = 2*(ds['x_al']-ds['x_al'][0])
        ds['nC'] = 2*ds['nC']
        
        msk = ds.simulated_noise_ssh_karin/ds.simulated_noise_ssh_karin
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.subplot(311)
        (ds.simulated_true_ssh_karin*msk).T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('TARGET: SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.subplot(312)
        (ds.simulated_true_geos_current*msk).T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('TARGET: Geos. current from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.subplot(313)
        vmin = np.nanpercentile(ds['simulated_true_ksi'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ksi'], 95)
        vdata = np.maximum(np.abs(vmin), np.abs(vmax))
        (ds.simulated_true_ksi*msk).T.plot(vmin=-vdata, vmax=vdata, cmap='BrBG', cbar_kwargs={'label': '[]'})
        plt.title('TARGET: Relative vorticity from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        
        plt.show()
        

    def display_demo(self, var_name='karin',msk=None, vmin=None, vmax=None):
        import copy
        ds = copy.deepcopy(self._dset)
        #ds = ds.isel(num_lines=slice(2400, 3000), drop=True)
        ds['time'] = 2*(ds['x_al']-ds['x_al'][0])
        ds['nC'] = 2*ds['nC']
        
        if msk is None:
            msk = ds['ssh_'+var_name]/ds['ssh_'+var_name]
        if vmin is None:
            vmin = np.nanpercentile(ds['ssh_'+var_name], 5)
        if vmax is None:
            vmax = np.nanpercentile(ds['ssh_'+var_name], 95)
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.subplot(311)
        (ds['ssh_'+var_name]*msk).T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('SSH '+var_name, fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        #plt.xlim(2400, 3000)
        plt.subplot(312)
        (ds['geos_current_'+var_name]*msk).T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from SSH '+var_name, fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(313)
        vmin_ksi = np.nanpercentile(ds['ksi_'+var_name]*msk, 5)
        vmax_ksi = np.nanpercentile(ds['ksi_'+var_name]*msk, 95)
        vdata = np.maximum(np.abs(vmin_ksi), np.abs(vmax_ksi))
        (ds['ksi_'+var_name]*msk).T.plot(vmin=-vdata, vmax=vdata, cmap='BrBG', cbar_kwargs={'label': '[]'})
        plt.title('Relative vorticity from Geos. current '+var_name, fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        #plt.xlim(2400, 3000)
        
        plt.show()
        
        return msk,vmin, vmax
        
        
    
    def display_demo_input(self):
        
        ds = self._dset 
        ds['time'] = 2*(ds['x_al']-ds['x_al'][0])
        ds['nC'] = 2*ds['nC']
        
        msk = ds.simulated_noise_ssh_karin/ds.simulated_noise_ssh_karin
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.subplot(311)
        ds.simulated_noise_ssh_karin.T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('INPUT: SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')

        plt.subplot(312)
        ds.simulated_noisy_geos_current.T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.subplot(313)
        vmin = np.nanpercentile(ds['simulated_true_ksi'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ksi'], 95)
        vdata = np.maximum(np.abs(vmin), np.abs(vmax))
        (ds.simulated_noisy_ksi*msk).T.plot(vmin=-vdata, vmax=vdata, cmap='BrBG', cbar_kwargs={'label': '[s$^{-1}$]'})
        plt.title('TARGET: Relative vorticity from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')

        
        plt.show()
        
        
    def display_result_quickstart(self):
        
        ds = self._dset 
        ds['time'] = 2*(ds['x_al']-ds['x_al'][0])
        ds['nC'] = 2*ds['nC']
        
        msk = ds.simulated_noise_ssh_karin/ds.simulated_noise_ssh_karin
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        
        plt.figure(figsize=(22, 12))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.4)
        
        plt.subplot(421)
        (ds.simulated_true_ssh_karin*msk).T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('TARGET: SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(422)
        (ds.simulated_true_geos_current*msk).T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('TARGET: Geos. current from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(423)
        ds.simulated_noise_ssh_karin.T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('INPUT: SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(424)
        ds.simulated_noisy_geos_current.T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(425)
        ds.ssh_karin_filt.T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('RESULT: Filtered SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(426)
        ds.geos_current_ssh_karin_filt.T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from Filtered SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        data = ds.ssh_karin_filt.T - (ds.simulated_true_ssh_karin*msk)
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 95)
        maxval = np.maximum(vmin, vmax)
        plt.subplot(427)
        data.plot(vmin=-maxval, vmax=maxval, cmap='coolwarm', cbar_kwargs={'label': '[m]'})
        plt.title('Filtered SSH true + KaRin noise - SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        data = (ds.geos_current_ssh_karin_filt.T - (ds.simulated_true_geos_current*msk).T)
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 95)
        maxval = np.maximum(vmin, vmax)
        plt.subplot(428)
        data.plot(vmin=-maxval, vmax=maxval, cmap='coolwarm', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from Filtered SSH true + KaRin noise - Geos. current from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.show()
        
    
        
        
    def plot_track(self,calib_var_name,swottrack_input):
        
        ds = self._dset
        ds0 = swottrack_input._dset 
        
        vmin = np.nanpercentile(ds0['ssh_true'], 5)
        vmax = np.nanpercentile(ds0['ssh_true'], 95)
         
        
        fig = plt.figure(figsize=(18,12))
        ax1 = fig.add_subplot(2,3,1)        
        plt.scatter(ds0.lon,ds0.lat,c=ds0['ssh_true'].values, vmin= vmin, vmax= vmax, cmap='Spectral_r')
        plt.colorbar()
        ax1.title.set_text('True ssh')
                   
        vmin = np.nanpercentile(ds0['ssh_err'], 5)
        vmax = np.nanpercentile(ds0['ssh_err'], 95)
                   
        ax2 = fig.add_subplot(2,3,2)        
        plt.scatter(ds0.lon,ds0.lat,c=ds0['ssh_err'].values, vmin= vmin, vmax= vmax, cmap='Spectral_r')
        plt.colorbar()
        ax2.title.set_text('SSH with errors')
        
        vmin = np.nanpercentile(ds0['ssh_true'], 5)
        vmax = np.nanpercentile(ds0['ssh_true'], 95)
        
        ax3 = fig.add_subplot(2,3,3)        
        plt.scatter(ds.lon,ds.lat,c=ds[calib_var_name].values, vmin= vmin, vmax= vmax, cmap='Spectral_r')
        plt.colorbar()
        ax3.title.set_text('Calib SSH')
         
     
        delta0 = ds0['ssh_err'].values - ds0['ssh_true'].values
        delta = ds[calib_var_name].values - ds0['ssh_true'].values
        
        vmin_delta = np.nanpercentile(delta0, 5)
        vmax_delta = np.nanpercentile(delta0, 95)
                   
        ax5 = fig.add_subplot(2,3,5)        
        plt.scatter(ds0.lon,ds0.lat,c=delta0, vmin= vmin_delta, vmax= vmax_delta, cmap='bwr')
        plt.colorbar()
        ax5.title.set_text('Errors')
                   
        vmin_delta = np.nanpercentile(delta, 5)
        vmax_delta = np.nanpercentile(delta, 95)
        
        ax6 = fig.add_subplot(2,3,6)        
        plt.scatter(ds.lon,ds.lat,c=delta, vmin= vmin_delta, vmax= vmax_delta, cmap='bwr')
        plt.colorbar()
        ax6.title.set_text('Calib SSH - True SSH')
                   
                   
        plt.show()
        
        
    def display_track(self):
        
        ds = self._dset
        
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        fig_ssh_true = ds['simulated_true_ssh_karin'].hvplot.quadmesh(x='longitude', y='latitude', clim=(vmin, vmax), cmap='Spectral_r', rasterize=True, title='True ssh karin')
        fig_noisy_ssh = ds['simulated_noise_ssh_karin'].hvplot.quadmesh(x='longitude', y='latitude', clim=(vmin, vmax), cmap='Spectral_r', rasterize=True, title='Noisy ssh karin')
        fig_filtered_ssh = ds['ssh_karin_filt'].hvplot.quadmesh(x='longitude', y='latitude', clim=(vmin, vmax), cmap='Spectral_r', rasterize=True, title='Filtered ssh karin')
    
        delta = ds['ssh_karin_filt'] - ds['simulated_true_ssh_karin']
        vmin_delta = np.nanpercentile(delta.values, 5)
        vmax_delta = np.nanpercentile(delta.values, 95)
        fig_delta_ssh_filtered_ssh_true = delta.hvplot.quadmesh(x='longitude', y='latitude', clim=(-np.abs(vmin_delta), np.abs(vmin_delta)), cmap='bwr', rasterize=True, title='Filtered ssh karin - True ssh karin')
    
        return (fig_ssh_true + fig_noisy_ssh + fig_filtered_ssh + fig_delta_ssh_filtered_ssh_true).cols(2)

        
    def apply_your_own_calib(self,thecalib,invar,outvar,**kwargs):
        """ apply your own calib, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = thecalib(ssha, **kwargs)
        self.__enrich_dataset(outvar, ssha_f)
        
        
       
    def apply_CERmethod_calib(self, invar, outvar):
        """ apply CER method calibration, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_calib = np.zeros_like(ssha)+np.nan  

        Nens=10
        ensanagap = run_CER_method(self,Nens)


        ssha_calib = ensanagap

        self.__enrich_dataset(outvar, ssha_calib)
        
        
       
    def apply_Projmethod_calib(self, invar, outvar, filter_proj=False):
        """ apply Proj method calibration, enrich dataset inplace """
        import numpy as np
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_calib = np.zeros_like(ssha)+np.nan  
         
        if 'ssh_oi' in list(self._dset.keys()):

            ssh_map = self._dset['ssh_oi'].values  

        else:
            ens_path='../inputs/dc_SWOTcalibGS_maps/dc_SWOTcalibGS_maps_20deg.nc'
            ssh_map = interp_ens_to_track(ens_path,self,Nens=1,nselect=1)[0,:,:]

        param0 = ['lin','alin','quad','aquad','cst','acst']
        
        ac1dparam0 = ac1d(param0)
        
        eta = ac1dparam0.invert_glo(self._dset.x_ac[:,0],ssha-ssh_map)
         
        hswath = ac1dparam0.eta2swath(eta,self._dset.x_ac[:,0])
        
        if filter_proj: 
            
            # Filter requirements.
            T = 553.0*2000         # Sample Period
            fs = 2000       # sample rate, Hz
            cutoff = 15      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
            nyq = 0.5 * fs  # Nyquist Frequency
            order = 5       # sin wave can be approx represented as quadratic
            n = int(T * fs) # total number of samples

            ssh_calib_lbf = np.zeros_like(hswath)
            ssh_calib_hbf = np.zeros_like(hswath)
            ssh_calib_bf = np.zeros_like(hswath)+hswath
            for i in range(np.shape(hswath)[1]):
 

                cutoff = 4   
                ssh_calib_lbf[:,i] = butter_lowpass_filter(np.hstack((hswath[:,i],hswath[::-1,i])), cutoff, fs, order)[:np.shape(hswath)[0]]
                cutoff = 50
                ssh_calib_hbf[:,i] = butter_highpass_filter(np.hstack((hswath[:,i],hswath[::-1,i])), cutoff, fs,order)[:np.shape(hswath)[0]]


            ssh_calib_bf = ssh_calib_lbf + ssh_calib_hbf

            ssha_calib = ssha-ssh_calib_bf
            
        else: 
        
            ssha_calib = ssha-hswath
        

        self.__enrich_dataset(outvar, ssha_calib)
        

        
    def apply_ac_track_slope_calib(self, invar, outvar):
        """ apply ac track slope calibration, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_calib = np.zeros_like(ssha)+np.nan  
        
        ssha = np.ma.masked_invalid(ssha) 
        a,b = np.polyfit(np.arange(int(-np.shape(ssha)[1]/2),int(np.shape(ssha)[1]/2)), np.mean(ssha,0), 1)
        ssha_calib = ssha - np.tile((a*np.arange(int(-np.shape(ssha)[1]/2),int(np.shape(ssha)[1]/2))),(np.shape(ssha)[0],1))
         
        self.__enrich_dataset(outvar, ssha_calib)
        
        
    def apply_detrending_calib(self, invar, outvar):
        """ apply ac track slope calibration, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_calib = np.zeros_like(ssha)+np.nan  

        # Import detrending functions
        sys.path.append('/Users/sammymetref/Documents/DATLAS/Notebooks/Experiments_Detrending/') 
        from function_detrending import obs_detrendswot 

        n_gap = 10

        ssha_detrend, aa1,bb1,cc1,ee11,ee12,ff11,ff12 = obs_detrendswot(ssha, n_gap, removealpha0=True,boxsize = 100000)

        self.__enrich_dataset(outvar, ssha_detrend)
        
        
    def apply_ac_track_slope_calib0(self, invar, outvar):
        """ apply ac track slope calibration, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_calib = np.zeros_like(ssha)+np.nan   
        ssha = np.ma.masked_invalid(ssha) 
        
        if np.shape(ssha)[0]!=0:
        
            print(np.shape(ssha))

            for i in range(np.shape(ssha)[0]): 

                #progress_bar(i,np.shape(ssha)[0]-1)

                if np.sum(~ssha[i,:].mask)>5: 


                    a,b = np.polyfit(np.arange(int(-np.shape(ssha)[1]/2),int(np.ceil(np.shape(ssha)[1]/2)))[~ssha[i,:].mask],  ssha[i,:][~ssha[i,:].mask],1) 

                    ssha_calib[i,:][~ssha[i,:].mask] = ssha[i,:][~ssha[i,:].mask] -  a*np.arange(int(-np.shape(ssha)[1]/2),int(np.ceil(np.shape(ssha)[1]/2)))[~ssha[i,:].mask] 

            ssha_calib = np.ma.masked_invalid(ssha_calib) 

            self.__enrich_dataset(outvar, ssha_calib)
        
        
        
    def apply_ac_track_slope_calib1(self, invar, outvar):
        """ apply ac track slope calibration v1, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_calib = np.zeros_like(ssha)+np.nan 
        ssha0 = ssha[:,:25]
        ssha1 = ssha[:,36:]
        
        nc_half=int(np.shape(ssha0)[1])
        ssha_half = ssha0[:,:]
        ssh_half_mean = np.mean(ssha_half,0)
        a,b = np.polyfit(range(nc_half), ssh_half_mean, 1)
        ssha_half1_calib = ssha_half - np.tile((a*np.arange(nc_half)+b),(np.shape(ssha_half)[0],1))
        
        nc_half=int(np.shape(ssha1)[1])
        ssha_half = ssha1[:,:]
        ssh_half_mean = np.mean(ssha_half,0)
        a,b = np.polyfit(range(nc_half), ssh_half_mean, 1) 
        ssha_half2_calib = ssha_half - np.tile((a*np.arange(nc_half)+b),(np.shape(ssha_half)[0],1))
         
        #ssha_calib = np.concatenate((ssha_half1_calib, ssha_half2_calib), axis=1)
        ssha_calib[:,:25] = ssha_half1_calib
        ssha_calib[:,36:] = ssha_half2_calib
        self.__enrich_dataset(outvar, ssha_calib)
        
        
        
    def apply_ac_track_slope_calib2(self, invar, outvar):
        """ apply ac track slope calibration v2, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        
        nc_half=int(np.shape(ssha)[1]/2)
        ssha_half = ssha[:,:nc_half] 
        ssha_half1_calib = np.zeros_like(ssha_half)
        for i in range(np.shape(ssha)[0]):
            ssh_half_mean = ssha_half[i,:]
            a,b = np.polyfit(range(nc_half), ssh_half_mean, 1)
            ssha_half1_calib[i,:] = ssha_half[i,:] - (a*np.arange(nc_half)+b), 
        
        ssha_half = ssha[:,nc_half:]
        ssha_half2_calib = np.zeros_like(ssha_half)
        for i in range(np.shape(ssha)[0]):
            ssh_half_mean = ssha_half[i,:]
            a,b = np.polyfit(range(nc_half), ssh_half_mean, 1) 
            ssha_half2_calib[i,:] = ssha_half[i,:] - (a*np.arange(nc_half)+b) 
         
        ssha_calib = np.concatenate((ssha_half1_calib, ssha_half2_calib), axis=1)
        self.__enrich_dataset(outvar, ssha_calib)

        
    def to_netcdf(self, l_variables, fname):
        """ write to netcdf file """
        if l_variables == 'all':
            l_variables = list(self.dset.data_vars)
        if 'time' not in l_variables:
            l_variables.append('time')

        to_write = self.dset[l_variables]
        to_write.to_netcdf(fname)

    def __check_var_exist(self, varname):
        """ checks for the availability of a variable in the dataset """
        if varname not in self._dset.data_vars:
            raise Exception('variable %s is not defined' % varname)

    def add_swath_variable(self, varname, array, replace=False):
        """ add a 2d variable to the dataset """
        
        if varname in self._dset.data_vars:
            if replace:
                self._dset = self._dset.drop(varname)
            else:
                raise Exception('variable %s already exists' %varname)
        self.__enrich_dataset(varname, array)

    def __enrich_dataset(self, varname: str,  array) -> None:
        """ add a new variable to the dataset """
        self._dset = self._dset.assign(dict(temp=(('time', 'nC'), array)))
        self._dset = self._dset.rename_vars({'temp': varname})

    def fill_nadir_gap(self, invar):
        """ fill nadir gap """
        self.__check_var_exist(invar)
        ssha = self.dset[invar].values
        self._nadir_mask = np.isnan(ssha)

        at = pyinterp.Axis(self.x_at)
        ac = pyinterp.Axis(self.x_ac)
        grid = pyinterp.grid.Grid2D(at,ac,ssha)
        has_converged, filled = pyinterp.fill.gauss_seidel(grid)

        if has_converged:
            self.dset[invar][:] = filled
        else:
            self.dset[invar][:] = np.nan
            import warnings
            warnings.warn('nadir gap filling failed')
            #raise Exception('nadir gap filling failed')

    def empty_nadir_gap(self, invar):
        """ empty nadir gap by applying the mask back on """
        self.__check_var_exist(invar)
        if self.nadir_mask is not None:
            self.dset[invar].values[self.nadir_mask] = np.nan
            
            
    
    @property
    def nadir_mask(self):
        """ return nadir mask """
        return self._nadir_mask

    @property
    def x_at(self):
        return np.arange(self.dset.dims['time'])

    @property
    def x_ac(self):
        return np.arange(self.dset.dims['nC'])

    @property
    def longitude(self):
        return self.dset.longitude.values

    @property
    def latitude(self):
        return self.dset.latitude.values

    @property
    def cycle(self):
        return self.dset.attrs['cycle_number']

    @property
    def track(self):
        return self.dset.attrs['pass_number']

    @property
    def filename(self):
        return self._fname

    @property
    def nx(self):
        return self._dset.dims['num_lines']

    @property
    def ny(self):
        return self._dset.dims['num_pixels']

    @property
    def dset(self):
        return self._dset

    @property
    def minlon(self):
        return np.min(self._dset.longitude.values)

    @property
    def maxlon(self):
        return np.max(self._dset.longitude.values)

    @property
    def minlat(self):
        return np.min(self._dset.latitude.values)

    @property
    def maxlat(self):
        return np.max(self._dset.latitude.values)





        
   
def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)