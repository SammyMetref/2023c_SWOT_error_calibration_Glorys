import xarray as xr
import numpy as np
import sys
import pyinterp
import pyinterp.fill
import matplotlib.pylab as plt
from scipy import interpolate
sys.path.append('..')  

import numpy
class ac1d():

    def __init__(self, param):
        self.param=param


    def operg(self,xac, eta=None, return_geta=False, return_g=False, param=None):
        if param==None: param=self.param
        G = numpy.empty((len(xac), len(param))) + numpy.nan
        for ip in range(len(param)):
            if param[ip]=='lin':
                G[:,ip] = xac*1e3*1e-6
            elif param[ip]=='alin':
                G[:,ip] = numpy.abs(xac)*1e3*1e-6
            elif param[ip]=='quad':
                G[:,ip] = (xac)**2
            elif param[ip]=='aquad':
                G[:,ip] = numpy.sign(xac)*(xac)**2
            elif param[ip]=='cst':
                G[:,ip] = 1.
            elif param[ip]=='acst':
                G[:,ip] = numpy.sign(xac)
            else:
                print('param not recognized')
        if  return_geta:
            Geta = numpy.dot(G,eta)
            return Geta
        elif  return_g:
            return G
        else:
            return None

    def invert(self,G, Ri, y):
        M=numpy.dot(G.T,Ri*G)
        eta = numpy.dot(numpy.dot(numpy.linalg.inv(M),G.T), Ri.squeeze()*y)
        return eta

    def invert_glo(self, xac, ssh, Ri=None, param=None):
        if param==None: param=self.param
        nit = numpy.shape(ssh)[0]
        if (Ri==None): Ri = numpy.ones((nit, len(xac)))
        eta=numpy.empty((len(self.param), nit))+numpy.nan
        for it in range(nit):
            indsel = numpy.isnan(ssh[it,:])==False
            if sum(indsel)>40:
                G = self.operg(xac[indsel], return_g=True, param=param)
                eta[:,it] = self.invert(G, numpy.full((1,sum(indsel)),Ri[it,indsel]).T,  ssh[it,indsel])
        return eta
        # return  [eta[k,:] for k in range(len(self.param))]


    def eta2swath(self, eta, xac, param=None):
        if param==None: param=self.param
        hswath = numpy.empty((numpy.shape(eta)[1],len(xac))) + numpy.nan
        for it in range(numpy.shape(eta)[1]):
            if sum(numpy.isnan(eta[:,it]))==0:
                hswath[it,:] = self.operg(xac, eta=eta[:,it], return_geta=True, param=param)
        return hswath

    

from scipy.signal import butter,filtfilt


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y