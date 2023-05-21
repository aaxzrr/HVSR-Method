import obspy as ob
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

class tcs():
    
    #cek dt dan nsamples
    def check_input(values_dict):
        ns = values_dict["ns"]
        #check apakah data memiliki dt yang sama dan apakah jumlah sampel sudah sama
        dt = ns.dt
        nsamples = ns.samples
        for key, value in values_dict.item():
            if key == "ns":
                continue
            if value.dt != dt:
                if value.dt != dt:
                    msg = "All components must have equal `dt`."
                raise ValueError(msg)

            if value.nsamples != nsamples:
                txt = "".join([f"{_k}={_v.nsamples} " for _k,
                               _v in values_dict.items()])
                msg = f"Components are different length: {txt}"
                raise ValueError(msg)
        return (values_dict["ns"], values_dict["ew"], values_dict["vt"])
    
    #inisialisasi tiga komponen mseed
    def __init__(self, ns, ew, vt, meta=None):
        values_dict = {"ns": ns, "ew": ew, "vt": vt}
        self.ns, self.ew, self.vt = self.check_input(values_dict)
        
        meta = {} if meta is None else meta
        self.meta = {"File Name": "NA", **meta}
        
    #stat mseed
    def stat(self):
        for comp in [self.ew, self.ns, self.vt]:
            comp.stats()
            
    #bandpass filter in time domain, geopsy tidak menerapkan ini     
    def bandpass_filter(self, flow, fhigh, order):
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)
            
    #detrend in time domain
    def detrend(self):
        for comp in [self.ew, self.ns, self.vt]:
            comp.detrend()
            
    #cosine taper in time domain
    def cosine_taper(self, width):
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)
    
    #plot data in timeseries  
    def plotdata(self):
        for comp in [self.ew, self.ns, self.vt]:
            comp.plot()
            
    #transformasi fourier
    def fouriertransform(self, **kwargs):
        ffts = {}
        for attr in ["ew", "ns", "vt"]:
            tseries = getattr(self, attr)
            dt = tseries.stats.delta
            npts = tseries.stats.npts
            yf = scipy.fftpack.fft(tseries, **kwargs)
            xf = np.linspace(0.0, 1.0/(2.0*dt), int(npts/2))
            plt.figure()
            plt.plot(xf, 2.0/npts * np.abs(yf[:int(npts/2)]),color='green')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.show()
            ffts[attr] = yf
        return ffts
    
    #representation of object
    def __repr__(self):
        return f"Sensor3c(ns={self.ns}, ew={self.ew}, vt={self.vt}, meta={self.meta})"   
    
    

    
    
    
    
    
    
    