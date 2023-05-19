
import obspy as ob
import warnings
#from scipy.fft import fft
import numpy as np

class preprocessing():
    def check_input(values_dict):
        #input harus berupa time series data
        print("You must input a time series data")
        ns = values_dict["ns"]
        
        #check apakah data memiliki dt yang sama dan apakah jumlah sampel sudah sama
        dt = ns.dt
        nsamples = ns.samples
        for key, value in values_dict.item():
            if key == "ns":
                continue
            if value.dt != dt:
                for dt in ns:
                    ar = ns.resample(float(input("sampling rate of the resampled signal: ")))
                    ar += ns
            if value.nsamples != nsamples:
                for nsamples in ns:
                    starttime = ob.UTCDateTime(str(input("input endtime based on format obspy UTCDateTime: ")))
                    endtime = ob.UTCDateTime(str(input("input endtime based on format obspy UTCDateTime": )))
                    ar = ns.trim(starttime, endtime)
        return (values_dict["ns"], values_dict["ew"], values_dict["vt"])
    
    #deklarasi data N (ns), E(ew), Z(vt)
    def __init__(self, ns, ew, vt, meta=None):
        values_dict = {"ns": ns, "ew": ew, "vt": vt}
        self.ns, self.ew, self.vt = self.check_input(values_dict)
        
        meta = {} if meta is None else meta
        self.meta = {"File Name": "NA", **meta}
    
    #read the 3c component in various condition file name
    def from_mseed(cls, fname=None, fnames_1c=None):
        if fname is not None and fnames_1c is None:
            traces = ob.read(fname, format="mseed")
        elif fnames_1c is not None and fname is None:
            trace_list = []
            for key in ["e", "n", "z"]:
                stream = ob.read(fnames_1c[key], format="mseed")
                if len(stream) > 1:
                    msg = f"File {fnames_1c[key]} contained {len(stream)}"
                    msg += "traces, rather than 1 as was expected."
                    raise IndexError(msg)
                trace = stream[0]
                if trace.meta.channel[-1] != key.capitalize():
                    msg = "Component indicated in the header of"
                    msg += f"{fnames_1c[key]} is {trace.meta.channel[-1]}"
                    msg += f"which does not match the key {key} specified."
                    msg += "your digitizer's header is incorrect."
                    warnings.warn(msg)
                    trace.meta.channel = trace.meta.channel[:-1] + key.capitalize()
                trace_list.append(trace)
            traces = ob.Stream(trace_list)
            fname = fnames_1c["n"]
        else:
            msg="`fnames_1c` and `fname` cannot both be defined or both be `None`."
            raise ValueError(msg)
        
        if len(traces) != 3:
            msg = f"Provided {len(traces)} traces, must only provide 3."
            raise ValueError(msg)
        
        found_ew, found_ns, found_vt = False, False, False
        for trace in traces:
            if trace.meta.channel.endswith("E") and not found_ew:
                ew = ts.from_trace(trace)
                found_ew = True
            elif trace.meta.channel.endwith("N") and not found_ns:
                ns = ts.from_trace(trace)
                found_ns = True
            elif trace.meta.channel.endswith("Z") and not found_vt:
                vt = ts.from_trace(trace)
                found_vt = True
            else:
                msg = "Missing, duplicate, or incorrectly named components."
                raise ValueError(msg)
        meta = {"File Name": fname}
        return cls(ns, ew, vt, meta)
    
    #create sensor 3c object from dictionary
    def from_dict(cls, dictionary):
        components = []
        for comp in ["ns", "ew", "vt"]:
            components.append(ts.from_dict(dictionary[comp]))
        return cls(*components, meta=dictionary.get("meta"))
    
    def normalization_factor(self):
        factor = 0
        for attr in ["ns", "ew", "vt"]:
            cmax = np.max(np.abs(getattr(self, attr).amplitude))
            factor = cmax if cmax > factor else factor
        return factor
    
    def split(self, windowlength):
        for attr in ["ew", "ns", "vt"]:
            getattr(self, attr).split(windowlength)
    
    def detrend(self):
        for comp in [self.ew, self.ns, self.vt]:
            comp.detrend()
    
    def bandpass_filter(self, flow, fhigh, order):
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)
    
    def cosine_taper(self, width):
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)
            
    def transform(self, **kwargs):
        ffts = {}
        for attr in ["ew", "ns", "vt"]:
            tseries = getattr(self, attr)
            fft = fft.from_timeseries(tseries, **kwargs)
            ffts[attr] = fft
            return ffts
    def _combine_horizontal_fd(method, ns, ew):
        if method == "squared-average":
            horizontal = np.sqrt((ns.mag*ns.mag + ew.mag*ew.mag)/2)
        elif method == "geometric-mean":
            horizontal = np.sqrt(ns.mag*ew.mag)
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)
        return FourierTransform(horizontal, ns.frequency, dtype=float)
    
    def _make_hvsr(self, method, resampling, bandwidth, f_low=None, f_high=None, azimuth=None):
        if method in ["squared-average", "geometric-mean"]:
            ffts = self.transform()
            hor = self._combine_horizontal_fd(
                method=method, ew=ffts["ew"], ns=ffts["ns"])
            ver = ffts["vt"]
            del ffts
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

        self.meta["method"] = method
        self.meta["azimuth"] = azimuth

        if resampling["res_type"] == "linear":
            frq = np.linspace(resampling["minf"],
                              resampling["maxf"],
                              resampling["nf"])
        elif resampling["res_type"] == "log":
            frq = np.geomspace(resampling["minf"],
                               resampling["maxf"],
                               resampling["nf"])
        else:
            msg = f"`res_type`={resampling['res_type']} has not been implemented."
            raise NotImplementedError(msg)

        hor.smooth_konno_ohmachi_fast(frq, bandwidth)
        ver.smooth_konno_ohmachi_fast(frq, bandwidth)
        hor._amp /= ver._amp
        hvsr = hor
        del ver

        if self.ns.nseries == 1:
            window_length = max(self.ns.time)
        else:
            window_length = max(self.ns.time[0])

        self.meta["Window Length"] = window_length

        return Hvsr(hvsr.amplitude, hvsr.frequency, find_peaks=False,
                    f_low=f_low, f_high=f_high, meta=self.meta)
    
    #dictionary representation of sensor 3c
    def to_dict(self):
        dictionary = {}
        for name in ["ns", "ew", "vt"]:
            value = getattr(self, name).to_dict()
            dictionary[name] = value
        dictionary["meta"] = self.meta
        return dictionary
    
    #iterable representation of sensor 3c object
    def __iter__(self):
        return iter((self.ns, self.ew, self.vt))
    
    #human readable representation of sensor 3c object
    def __str__(self):
        return f"Sensor3c at {id(self)}"
    
    #representation of sensor 3c object
    def __repr__(self):
        return f"Sensor3c(ns={self.ns}, ew={self.ew}, vt={self.vt}, meta={self.meta})"   
    
                    
            
                        
                
        