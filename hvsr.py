
import numpy as np
import fast_konno_ohmachi as fko
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from obspy.core import UTCDateTime
import obspy
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use('TkAgg')  # Use the Tkinter backend

file_path = "D:\\magang_software\\HVSR-master\\bin\\UT.STN12.A2_C150.miniseed"
# file_path = 'Z6.AF1.HH_1.mseed'

# start = '2004-05-21T00:00:00.000000'
# end = '2004-05-21T00:30:00.000000'

#Membaca data MiniSEED 
stw = obspy.read(file_path)
# stw = stb.trim(UTCDateTime(start),UTCDateTime(end))
#prepro sederhana
st = stw.resample(10.0)
sta = st.detrend("linear")
sta = sta.taper(max_percentage=0.05,type='cosine')
tr_copy = sta.copy()
tr_copy.filter('bandpass', freqmin= 0.1, freqmax= 5, corners=4, zerophase=True)
#st.plot()
# Memecah komponen X, Y, dan Z
trace_x = tr_copy.select(channel="*N")[0]
trace_y = tr_copy.select(channel="*E")[0]
trace_z = tr_copy.select(channel="*Z")[0]

class hvsr:
    def __init__(self, trace_n, trace_w, trace_v, size):
        self.trace_n = trace_n
        self.trace_w = trace_w
        self.trace_v = trace_v
        self.size = size
    
    def time(self, trace):
        """
        function for limit the time in UTC and log it into pandas to recalled later.
        
        keyword argument:
        trace -- trace in the stream
        
        return: time table with 2 column.
        """
        self.trace = trace
        start = self.trace.stats.starttime
        end = self.trace.stats.endtime
        #make a pandas for log the time limit
        self.df = pd.DataFrame(columns=['start','end'])
        i = 0
        #for limit the time
        while start + i * self.size < end:
            self.df = pd.concat([self.df, pd.DataFrame({'start': start + i * self.size,'end': 
                min(start + (i + 1) * self.size, end)}, index=[0])], ignore_index=True)  # noqa: E501
            i += 1
        return self.df
        
    def window(self,trace):
        """ 
        function to cut the trace into several windows according to the specified size.
        
        keyword argument:
        trace -- trace in the stream
        
        return: array with window chunks
        """
        self.time(trace)
        self.result = []
        #table reading to retrieve the contents of the table in the form of a range window
        for _, row in self.df.iterrows():
            st = UTCDateTime(row['start'])
            en = UTCDateTime(row['end'])
            if st != en:
                #create a loop to cut the trace
                trim = trace.slice(st, en)
                if len(trim) > 0:
                    self.result.append(trim)
        return self.result

    def fft(self, window):
        """ 
        function for Fast Fourier Transform for each window.
        
        Keyword arguments:
        window -- trace that has been cut into several windows
        
        return: result from Fast Fourier Transform in array
        """
        res_win = window
        fft_res = []
        for i in range(len(res_win)):
            fft = np.fft.fft(res_win[i])
            fft_res.append(fft)
        return fft_res
    # def reject_windows(self,freq,spec):
        
        
    def get_hvsr_smooth(self):  # noqa: E999
        """ 
        main process in HVSR, function for count the HVSR velue and plot it.
        """
        win_n, win_w, win_v = self.window(self.trace_n), self.window(self.trace_w), self.window(self.trace_v)  # noqa: E501
        n ,w= self.fft(win_n), self.fft(win_w)
        #combining horizontal with geomatric mean
        H =  [np.sqrt(x * y) for x, y in zip(n, w)]
        V = self.fft(win_v)
        #combining Horizontal and vertical into H/V
        hvsr = [x / y for x, y in zip(H, V)]

        #takes a sample, length window of either traces
        if len(win_v[0]) == len(win_n[0]) == len(win_w[0]):
            win_lenght = len(win_v[0])
        #make amplitude/spectrum for HVSR in array
        spec = []
        for i in range(len(hvsr)):
            no_abs = hvsr[i][:int(win_lenght/2)]
            spec.append(np.abs(no_abs))

        #get the frequency for  X-axis
        freq = np.fft.fftfreq(win_lenght, d=1/win_v[0].stats.sampling_rate)
        freq_x = freq[:int(win_lenght/2)]
        #get array smoothing result for Y-axis 
        smooth_res = []
        for i in range(len(spec)):
            if (spec[i].shape)!=(freq_x.shape):
                sel = np.int(np.abs(tuple(x - y for x, y in zip(spec[i].shape, freq_x.shape))))  # noqa: E501
                if (spec[i].shape)<(freq_x.shape):
                    freq_x = freq_x[:-sel]    
                elif (spec[i].shape)>(freq_x.shape):
                    spec[i] = spec[i][:-sel]
            smooth = fko.slow_konno_ohmachi(spec[i], freq_x, smooth_coeff=40)
            smooth_res.append(smooth)
            
        median = self.median(smooth_res)
        mean = self.mean(smooth_res)
        
        # self.plot(freq_x,smooth_res,median=median)
        self.plot(freq_x,smooth_res,mean=mean)
    
    def plot(self, x_axis, data, mean=None, median=None, x_lim=None, y_lim=None):
        """ 
        function for plot the mean or the median from H/V method.
        
        keyword argument:
        x_axis -- frquency to plot in x-axis 
        mean -- function mean to count mean velue and standart deviation data
        median -- fungtion median to count median velue and inerquartil data
        x_lim -- integer to get or set the x limits of the current axes
        y_lim -- integer to get or set the y limits of the current axes
        """
        for i in data:
            plt.plot(x_axis,np.log(i))
        if mean is not None:
            _mean, std = mean
            self.mean_freq_peak(data,x_axis,_mean) 
            plt.plot(x_axis,np.log(_mean) - std,label='Median Curve', color='black',linestyle = 'dashed') # noqa: E501
            plt.plot(x_axis,np.log(_mean),label='Median Curve', color='black')
            plt.plot(x_axis,np.log(_mean) + std,label='Median Curve', color='black',linestyle = 'dashed') # noqa: E501
        if median is not None:
            self.med_freq_peak(data,x_axis,median[1]) 
            plt.plot(x_axis,median[0],label='Median Curve', color='black',linestyle = 'dashed') # noqa: E501
            plt.plot(x_axis,median[1],label='Median Curve', color='black')
            plt.plot(x_axis,median[2],label='Median Curve', color='black',linestyle = 'dashed')  # noqa: E501
        plt.xlim(0,x_lim)
        plt.ylim(0,y_lim)
        plt.show()
    
    def med_freq_peak(self, data, freq, median):
        freq_velue,max = self.peak_n_freq(data, freq,median, 1)
        quartil = []
        q1, q2 = np.percentile(sorted(freq_velue), [25, 75], interpolation='midpoint')
        quartil.append((q1,q2))
        plt.fill_betweenx(freq*10, q1,q2, facecolor='green', alpha=.5)
        # print(quartil)
        return quartil
              
    def mean_freq_peak(self, data, freq, mean) :
        freq_velue, max = self.peak_n_freq(data, freq, mean)
        print(max)
        _std = np.std(np.log(freq_velue))
        max_r = max[0]+_std
        max_l = max[0]-_std
        plt.fill_betweenx(freq*10, max[0],max_r, facecolor='grey', alpha=.5)
        plt.fill_betweenx(freq*10, max_l,max[0], facecolor='red', alpha=.5)
        return _std
    
    def _find_indices(self, array, *values):
        indices = [int(np.where(array == value)[0]) for value in values]
        return indices

    def peak_n_freq(self, data, freq, medmen, lim=0):
        max, min= self.peakdetect(np.log(medmen), freq)
        x_max = [max[0]]
        y_max = [max[1]]
        plt.plot(x_max, y_max, marker='x', color='black', linestyle='None')
        for entry in min:
            x_min = [entry[0]]
            y_min = [entry[1]]
            plt.plot(x_min, y_min, marker='o', color='black', linestyle='None')
            
        freq_velue=[]
        if lim == 1:
            _lim_min = [x[0] for x in min]
            _lim_max = [max[0]]
            limit_min= self._find_indices(freq,*_lim_min)
            limit_mx= self._find_indices(freq,_lim_max)
            print('limit=',limit_min)
            if limit_min < limit_mx:
                for velue in data:
                    _max, _min= self.peakdetect(velue[limit_min[0]:], freq[limit_min[0]:])  # noqa: E501
                    freq_velue.append(_max[0])
            if limit_min > limit_mx:
                for velue in data:
                    _max, _min= self.peakdetect(velue[:limit_min[0]], freq[:limit_min[0]])  # noqa: E501
                    freq_velue.append(_max[0])
        elif lim == 0:
            for velue in data:
                _max, _min= self.peakdetect(velue, freq)  # noqa: E501
                freq_velue.append(_max[0])
        return freq_velue, max
            
    def mean(self, spec):
        spec = np.array(spec)
        spec_mean = spec.sum(axis=0) / spec.shape[0]
        std = np.std(np.log(spec), axis = 0)
        
        spec_mean = self.moving_average(spec_mean, 3)
        std = self.moving_average(std, 3)
        return spec_mean, std
    
    def moving_average(self, data, movingwin_size):
        result = []
        for i in range (len(data)):
            window_avg = round(np.sum(data[
                i-movingwin_size:i+movingwin_size])/((movingwin_size*2)), 2)
            result.append(window_avg)
            data[i]=window_avg
        return np.array(result)
            
    def median(self, spec):
        self.median_result = [], [], []
        num_columns = len(spec[0])
        for i in range(num_columns):
            col_vel = [spec[j][i] for j in range(len(spec))]
            col_vel.sort()
            q1, med_vel, q2 = np.percentile(col_vel, [25, 50, 75], interpolation='midpoint')  # noqa: E501
            self.median_result[0].append(q1)
            self.median_result[1].append(med_vel)
            self.median_result[2].append(q2)
        return self.median_result
    
    def _datacheck_peakdetect(self,x_axis, y_axis):
        if x_axis is None:
            x_axis = range(len(y_axis))
        
        if len(y_axis) != len(x_axis):
            raise ValueError( 
                    "Input vectors y_axis and x_axis must have same length")
        
        #needs to be a numpy array
        y_axis = np.array(y_axis)
        x_axis = np.array(x_axis)
        return x_axis, y_axis
    
    def peakdetect(self, y_axis, x_axis, lookahead=200, delta=0):
        max_peaks = []
        min_peaks = []
        dump = []  # Used to pop the first hit which almost always is false

        y_axis = [float(y) for y in y_axis]
        x_axis = [float(x) for x in x_axis]
        lookahead = int(len(y_axis)/20)

        # check input data
        x_axis, y_axis = self._datacheck_peakdetect(x_axis, y_axis)
        # store data length for later use
        length = len(y_axis)

        # perform some checks
        if lookahead < 1:
            raise ValueError("Lookahead must be '1' or above in value")
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError("delta must be a positive number")

        # maxima and minima candidates are temporarily stored in
        # mx and mn respectively
        mn, mx = np.Inf, -np.Inf

        # Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
                mnpos = x

            ####look for max####
            if y < mx - delta and mx != np.Inf:
                if y_axis[index:index + lookahead].max() < mx:
                    max_peaks.append([mxpos, mx])
                    dump.append(True)
                    # set algorithm to only find minima now
                    mx = np.Inf
                    mn = np.Inf
                    if index + lookahead >= length:
                        # end is within lookahead no more peaks can be found
                        break
                    continue
    
            ####look for min####
            if y > mn + delta and mn != -np.Inf:
                # Minima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].min() > mn:
                    min_peaks.append([mnpos, mn])
                    dump.append(False)
                    # set algorithm to only find maxima now
                    mn = -np.Inf
                    mx = -np.Inf
                    if index + lookahead >= length or len(min_peaks) >= 2:
                        # end is within lookahead no more peaks can be found
                        break
        # Remove the false hit on the first value of the y_axis
        try:
            if dump[0]:
                max_peaks.pop(0)
            else:
                min_peaks.pop(0)
            del dump
        except IndexError:
            # no peaks were found, should the function return empty lists?
            pass
    
        # print(sorted(max_peaks, key=lambda xx: xx[1]), min_peaks)
        return [sorted(max_peaks, key=lambda xx: xx[1])[0], min_peaks]


run = hvsr(trace_x,trace_y,trace_z, 60)
run.get_hvsr_smooth()
