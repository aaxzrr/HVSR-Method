import numpy as np
import fast_konno_ohmachi as fko
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from obspy.core import UTCDateTime
warnings.filterwarnings("ignore")
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

    def get_hvsr_smooth(self):
        """ 
        main process in HVSR, function for count the HVSR velue and plot it.
        """
        win_n, win_w, win_v = self.window(self.trace_n), self.window(self.trace_w), self.window(self.trace_v)
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
                sel = np.int(np.abs(tuple(x - y for x, y in zip(spec[i].shape, freq_x.shape))))
                if (spec[i].shape)<(freq_x.shape):
                    freq_x = freq_x[:-sel]    
                elif (spec[i].shape)>(freq_x.shape):
                    spec[i] = spec[i][:-sel]
            smooth = fko.slow_konno_ohmachi(spec[i], freq_x, smooth_coeff=40)
            #plot for each windows
            plt.plot(freq_x,smooth)
            smooth_res.append(smooth)
        self.plot(freq_x,median= self.median(smooth_res),x_lim=2,y_lim=10)
    
    def plot(self,x_axis, mean=None, median=None, x_lim=None, y_lim=None):
        """ 
        function for plot the mean or the median from H/V method.
        
        keyword argument:
        x_axis -- frquency to plot in x-axis 
        mean -- function mean to count mean velue and standart deviation data
        median -- fungtion median to count median velue and inerquartil data
        x_lim -- integer to get or set the x limits of the current axes
        y_lim -- integer to get or set the y limits of the current axes
        """
        if mean is not None:
            mean, std = mean
            plt.plot(x_axis,mean - std,label='Median Curve', color='black',linestyle = 'dashed')
            plt.plot(x_axis,mean,label='Median Curve', color='black')
            plt.plot(x_axis, mean + std,label='Median Curve', color='black',linestyle = 'dashed')
        elif median is not None:
            plt.plot(x_axis,median[0],label='Median Curve', color='black',linestyle = 'dashed')
            plt.plot(x_axis,median[1],label='Median Curve', color='black')
            plt.plot(x_axis,median[2],label='Median Curve', color='black',linestyle = 'dashed')
        plt.xlim(0,x_lim)
        plt.ylim(0,y_lim)
        plt.show()
        
    def mean(self, spec):
        spec = np.array(spec)
        spec_mean = spec.sum(axis=0) / spec.shape[0]
        std = np.std(spec, axis = 0)
        
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
            q1, med_vel, q2 = np.percentile(col_vel, [25, 50, 75], interpolation='midpoint')
            self.median_result[0].append(q1)
            self.median_result[1].append(med_vel)
            self.median_result[2].append(q2)
        return self.median_result
