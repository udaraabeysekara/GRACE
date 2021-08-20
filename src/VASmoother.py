import numpy as np
from kolzur_filter import _kz_coeffs
import pywt
from scipy import signal

import matplotlib.pyplot as plt
import pandas as pd

class VASmoother():

    #constructor 
    def __init__(self, curvature_threshold_1 = 0.0006, curvature_threshold_2 = 0.0006,\
                 threshold_2_min_gap = 21,kz_window_to_start=3, end_n = 1, wavelet_name='sym7',wavelet_mode_name = 'symmetric',\
                max_iterations = 100,initial_smoothness = 0.001, verbose = False):
        
        self.curvature_threshold_1 = curvature_threshold_1
        self.curvature_threshold_2 = curvature_threshold_2
        self.threshold_2_min_gap = threshold_2_min_gap
        self.kz_window_to_start = kz_window_to_start
        self.wavelet_name = wavelet_name
        self.wavelet_mode_name = wavelet_mode_name
        self.max_iterations = max_iterations
        self.initial_smoothness = initial_smoothness
        self.verbose = verbose
        self.end_n = end_n

    def print_parameters(self):
        print('curvature_threshold_1', self.curvature_threshold_1)
        print('curvature_threshold_2', self.curvature_threshold_2)
        print('threshold_2_min_gap', self.threshold_2_min_gap)
        print('kz_window_to_start', self.kz_window_to_start)
        print('wavelet_name', self.wavelet_name)
        print('wavelet_mode_name', self.wavelet_mode_name)
        print('max_iterations', self.max_iterations)
        print('initial_smoothness', self.initial_smoothness)

    #Following set of functions are optional tools.
    
    #This function extend the end of the data set by adding constant values
    def constant_padding_end(self, data,padding_length,padding_constant):
        #data - input data
        #padding_length - number of new data points to add
        #padding_constant - the value of the new data points
        return np.concatenate([data,np.full((padding_length), padding_constant)])
    
    #This function extend the data array adding elements to the beginning of the data set
    def constant_padding_beginning(self, data,padding_length,padding_constant):
        #data - input data 
        #padding_length - number of new data points to add
        #padding_constant - the vlaue of the new data points
        return np.concatenate([np.full((padding_length), padding_constant),data])
    
    #This function does two things
    #1) Takes the last N number of data points and add it back to the data array
    #2) Takes the first N number of data points, flip the N-data points and add it back to the begining of the data array
    #Note that here the last value of the data set get repeated
    def symmetric_padding(self, data, period):
        #data - input data
        #period - number of data points to repeat
        added_to_the_end = np.concatenate([data,np.flip(data[-1*period:])])
        return(np.concatenate([np.flip(added_to_the_end[:period]),added_to_the_end]))

    #This function does two things
    #1) Takes the last N number of data points, drop the last data point and add it back to the data array
    #2) Takes the first N number of data points, flip the array, drop the last data point and add it back to the data array
    #Note the different of this one from symmetric_padding is the last data point does not repeat
    def mirror_padding(self, data, period):
        #data - input data
        #period - the number of data points to add
        added_to_the_end = np.concatenate([data,np.flip(data[-1*period-1:-1])])
        return(np.concatenate([np.flip(added_to_the_end[1:period+1]),added_to_the_end]))

    #This function function does two things
    #1) Takes the last N number of data points and add it back to the data array
    #2) Takes the first N number of data points and add it back to the begining of the data array
    #Note that the difference of this from symmetric_padding is this function does not flip data
    def periodic_padding(slef, data, period):
        #data - input data
        #period - the number of data points to add        
        added_to_the_end = np.concatenate([data,data[-1*period:]])
        return(np.concatenate([added_to_the_end[:period],added_to_the_end]))

    # (x_1,y_1) and (x_2,y_2) are two data points
    #This function calculated the y value at a given data point x_0
    def cal_h(slef, x_1,y_1,x_0,x_2,y_2):
        #(x_1, y_1) - cartesian coordinate of point 1
        #(x_2, y_2) - cartesian coordinate of point 2
        #x_0 - x coordinate of point 2
        m = (y_1 - y_2)/(x_1 - x_2)
        c = ((x_1/x_2)*y_2 - y_1)/((x_1/x_2)-1)
        return m*x_0 + c

    #Raw data has fluctiations. This function finds the places that has valleys and peaks.
    #Then calculate a weight based on the height/depth of the peaks/valleys
    #These weights will be used in smoothing
    #The purpose of the weights is to penalize the valleys/peaks
    def weight_cal(self, input_data,peak_find_data):
        #input_data - The array to clauclate weights
        #peak_find_data - The user has an option to use a different array to find peaks/valleys
    
        # Find peaks(min).   
        peak_indexes = signal.argrelextrema(peak_find_data, np.greater)
        peak_indexes = peak_indexes[0]
 
        # Find valleys(min).
        valley_indexes = signal.argrelextrema(peak_find_data, np.less)
        valley_indexes = valley_indexes[0]

        turn_points = np.sort(np.concatenate((peak_indexes,valley_indexes),axis=0))

        weight_arr = np.zeros(len(input_data))
        if(turn_points.size == 0):
            weight_arr[weight_arr == 0] = 1
            return  weight_arr
    
        first_element = turn_points[0]-1
        if(first_element < 0):
            first_element = 0
        last_element = turn_points[-1]+1
        if(last_element >= len(input_data)):
            last_element = len(input_data) - 1

        turn_points = np.insert(turn_points,0,first_element)
        turn_points = np.append(turn_points,last_element)
    
        peak_arr = np.array([])
        for i in range(1,len(turn_points)-1):
            h = self.cal_h(turn_points[i-1],input_data[turn_points[i-1]],turn_points[i],turn_points[i+1],input_data[turn_points[i+1]])
            peak_arr = np.append(peak_arr, input_data[turn_points[i]]-h)
    
        if (np.std(peak_arr) < 1):
            peak_arr_std = 1
        else:
            peak_arr_std = np.std(peak_arr)
        peak_arr = np.abs((peak_arr-np.mean(peak_arr))/peak_arr_std)
        peak_w = 1./(1+peak_arr)
    
    
        for i in range(0,len(weight_arr)):
            is_in = np.where(turn_points[1:-1] == i)
            if(np.size(is_in) > 0):
                weight_arr[i] = peak_w[is_in[0]]
        

        
        for end_i in range(1,self.end_n+1):
            
            h_end = np.abs(input_data[-1*self.end_n-1] - input_data[end_i])
            h_end_normalized = np.abs(h_end/peak_arr_std)
            end_weight = 1./(1+h_end_normalized)
            weight_arr[-1*end_i] =  end_weight
            
        
        weight_arr[weight_arr == 0] = 1
        weight_arr[np.isnan(weight_arr)] = 1 
        
        return weight_arr*weight_arr

    #This function smooth the data with Kolmogorov–Zurbenko filter.
    #In addition to standard Kolmogorov–Zurbenko filter this function weights the data by an externally provided weight
    def weighted_kz(self, input_data,window,iterations,external_weights):
        #input_data - input data set as a 1D numpy array
        #window - the window size of the KZ filter
        #iterations - number of iterations for the KZ filter
        #external_weights - set of weights, length of the weights should be equal to the length of the input data
        effective_window = window + int(window/2)*2*(iterations-1)
        relative_weight = _kz_coeffs(window, iterations)
        m = int(effective_window/2)
        MA = []
        for i in range(0,len(input_data)):
            pl = i - m
            ph = i + m
            if(pl < 0):
                pl = 0
            if(ph > len(input_data)-1):
                ph = len(input_data)-1
            weight_trim = effective_window - (ph+1-pl)
            if(0 < weight_trim):
                if(pl == 0):
                    kz_weights = relative_weight[weight_trim:]
                else:
                    kz_weights = relative_weight[:-1*weight_trim]
            else:
                kz_weights = relative_weight
            MA.append(np.average(input_data[pl:ph+1],weights=kz_weights*external_weights[pl:ph+1]))
        return np.array(MA)

    #This function smooth the data with a modified version of the Kolmogorov–Zurbenko filter.
    #This modified version uses two external weights.
    #One set of weights is gives penalties to fluctuations in data
    #The second set of weights is the theoritical shape of the curve
    def weighted_kz_theory(self, input_data,window,iterations,external_weights,theory):
        #input_data - input data set as a 1D numpy array
        #window - the window size of the KZ filter
        #iterations -  number of iterations for the KZ filter
        #external_weights - set of weights, length of the weights should be equal to the length of the input data
        #theory - set of weights for a theoritical curve, length of the weights should be equal to the length of the input data
        
        effective_window = window + int(window/2)*2*(iterations-1)
        relative_weight = _kz_coeffs(window, iterations)
        m = int(effective_window/2)
        MA = []
        for i in range(0,len(input_data)):
            pl = i - m
            ph = i + m
            if(pl < 0):
                pl = 0
            if(ph > len(input_data)-1):
                ph = len(input_data)-1
            weight_trim = effective_window - (ph+1-pl)
            if(0 < weight_trim):
                if(pl == 0):
                    kz_weights = relative_weight[weight_trim:]
                else:
                    kz_weights = relative_weight[:-1*weight_trim]
            else:
                kz_weights = relative_weight
            MA.append(np.average(input_data[pl:ph+1],weights=kz_weights*external_weights[pl:ph+1]*theory[pl:ph+1]))
        return np.array(MA)

    #This function smooth the data with a modified version of the Kolmogorov–Zurbenko filter.
    #This modified version uses two external weights.
    #One set of weights is gives penalties to fluctuations in data
    #The second set of weights is the theoritical shape of the curve
    #The use also could give different widow sizes and iterations for each data point
    def variable_window_weighted_kz_theory(self, input_data,window_arr,iterations_arr,external_weights,theory):
        #input_data - input data set as a 1D numpy array 
        #window_arr - the window sizes for the KZ filter, this is a 1D numpy array with same length to the data points
        #iterations_arr - the number of iterations for the KZ filter, this is a 1D numpy array with same length to the data points
        #external_weights - external set of weights
        #theory  - external set of weights to set the expected shape of the smoothed curve
        MA = []
        for i in range(0,len(input_data)):
            window = window_arr[i]
            iterations = iterations_arr[i]
        
            effective_window = window + int(window/2)*2*(iterations-1)
            relative_weight = _kz_coeffs(window, iterations)
            m = int(effective_window/2)

            pl = i - m
            ph = i + m
            if(pl < 0):
                pl = 0
            if(ph > len(input_data)-1):
                ph = len(input_data)-1
            weight_trim = effective_window - (ph+1-pl)
            if(0 < weight_trim):
                if(pl == 0):
                    kz_weights = relative_weight[weight_trim:]
                else:
                    kz_weights = relative_weight[:-1*weight_trim]
            else:
                kz_weights = relative_weight
            MA.append(np.average(input_data[pl:ph+1],weights=kz_weights*external_weights[pl:ph+1]*theory[pl:ph+1]))
        return np.array(MA)

    #This function smooth the data with a modified version of the Kolmogorov–Zurbenko filter.
    #This modified version uses external weights.
    #Weights is gives penalties to fluctuations in data
    #This user also could give different widow sizes and iterations for each data point  
    def variable_window_weighted_kz(self, input_data,window_arr,iterations_arr,external_weights):
        #input_data - input data set as a 1D numpy array 
        #window_arr - the window sizes for the KZ filter, this is a 1D numpy array with same length to the data points
        #iterations_arr - the number of iterations for the KZ filter, this is a 1D numpy array with same length to the data points
        #external_weights - external set of weights
        MA = []

        for i in range(0,len(input_data)):

            window = window_arr[i]
            iterations = iterations_arr[i]
        
            effective_window = window + int(window/2)*2*(iterations-1)
            relative_weight = _kz_coeffs(window, iterations)
            m = int(effective_window/2)
        
            pl = i - m
            ph = i + m
            if(pl < 0):
                pl = 0
            if(ph > len(input_data)-1):
                ph = len(input_data)-1
            weight_trim = effective_window - (ph+1-pl)
            if(0 < weight_trim):
                if(pl == 0):
                    kz_weights = relative_weight[weight_trim:]
                else:
                    kz_weights = relative_weight[:-1*weight_trim]
            else:
                kz_weights = relative_weight
            MA.append(np.average(input_data[pl:ph+1],weights=kz_weights*external_weights[pl:ph+1]))
        return np.array(MA)

    
    #This function calculates the standard diviation in a given window of the input data
    def cal_moving_std(self, input_data,window):
        #input_data - input data set as a 1D numpy array
        #window - window to calculate the stanrad diviation
        count = 0
        tot_std = 0
        for i in range(0,len(input_data)-window):
            count = count + 1
            tot_std  = tot_std + np.std(input_data[i:i+window])
        return tot_std/count

    
    #This function takes the input_data data, calculates the weights, using weight_cal then smooth using KZ 
    #filter window size window_size and number of iterations kz_iterations.
    #This function is repeated until the further smoothing does not change the level of smoothness by a significant value. 
    #Refer the technical note for more details. 
    def perform_initial_smooth(self, input_data, window_size,kz_iterations):
        #input_data - input data set as a 1D numpy array
        #window_size - Window size to use in kz filter
        #kz_iterations - Number of iterations to use in kz filter
        var_arr = np.array([])
    
        weight_arr_itr = self.weight_cal(input_data,input_data)
 
        smooth_itr = self.weighted_kz(input_data,window_size,kz_iterations,weight_arr_itr)
        weight_arr_c = weight_arr_itr
    
        var_arr = np.append(var_arr, self.cal_moving_std(smooth_itr,7))
    
        for i in range(0,9):

            weight_arr_itr = self.weight_cal(input_data,smooth_itr)
            weight_arr_c = weight_arr_c*weight_arr_itr
            smooth_itr = self.weighted_kz(input_data,window_size,kz_iterations,weight_arr_c)
        
            var_arr = np.append(var_arr, self.cal_moving_std(smooth_itr,7))
            if (self.cal_moving_std(smooth_itr,7) == 0):
                return smooth_itr, weight_arr_c
        var_norm = np.abs(1- var_arr[1:]/var_arr[:-1])
    
    
        itr_counter = 10
       
        while np.mean(var_norm[i:i+10]) >= self.initial_smoothness and itr_counter < self.max_iterations:
    
            weight_arr_itr = self.weight_cal(input_data,smooth_itr)
            
            weight_arr_c = weight_arr_c*weight_arr_itr
            weight_arr_c[weight_arr_c == 0] = np.finfo(np.float64).tiny
            
            smooth_itr = self.weighted_kz(input_data,window_size,kz_iterations,weight_arr_c)
        
            var_arr = np.append(var_arr, self.cal_moving_std(smooth_itr,7))
        
            var_norm = np.abs(1- np.array(var_arr)[1:]/np.array(var_arr)[:-1])
            itr_counter = itr_counter + 1
        
        if(itr_counter >= self.max_iterations and self.verbose):
            print('WARNING !! Event after ',self.max_iterations, 'iterations smoothi did not setle.')
        
        return smooth_itr, weight_arr_c


    #This function performs the initial smoothing using the perform_initial_smooth
    #Then apply wavelet to remove high frequency component
    def perform_wvlt_smoothing(self, Input_data, window = 3, kz_iterations=2,\
                               Delta_Smooth = 0):
        #Input_data - Input data as a 1D numpy array
        #waveletname - Name of the wavelet
        #window - window to used in kz filter
        #kz_iterations - number of iterations to use in the kz filter
        #Delta_Smooth
        smoothed_data, final_weight = self.perform_initial_smooth(Input_data,window,kz_iterations)    
    
        #Now remove the high frequency components using wavelet transformation
        
        (cA, cD) = pywt.dwt(smoothed_data, self.wavelet_name,mode=self.wavelet_mode_name)
        (cA_1, cD_1) = pywt.dwt(cA, self.wavelet_name,mode=self.wavelet_mode_name)
        wv_smooth_1 = pywt.idwt(cA_1, None,self.wavelet_name)
        wv_smooth = pywt.idwt(wv_smooth_1, None,self.wavelet_name)    
        wvlt_smoothed = wv_smooth

        #Use the wavelet transformed smoothed data set as the trend.
        #Then use that trend to apply weights in a smoothing window
        #After this step you get a smooth function with trend corrected.
        #In this step you could smooth with a smaller window
        #New Window size = Old window size - Delta_Smooth is the substration
    
        theory_weight = 1.0/wvlt_smoothed

        testing_sm = self.weighted_kz_theory(Input_data,window-Delta_Smooth,kz_iterations,final_weight,theory_weight)

        (cA, cD) = pywt.dwt(testing_sm, self.wavelet_name,mode=self.wavelet_mode_name)
        (cA_1, cD_1) = pywt.dwt(cA, self.wavelet_name,mode=self.wavelet_mode_name)
        wv_smooth_1 = pywt.idwt(cA_1, None,self.wavelet_name)
        wv_smooth = pywt.idwt(wv_smooth_1, None,self.wavelet_name)

        trend_corrected_smooth = wv_smooth

        #During the process of wavelet transformation and inverse wavelet transformation it adds extra data points to the smoothed function.
        #Following loop is a quick solution to get the length of the array right. 
        #May be there is a better way to do this.
    
        Delta = len(trend_corrected_smooth) - len(Input_data)
        right = int(Delta/2)
        left = Delta - right
        if Delta == 0:
            final = trend_corrected_smooth
        elif right <= 0:
            final = trend_corrected_smooth[left:]
        else:
            final = trend_corrected_smooth[left:-1*right]
    
        return final

    def perform_wvlt_smoothing_testing(self, Input_data, window = 3, kz_iterations=2, Delta_Smooth = 0):
        #Input_data - Input data as a 1D numpy array
        #waveletname - Name of the wavelet
        #window - window to used in kz filter
        #kz_iterations - number of iterations to use in the kz filter
        #Delta_Smooth
        smoothed_data, final_weight = self.perform_initial_smooth(Input_data,window,kz_iterations)    
    
        #Now remove the high frequency components using wavelet transformation
        
        (cA, cD) = pywt.dwt(smoothed_data, self.wavelet_name,mode=self.wavelet_mode_name)
        (cA_1, cD_1) = pywt.dwt(cA, self.wavelet_name,mode=self.wavelet_mode_name)
        wv_smooth_1 = pywt.idwt(cA_1, None,self.wavelet_name)
        wv_smooth = pywt.idwt(wv_smooth_1, None,self.wavelet_name)    
        wvlt_smoothed = wv_smooth

        #Use the wavelet transformed smoothed data set as the trend.
        #Then use that trend to apply weights in a smoothing window
        #After this step you get a smooth function with trend corrected.
        #In this step you could smooth with a smaller window
        #New Window size = Old window size - Delta_Smooth is the substration
    
        theory_weight = 1.0/wvlt_smoothed

        testing_sm = self.weighted_kz_theory(Input_data,window-Delta_Smooth,kz_iterations,final_weight,theory_weight)

        (cA, cD) = pywt.dwt(testing_sm, self.wavelet_name,mode=self.wavelet_mode_name)
        (cA_1, cD_1) = pywt.dwt(cA, self.wavelet_name,mode=self.wavelet_mode_name)
        wv_smooth_1 = pywt.idwt(cA_1, None,self.wavelet_name)
        wv_smooth = pywt.idwt(wv_smooth_1, None,self.wavelet_name)

        trend_corrected_smooth = wv_smooth

        #During the process of wavelet transformation and inverse wavelet transformation it adds extra data points to the smoothed function.
        #Following loop is a quick solution to get the length of the array right. 
        #May be there is a better way to do this.
    
        Delta = len(trend_corrected_smooth) - len(Input_data)
        right = int(Delta/2)
        left = Delta - right
        if Delta == 0:
            final = trend_corrected_smooth
        elif right <= 0:
            final = trend_corrected_smooth[left:]
        else:
            final = trend_corrected_smooth[left:-1*right]
    
        return final, smoothed_data, wvlt_smoothed, testing_sm, trend_corrected_smooth


    #Even after applying an aggressive smoothing the curve might still have segments with high curvatures.
    #The user can apply an aggressive smoothing and this function identifies the outliers of the curvaatures
    #The user can select the parameters for teh aggressive smoothing using kz_window, kz_iterations
    def seperate_outlier_curvatures(self, data, kz_window, kz_iterations,\
                                    Delta_Smooth, waveletname = None):
        #data - input data as a 1D numpy array
        #waveletname - name of the wavelet to use for obtaining the aggressively smoothed curve
        #kz_window - the kz filter window for obtaining the aggressively smoothed curve
        #kz_iterations - the number of iterations for obtaining the aggressively smoothed curve
        #Delta_Smooth - one can use different kz window size when correcting for expectuve curve shape
        if waveletname == None:
            waveletname = self.wavelet_name
            
        agressive_smoothed = self.perform_wvlt_smoothing(data, window = kz_window,kz_iterations=kz_iterations,\
                                   Delta_Smooth = Delta_Smooth)
            
        agressive_Rt = agressive_smoothed[1:]/agressive_smoothed[:-1]
        agressive_Rt_d = np.diff(agressive_Rt)
        agressive_Rt_dd = np.diff(agressive_Rt_d)
        agressive_Rt_cur = agressive_Rt_dd*agressive_Rt_dd

        IQR = np.quantile(agressive_Rt_cur, 0.75) - np.quantile(agressive_Rt_cur, 0.25)
        EndWhisker = np.quantile(agressive_Rt_cur, 0.75) + 1.5*IQR
        without_outliers = agressive_Rt_cur[np.where(agressive_Rt_cur < EndWhisker)]
        index_included = np.where(agressive_Rt_cur < EndWhisker)[0]
        index_excluded = np.where(agressive_Rt_cur >= EndWhisker)[0]
    
        return index_included, index_excluded

    #This function combines several fuctions to perform the smoothing
    def run_va_smoother(self, data, Delta_Smooth,index_included):
    
        smoothing_mod = np.empty((0, 2), int)
        kz_window = self.kz_window_to_start
        above_thresh_1 = 5
        thresh_2_gap = self.threshold_2_min_gap + 5
        second_condition = False
    
        curvature_arr = np.empty((0, len(data)-3), float)
    
        while (above_thresh_1 > 0 or not(second_condition)):
            for kz_itr in np.arange(1,4,1):
            
                smoothing_mod = np.append(smoothing_mod,np.array([[kz_window, kz_itr]]),axis=0)
                smoothed = self.perform_wvlt_smoothing(data, window = kz_window,kz_iterations=kz_itr,\
                                                  Delta_Smooth = Delta_Smooth)
                                    
                smoothed_Rt = smoothed[1:]/smoothed[:-1]
                smoothed_Rt_d = np.diff(smoothed_Rt)
                smoothed_Rt_dd = np.diff(smoothed_Rt_d)
                curvature = smoothed_Rt_dd*smoothed_Rt_dd
            
                curvature_arr = np.append(curvature_arr, [curvature], axis=0)
            
                curvature = curvature[index_included]
                   
                
                above_thresh_1 = len(np.where(curvature > self.curvature_threshold_1)[0])
                above_threshold_2 = np.where(curvature > self.curvature_threshold_2)
                if(len(above_threshold_2[0])>1):
                    thresh_2_gap = np.min(np.diff(above_threshold_2))
                    if(thresh_2_gap < self.threshold_2_min_gap):
                        second_condition = True
                    else:
                        second_condition = False
                else:
                    second_condition = True

                if above_thresh_1 <= 0 :#and thresh_2_gap < threshold_2_min_gap:
                    return kz_window,kz_itr,smoothed,curvature_arr,smoothing_mod

        
            kz_window = kz_window + 2
        #Note that kz_window increased by 2 before the while condition checks. 
        #Therefore, we substract the kz_window by 2
        return kz_window-2,kz_itr,smoothed,curvature_arr,smoothing_mod



    #This function identifies different segments in the curve
    #Threshold_for_segmenting is the smoothenes needed in the final smoothed curve
    #min_segment_size is the minimum length of a segment
    def segment_curvature_values(self, curvature_arr, threshold_for_segmenting = 0.0006, min_segment_size = 3):

        iterations_req = np.array([],int)
        #Find how many iterations are needed to achieve the smoothness defined in the threshold
        for i in range(0,np.shape(curvature_arr)[1]):
            row_n_met_condition = np.where(curvature_arr[:,i] > threshold_for_segmenting)
            if (len(row_n_met_condition[0]) == 0):
                iterations_req = np.append(iterations_req,np.shape(curvature_arr)[0]-1)
            else:
                iterations_req = np.append(iterations_req,np.max(row_n_met_condition))

        #Now the iterations_req array has the number of itereations that each data point need to smooth
        #Find the places that the iteration number changes
        break_points = np.where(np.diff(iterations_req) != 0)[0]
        #Generate a two 2 array, each element says the start and the end of a given segment [start,end]
        left = break_points + 1
        left = np.insert(left, 0, 0, axis=0)
        break_points = np.append(break_points, len(iterations_req)-1)
        paired_segments = np.vstack((left, break_points)).T
        #Create an array for the length of the each segment
        segment_length =  break_points - left + 1
        #Create an array with the number of iterations that each segment needed
        segment_itr = np.array([],int)
        for i in range(0,np.shape(paired_segments)[0]):
            segment_itr = np.append(segment_itr, iterations_req[paired_segments[i,0]])    

        while np.min(segment_length) < min_segment_size:
            lowest_index = np.min(np.where(segment_length < min_segment_size))    
            if lowest_index == 0:
                selected_window = np.max([segment_itr[lowest_index],segment_itr[lowest_index+1]])
                #update paired_segments
                start = paired_segments[0,:][0]
                end = paired_segments[1,:][1]
                paired_segments = np.delete(paired_segments,0,axis=0)
                paired_segments[0,:] = [start,end]
                #update segment_length
                segment_length[1] = segment_length[0] + segment_length[1]
                segment_length = np.delete(segment_length,0)
                #update segment_itr
                segment_itr[1] = np.max([segment_itr[0], segment_itr[1]])
                segment_itr = np.delete(segment_itr,0)
            elif(lowest_index == np.shape(paired_segments)[0] - 1):
                #update paired_segments
                start = paired_segments[-2,:][0]
                end = paired_segments[-1,:][1]
                paired_segments = np.delete(paired_segments,-1,axis=0)
                paired_segments[-1,:] = [start,end]
                #update segment_length
                segment_length[-2] = segment_length[-1] + segment_length[-2]
                segment_length = np.delete(segment_length,-1)
                #update segment_itr
                segment_itr[-2] = np.max([segment_itr[-1], segment_itr[-2]])
                segment_itr = np.delete(segment_itr,-1)    
            else:
                if(segment_itr[lowest_index - 1] < segment_itr[lowest_index]):
                    #update paired_segments
                    start = paired_segments[lowest_index - 1,:][0]
                    end = paired_segments[lowest_index ,:][1]
                    paired_segments[lowest_index,:] = [start,end]
                    paired_segments = np.delete(paired_segments,lowest_index - 1,axis=0)        
                    #update segment_length
                    segment_length[lowest_index] = segment_length[lowest_index - 1] + segment_length[lowest_index]
                    segment_length = np.delete(segment_length,lowest_index - 1)
                    #segment_itr
                    segment_itr[lowest_index] = np.max([segment_itr[lowest_index-1], segment_itr[lowest_index]]) 
                    segment_itr = np.delete(segment_itr,0)
                else:
                    #update paired_segments
                    start = paired_segments[lowest_index,:][0]
                    end = paired_segments[lowest_index + 1,:][1]
                    paired_segments[lowest_index+1,:] = [start,end]
                    paired_segments = np.delete(paired_segments,lowest_index,axis=0)        
                    #update segment_length
                    segment_length[lowest_index+1] = segment_length[lowest_index + 1] + segment_length[lowest_index]
                    segment_length = np.delete(segment_length,lowest_index)
                    #segment_itr
                    segment_itr[lowest_index+1] = np.max([segment_itr[lowest_index+1], segment_itr[lowest_index]])
                    segment_itr = np.delete(segment_itr,lowest_index)

        clustered_itr = np.array([],int)
        for freq,itr in zip(segment_length,segment_itr):
            clustered_itr = np.append(clustered_itr,np.full((freq), itr))
        
        return clustered_itr
    
   


    #This function performs initial smoothing with different KZ windows and KZ iterations for each data point
    def variable_window_perform_initial_smooth(self, input_data, window_arr,kz_iterations_arr):
        var_arr = np.array([])
    
        weight_arr_itr = self.weight_cal(input_data,input_data)
    
        smooth_itr = self.variable_window_weighted_kz(input_data,window_arr,kz_iterations_arr,weight_arr_itr)
        weight_arr_c = weight_arr_itr
    
        var_arr = np.append(var_arr, self.cal_moving_std(smooth_itr,7))
    
        for i in range(0,9):

            weight_arr_itr = self.weight_cal(input_data,smooth_itr)
            weight_arr_c = weight_arr_c*weight_arr_itr
            smooth_itr = self.variable_window_weighted_kz(input_data,window_arr,kz_iterations_arr,weight_arr_c)
        
            var_arr = np.append(var_arr, self.cal_moving_std(smooth_itr,7))

        var_norm = np.abs(1- np.array(var_arr)[1:]/np.array(var_arr)[:-1])
    
        itr_counter = 10
       
        while np.mean(var_norm[i:i+10]) >= self.initial_smoothness and itr_counter < self.max_iterations:
    
            weight_arr_itr = self.weight_cal(input_data,smooth_itr)
            weight_arr_c = weight_arr_c*weight_arr_itr
            smooth_itr = self.variable_window_weighted_kz(input_data,window_arr,kz_iterations_arr,weight_arr_c)
        
            var_arr = np.append(var_arr, self.cal_moving_std(smooth_itr,7))
        
            var_norm = np.abs(1- np.array(var_arr)[1:]/np.array(var_arr)[:-1])
            itr_counter = itr_counter + 1
        
        if(itr_counter >= self.max_iterations  and self.verbose):
            print('WARNING !! Event after ',self.max_iterations, 'iterations smoothi did not setle.')
        
        return smooth_itr, weight_arr_c

    #This function applies the waveltet to remove the high frequency component
    def wvlt_cleaning(self, smoothed_data, unsmoothed_data,window_arr, iterations_arr, final_weight,Delta_Smooth=0):
   
        #Now remove the high frequency components using wavelet transformation
        (cA, cD) = pywt.dwt(smoothed_data, self.wavelet_name,mode=self.wavelet_mode_name)
        (cA_1, cD_1) = pywt.dwt(cA, self.wavelet_name,mode=self.wavelet_mode_name)
        wv_smooth_1 = pywt.idwt(cA_1, None,self.wavelet_name)
        wv_smooth = pywt.idwt(wv_smooth_1, None,self.wavelet_name)    
        wvlt_smoothed = wv_smooth

        #Use the wavelet transformed smoothed data set as the trend.
        #Then use that trend to apply weights in a smoothing window
        #After this step you get a smooth function with trend corrected.
        #In this step you could smooth with a smaller window
        #New Window size = Old window size - Delta_Smooth is the substration
    
        theory_weight = 1.0/wvlt_smoothed
        #window_arr = window_arr - Delta_Smooth
    
        testing_sm = self.variable_window_weighted_kz_theory(unsmoothed_data,window_arr,iterations_arr,final_weight,theory_weight)
        (cA, cD) = pywt.dwt(testing_sm, self.wavelet_name,mode=self.wavelet_mode_name)
        (cA_1, cD_1) = pywt.dwt(cA, self.wavelet_name,mode=self.wavelet_mode_name)
        wv_smooth_1 = pywt.idwt(cA_1, None,self.wavelet_name)
        wv_smooth = pywt.idwt(wv_smooth_1, None,self.wavelet_name)

        trend_corrected_smooth = wv_smooth

        #During the process of wavelet transformation and inverse wavelet transformation it adds extra data points to the smoothed function.
        #Following loop is a quick solution to get the length of the array right. 
        #May be there is a better way to do this.
        Delta = len(trend_corrected_smooth) - len(smoothed_data)
        right = int(Delta/2)
        left = Delta - right
        if Delta == 0:
            final = trend_corrected_smooth
        elif right <= 0:
            final = trend_corrected_smooth[left:]
        else:
            final = trend_corrected_smooth[left:-1*right]
    
        return final


    #This function combines several functions and perform the smoothing with different smoothing parameters
    #The parameters are identified by segment_curvature_values
    def segment_wise_smoothing(self, raw_data,curvature_arr,smoothing_mods, min_segment_size,\
                              threshold_for_segmenting = None):
        if(threshold_for_segmenting == None):
            threshold_for_segmenting = self.curvature_threshold_1
        identified_segments = self.segment_curvature_values(curvature_arr,threshold_for_segmenting = threshold_for_segmenting\
                                                       , min_segment_size = min_segment_size)

        window_arr = np.array([],int)
        iterations_arr = np.array([],int)

        for i in identified_segments:
            window_arr = np.append(window_arr, smoothing_mods[i][0])
            iterations_arr = np.append(iterations_arr, smoothing_mods[i][1])
    
        window_arr = np.append(window_arr,window_arr[:-3])
        iterations_arr = np.append(iterations_arr, iterations_arr[:-3])

        segmented_smooth, segmented_weight = self.variable_window_perform_initial_smooth(raw_data,\
                                                                                         window_arr,iterations_arr)

        segmented_smooth_curve = self.wvlt_cleaning(segmented_smooth,raw_data,window_arr,iterations_arr,segmented_weight)
        return segmented_smooth_curve




