import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

########################################################################################################################

# Importing Forex Data from 2015 until 2019
data = pd.read_csv (r'C:\Users\Vasco\Documents\Ms Data Science\1Semester\3Research Project 1\FX_Data_Russian\EURUSD_2015_2019_Volume_Data.txt',dtype=str)
data['DATETIME'] = data['DATE']+' '+data['TIME']
# data['DATETIME'] = data['DATE'].map(str)+' '+data['TIME'].map(str)
# data['<DATETIME>'] = data['<DATE>'].str[0:4]+'/'+data['<DATE>'].str[4:6]+'/'+data['<DATE>'].str[6:]+ ' ' + data['<TIME>'].str[0:2]+':'+data['<TIME>'].str[2:4]+':'+data['<TIME>'].str[4:6]
data = data.drop(['TICKER','PER','DATE','TIME'],axis=1)
# print(data.head(5))
# print(data.dtypes)

########################################################################################################################

# Converting the data types
data['DATETIME'] = pd.to_datetime(data['DATETIME'],format='%Y%m%d %H%M%S')#Time in UTC
data['OPEN'] = pd.to_numeric(data['OPEN'],downcast='float')
data['HIGH'] = pd.to_numeric(data['HIGH'],downcast='float')
data['LOW'] = pd.to_numeric(data['LOW'],downcast='float')
data['CLOSE'] = pd.to_numeric(data['CLOSE'],downcast='float')
data['VOL'] = pd.to_numeric(data['VOL'],downcast='integer')
# print(data.head(5))
# print(data.dtypes)

########################################################################################################################

# Basic Variable creation using Time
data['WEEKDAY'] = data.DATETIME.dt.weekday #monday=0
data['MONTH'] = data.DATETIME.dt.month #January=1
data['DAY'] = data.DATETIME.dt.day
data['HOUR'] = data.DATETIME.dt.hour

# Ploting some histograms
plt.figure(1)
plt.subplot(2, 2, 1)
data['MONTH'].hist()
plt.title('Transactions per Month')
plt.subplot(2, 2, 2)
data['DAY'].hist()
plt.title('Transactions per Day of the Month')
plt.subplot(2, 2, 3)
data['WEEKDAY'].hist()
plt.title('Transactions per Weekday')
plt.subplot(2, 2, 4)
data['HOUR'].hist()
plt.title('Transactions per Hours of the Day (UTC TimeZone)')

########################################################################################################################

plt.figure(2)
Timeline = plt.plot(data['DATETIME'],data['OPEN'])
plt.title('Currency pair EUR/USD 2015-2019')
plt.xlabel('Time')
plt.ylabel('Open price')

########################################################################################################################
########################################### MOVING AVERAGES ############################################################
# I have to make it weighted so the newer elements weight more that the older ones

# # Defining a function to create moving averages for our variables
# def SMA(var_name,MAwindow):
#     data[str(var_name+'_SMA'+str(MAwindow))] = data[var_name].rolling(window=MAwindow).mean()
#     return data[str(var_name+'_SMA'+str(MAwindow))]
#
# #Creating the variables
# SMA('OPEN',24)
# SMA('OPEN',50)
# SMA('OPEN',100)
# SMA('OPEN',700)
# SMA('OPEN',10000)
# SMA('CLOSE',24)
# SMA('CLOSE',50)
# SMA('CLOSE',100)
# SMA('CLOSE',700)
# SMA('CLOSE',10000)
# SMA('HIGH',24)
# SMA('HIGH',50)
# SMA('HIGH',100)
# SMA('HIGH',700)
# SMA('HIGH',10000)
# SMA('LOW',24)
# SMA('LOW',50)
# SMA('LOW',100)
# SMA('LOW',700)
# SMA('LOW',10000)

def EMA(var_name,MAspan):
    data[str(var_name+'_EMA'+str(MAspan))] = data[var_name].ewm(span=MAspan,adjust=False).mean()
    return data[str(var_name+'_EMA'+str(MAspan))]

# #Creating the variables
EMA('OPEN', 24) #6hours
EMA('OPEN', 50) #12hours
EMA('OPEN', 100) #Daily
EMA('OPEN', 700) #Weekly
EMA('OPEN', 10000) #Trimestral

plt.figure(3)
ax = plt.subplot()
ax.plot(data['DATETIME'], data['OPEN'], label='Original Data')
ax.plot(data['DATETIME'], data['OPEN_EMA24'], label='OPEN_EMA24')
ax.plot(data['DATETIME'], data['OPEN_EMA50'], label='OPEN_EMA50')
ax.plot(data['DATETIME'], data['OPEN_EMA100'], label='OPEN_EMA100')
ax.plot(data['DATETIME'], data['OPEN_EMA700'], label='OPEN_EMA700')
ax.plot(data['DATETIME'], data['OPEN_EMA10000'], label='OPEN_EMA10000')
ax.legend()
plt.title('Original data (Open) and Moving Averages')

########################################################################################################################

#fourier transform to try to see if there is a constant frequency in the signal
fft_calc = np.fft.fft(data['OPEN'])

n = len(data)
freq = np.fft.fftfreq(n)

plt.figure(4)
plt.title('Fourier Transform on Original Data')
plt.plot(freq[0:int(len(freq)/2)] , np.abs(fft_calc)[0:int(len(fft_calc)/2)]) # only plots half of the spectrum (positive)


########################################################################################################################

#Detrending the timeseries using differencing:
def detrend_diff(var_name):
    diff = [0]
    for i in range(1, len(data[var_name])):
        value = data[var_name][i] - data[var_name][i - 1]
        diff.append(value)
    return diff

#Detrending the timeseries using Moving average(window=10000):
def detrend_MA(var_name):
    diff = data[var_name] - data[str(var_name+'_EMA10000')]
    return diff

#detrending all de variables (I think it's better if we use detrending with MA):
data['open_detrend'] = detrend_MA('OPEN')
# data['close_detrend'] = detrend_MA('CLOSE')
# data['high_detrend'] = detrend_MA('HIGH')
# data['low_detrend'] = detrend_MA('LOW')

########################################################################################################################

plt.figure(5)
plt.plot(data['DATETIME'], data['open_detrend'], label='Detrended Data')
plt.legend()
plt.title('Detrended Data')

print(data.head(5))

########################################################################################################################

# #Exeepriment of fft with a pure sine wave:
# #Creating a signal:
# Fs = 400 #Sampling frequency
# f = 50 #Hz
# sample = 100
# x = np.arange(sample)
# signal = np.sin(2 * np.pi * f * x / Fs)

# #Perform fourrier transform on the signal:
# fft_calc = np.fft.fft(signal)

# #Create the x axis interval to plot the fft:
# freq = np.fft.fftfreq(sample,1./Fs)

# #plotting the original signal and its fft:
# plt.figure(6)
# plt.title('Fourier Transform')
# # plt.plot(x,signal)
# plt.plot(freq[0:int(len(freq)/2)],np.abs(fft_calc)[0:int(len(fft_calc)/2)])
#

############################### END OF EXPERIMENT ###############################

#fourier transform to try to see if there is a constant frequency in the detrended signal

# print(np.count_nonzero(~np.isnan(data['open_detrend']))) #to check how many NAN values there are in the detrended data

# signal = data['open_detrend'][9999:] #because, using simple moving average, the first 10000 are NaN due to the detrending with 10000MA
signal = data['open_detrend'] #use this if detrended with detrend_diff or EMA

fft_calc = np.fft.fft(signal)

n = len(signal)
freq = np.fft.fftfreq(n, .01) #how should be the x axis?

plt.figure(7)
plt.title('Fourier Transform on Detrended Data')
# plt.plot(data['DATETIME'],data['open_detrend'])
plt.plot(freq[0:int(len(freq)/2)], np.abs(fft_calc)[0:int(len(fft_calc)/2)]) # only plots half of the spectrum (positive)

# plt.show()
# print(data['open_detrend'])
########################################################################################################################
# Export Data do csv
data.to_csv('data_modified.csv')
#print(data['open_detrend'].size)
#print(freq.size)
#print(fft_calc.size)

########################################################################################################################
################################## Price Vs Volume & Seasonality Correlation Spearman ##################################
from scipy.stats import spearmanr
print('Correlation Open vs Volume: '+str(np.round(spearmanr(data['OPEN'], data['VOL'])[0], 5))+' (p-value: '+str(np.round(spearmanr(data['OPEN'], data['VOL'])[1], 5))+')')
print('Correlation Open vs Hour: '+str(np.round(spearmanr(data['OPEN'], data['HOUR'])[0], 5))+' (p-value: '+str(np.round(spearmanr(data['OPEN'], data['HOUR'])[1], 5))+')')
print('Correlation Open vs Day: '+str(np.round(spearmanr(data['OPEN'], data['DAY'])[0], 5))+' (p-value: '+str(np.round(spearmanr(data['OPEN'], data['DAY'])[1], 5))+')')
print('Correlation Open vs Weekday: '+str(np.round(spearmanr(data['OPEN'], data['WEEKDAY'])[0], 5))+' (p-value: '+str(np.round(spearmanr(data['OPEN'], data['WEEKDAY'])[1], 5))+')')
print('Correlation Open vs Month: '+str(np.round(spearmanr(data['OPEN'], data['MONTH'])[0], 5))+' (p-value: '+str(np.round(spearmanr(data['OPEN'], data['MONTH'])[1], 5))+')')

########################################################################################################################
############################################# Empirical Mode Decomposition #############################################
from PyEMD import EMD, Visualisation
start = 102420
finish = 104165
emd_month = "January"
emd_year = "2019"
# Define Signal and time
t = data['DATETIME'][start:finish].to_numpy()
# t1 = np.linspace(0, 1, 200) #Experiment
s1 = data['OPEN'][start:finish].to_numpy()
s2 = data['open_detrend'][start:finish].to_numpy()
# s3 = np.cos(11*2*np.pi*t1*t1) + 6*t1*t1 #Experiment pure wave
filename1 = 's1'+ emd_month + emd_year + '.csv'
np.savetxt(filename1, s1, delimiter=",")
filename2 = 's2' + emd_month + emd_year + '.csv'
np.savetxt(filename2, s2, delimiter=",")

# Execute EMD on signal (original vs detrended)
# Setting a max iteration for the EMD
EMD.FIXE = 1
EMD.FIXE_H = 1

IMFs1 = EMD().emd(s1, t)
IMFs2 = EMD().emd(s2, t)
N1 = IMFs1.shape[0]+1 # to see how many IMFs there are to plot
N2 = IMFs2.shape[0]+1 # to see how many IMFs there are to plot
# Plot IMFs
    # Plotting Original Signal
plt.figure(8)
plt.subplot(N1, 1, 1)
plt.plot(t, s1, 'r')
plt.title("Input signal: Open FX Data " + emd_month + " " + emd_year)
plt.xlabel("Time [Date]")
    # Plotting IMFs
for n, imf in enumerate(IMFs1):
    plt.subplot(N1, 1, n+2)
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time")
    filename = 's1imf' + str(n+1) + emd_month + emd_year + '.csv'
    np.savetxt(filename, imf, delimiter=",")

plt.figure(9)
plt.subplot(N2, 1, 1)
plt.plot(t, s2, 'r')
plt.title("Input signal: Open (deterended) FX Data " + emd_month + " " + emd_year)
plt.xlabel("Time [Date]")
for n, imf in enumerate(IMFs2):
    plt.subplot(N2, 1, n+2)
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time")
    filename = 's2imf' + str(n+1) + emd_month + emd_year + '.csv'
    np.savetxt(filename, imf, delimiter=",")
plt.show()

