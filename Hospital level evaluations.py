import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)      
from matplotlib import pyplot as plt
import os
import glob
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

Data=pd.read_csv("../Oxford/Data/iord_extract_20220325/Databases/DatabasePlanedTesting.csv")
print(Data)

# Planned
df=pd.read_csv("../Patients flow prediction/PlannedDaily.csv")
m, c = np.polyfit(df['Actual daily number of discharges'].values.tolist(), df['Predicted daily number of discharges'].values.tolist(), 1)
print(m, c)
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker
import matplotlib.pyplot
fig, ax = plt.subplots(figsize=(8,8), dpi=100)
ax.patch.set_facecolor("white")
ax.minorticks_on()
fig.patch.set_facecolor('white')
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 26
plt.rc('axes',edgecolor='black',linewidth=4.2)
plt.rc('legend',**{'fontsize':22})
plt.rcParams['figure.facecolor'] = 'black'
legend_properties = {'weight':'bold'}
def plot_grouped_df(grouped_df, ax,  x='Actual daily number of discharges', y='Predicted daily number of discharges', cmap = plt.cm.autumn_r):
    #colors = cmap(np.linspace(0.5, 1, len(grouped_df)))
    colors=['black','green','blue']
    for i, (name,group) in enumerate(grouped_df):
        group.plot(ax=ax,
                   kind='scatter', 
                   x=x, y=y,
                   color=colors[i],
                   label = name)
    # Hide grid lines
    ax.grid(False)
    plt.plot([12, 80], [m*12+c, m*80+c], 'k-', color='red')
    plt.legend(prop=legend_properties)
    ax.set_xlabel('Actual',fontweight='bold')
    ax.set_ylabel('Predicted',fontweight='bold')
    ax.yaxis.set_ticks(np.arange(10, 90, 10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels: 
        label.set_fontweight('bold')
    plt.savefig('/Users/jiandong/Oxford/Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/Aggregated Daily Planned Validation.svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig('/Users/jiandong/Oxford/Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/Aggregated Daily Planned Validation.pdf', bbox_inches='tight', transparent=True, dpi=300)

# now we can use this function to plot the groupby data with categorical values
plot_grouped_df(df[['Actual daily number of discharges','Predicted daily number of discharges','Daytype']].groupby('Daytype'),ax)



# Emergency
df=pd.read_csv("../Patients flow prediction/EmergencyDaily.csv")
df
import numpy as np
m, c = np.polyfit(df['Actual daily number of discharges'].values.tolist(), df['Predicted daily number of discharges'].values.tolist(), 1)
print(m, c)

import matplotlib.ticker
fig, ax = plt.subplots(figsize=(8,8), dpi=100)
ax.patch.set_facecolor("white")
ax.minorticks_on()
fig.patch.set_facecolor('white')
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 26
plt.rc('axes',edgecolor='black',linewidth=2.2)
plt.rc('legend',**{'fontsize':22})
plt.rcParams['figure.facecolor'] = 'black'
legend_properties = {'weight':'bold'}


def plot_grouped_df(grouped_df, ax,  x='Actual daily number of discharges', y='Predicted daily number of discharges', cmap = plt.cm.autumn_r):
    #colors = cmap(np.linspace(0.5, 1, len(grouped_df)))
    colors=['black','green','blue']
    for i, (name,group) in enumerate(grouped_df):
        group.plot(ax=ax,
                   kind='scatter', 
                   x=x, y=y,
                   color=colors[i],
                   label = name)
    # Hide grid lines
    ax.grid(False)
    plt.plot([53, 190], [m*53+c, m*190+c], 'k-', color='red')
    plt.legend(prop=legend_properties)
    ax.set_xlabel('Actual',fontweight='bold')
    ax.set_ylabel('Predicted',fontweight='bold')
    ax.yaxis.set_ticks(np.arange(30, 220, 20))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels: 
        label.set_fontweight('bold')
    plt.savefig('/Users/jiandong/Oxford/Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/Aggregated Daily Emergency Validation.svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig('/Users/jiandong/Oxford/Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/Aggregated Daily EMergency Validation.pdf', bbox_inches='tight', transparent=True, dpi=300)

# now we can use this function to plot the groupby data with categorical values
plot_grouped_df(df[['Actual daily number of discharges','Predicted daily number of discharges','Daytype']].groupby('Daytype'),ax)


from scipy.stats import sem
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


# Calculate MSE, MSPE, RMSE, RMSPE, SEM, SEM percentage, MAE, MAPE
# Emergency
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
os.chdir("../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/")
Outcome='Imminent discharge within 24 hours'
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension)) if 'Daily prediction summary DatabaseEmergency' in i]
print(all_filenames)
result=[]
for f in all_filenames:
    TestData=pd.read_csv(f)
    MSE=np.square(np.subtract(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())).mean() 
    MSPE=mean_squared_error(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist(), squared=True)
    RMSE=math.sqrt(MSE)
    RMSPE=mean_squared_error(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist(), squared=False)
    SEM = sem(np.subtract(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())).mean() 
    SEM_percentage=SEM/TestData['Actual number (Imminent discharge within 24 hours)'].mean()
    MAE=mae(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())
    MAPE=mean_absolute_percentage_error(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())
    result.append([f, MSE, MSPE, RMSE, RMSPE, SEM, SEM_percentage, MAE, MAPE])

# Planned
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
os.chdir("../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/")
Outcome='Imminent discharge within 24 hours'
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension)) if 'Daily prediction summary DatabasePlaned' in i]
print(all_filenames)
for f in all_filenames:
    TestData=pd.read_csv(f)
    MSE=np.square(np.subtract(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())).mean() 
    MSPE=mean_squared_error(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist(), squared=True)
    RMSE=math.sqrt(MSE)
    RMSPE=math.sqrt(MSPE)
    SEM = sem(np.subtract(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())).mean() 
    SEM_percentage=SEM/TestData['Actual number (Imminent discharge within 24 hours)'].mean()
    MAE=mae(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())
    MAPE=mean_absolute_percentage_error(TestData['Actual number (Imminent discharge within 24 hours)'].values.tolist(),TestData['Expected number (Imminent discharge within 24 hours)'].values.tolist())
    result.append([f, MSE, MSPE, RMSE, RMSPE, SEM, SEM_percentage, MAE, MAPE])
result=pd.DataFrame(result)    
result.columns=['file', 'MSE', 'MSPE', 'RMSE', 'RMSPE', 
                'SEM', 'SEM_percentage', 'MAE', 'MAPE']
result.to_csv("../Emergency and Planned daily error measures not rounded.csv")   

    


