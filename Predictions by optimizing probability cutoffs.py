from numpy import argmax
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)      
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
plt.style.use('fivethirtyeight') 
import os
import glob
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
import holoviews as hv
from bokeh.io import show


# aggregate predictions of specific prior LOS model
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
os.chdir("../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/")
Outcome='Imminent discharge within 24 hours'
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension)) if 'Predicted Emergency' in i and 'testing' in i]
print(all_filenames)

filesavename='Aggregated emergency'

# select best model-specific probability threshold by 
# 1. J-statistic cutoff optimizing approach
# 2. F1 score cutoff optimizing approach
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']) 
dfs=[]
start_date = date(2019, 2, 2)
end_date = date(2020, 1, 31)
for f in all_filenames:
    TestData=pd.read_csv(Path+f)
    TestData['Index date']=pd.to_datetime(TestData['Index date'], format='%Y-%m-%d')
    TestData['group']=f
    print(TestData['Index date'].value_counts(), TestData.shape)
    plt.hist(TestData['Predicted probability (Imbalanced Data Model)'])
    plt.xlabel('Predicted 24-hour discharge probability for patients in '+f.split(' ')[1])
    plt.ylabel('Number of admissions')
    plt.tight_layout()
    plt.savefig(Path+'Predicted 24-hour discharge probability for patients in '+f.split(' ')[1]+'.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    saveTestData=[] 
    for now_date in daterange(start_date, end_date):
        TestData_day_t=TestData[TestData['Index date']==now_date.strftime("%Y-%m-%d")]  
        # J-statistic cutoff optimizing approach
        if TestData_day_t[Outcome].sum()>0:
            threshold = Find_Optimal_Cutoff(TestData_day_t[Outcome], TestData_day_t['Predicted probability (Imbalanced Data Model)'])
        else:
            threshold=[1]   
        TestData_day_t['Predicted with YoudenJ-optimized cutoff']=TestData_day_t['Predicted probability (Imbalanced Data Model)'].map(lambda x: 1 if x > threshold[0] else 0)        
        # F1 score cutoff optimizing approach
        if TestData_day_t[Outcome].sum()>2:
            precision, recall, thresholds = precision_recall_curve(TestData_day_t[Outcome], TestData_day_t['Predicted probability (Imbalanced Data Model)'])
            fscore = (2 * precision * recall) / (precision + recall)
            threshold = thresholds[argmax(fscore)]
        else:
            threshold=1
        TestData_day_t['Predicted with F1-score-optimized cutoff']=TestData_day_t['Predicted probability (Imbalanced Data Model)'].map(lambda x: 1 if x >threshold  else 0)
        saveTestData.append(TestData_day_t)
    df = pd.concat(saveTestData, axis=0)
    df['Model']=f.split(' ')[1]
    dfs.append(df)
  
# number of patients in hospital on each index date in test data
TestData = pd.concat(dfs, axis=0)
#combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
TestData['Model'].value_counts()
print(TestData['Index date'].value_counts(), TestData.shape)
pd.DataFrame(TestData.groupby('Index date')['Predicted probability (Balanced Data Model)'].sum()).to_csv("../Summary/DailyPatientNumberStats.csv")


# Patient groups by duration from latest admission to decision day
TestData['AdmissionDate']=pd.to_datetime(TestData['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
TestData['Index date']=pd.to_datetime(TestData['Index date'], dayfirst=True).dt.tz_localize(None)
TestData['Adm2index, hours']=divmod((TestData['Index date']-TestData['AdmissionDate']).dt.total_seconds(), 3600)[0]
conditions=[(TestData['Adm2index, hours']>=24*0) & (TestData['Adm2index, hours']<24*1),
            (TestData['Adm2index, hours']>=24*1) & (TestData['Adm2index, hours']<24*2),
            (TestData['Adm2index, hours']>=24*2) & (TestData['Adm2index, hours']<24*3),
            (TestData['Adm2index, hours']>=24*3) & (TestData['Adm2index, hours']<24*4),
            (TestData['Adm2index, hours']>=24*4) & (TestData['Adm2index, hours']<24*5),
            (TestData['Adm2index, hours']>=24*5) & (TestData['Adm2index, hours']<24*6),
            (TestData['Adm2index, hours']>=24*6) & (TestData['Adm2index, hours']<24*11),
            (TestData['Adm2index, hours']>=24*11) & (TestData['Adm2index, hours']<24*14),
            (TestData['Adm2index, hours']>=24*14) & (TestData['Adm2index, hours']<24*21),
            (TestData['Adm2index, hours']>=24*21) & (TestData['Adm2index, hours']<24*28),
            (TestData['Adm2index, hours']>=24*28)]
choices=['Being in hospital 0-24 hours (D1)', 'Being in hospital 24-48 hours (D2)',
         'Being in hospital 48-72 hours (D3)', 'Being in hospital 72-96 hours (D4)',
         'Being in hospital 96-120 hours (D5)', 'Being in hospital 120-144 hours (D6)',
         'Being in hospital 144-264 hours (D7-D10)', 'Being in hospital 264-336 hours (D11-D13)',
         'Being in hospital 336-504 hours (D13-D20)', 'Being in hospital 504-672 hours (D21-D27)',
         'Being in hospital >672 hours (D28+)']
TestData['Being in hospital group'] = np.select(conditions, choices)



# Equal classification cut-off probability approach
from sklearn.metrics import confusion_matrix
# daily model measure confidence interval summary        
DailyResults=[]
DailyAccuracy=[]
for now_date in daterange(start_date, end_date):
    TestData_day_t=TestData[TestData['Index date']==now_date.strftime("%Y-%m-%d")]
    ExpectedOutcomeNumber=TestData_day_t['Predicted probability (Imbalanced Data Model)'].sum()
    ActualOutcomeNumber=TestData_day_t[Outcome].sum()
    real=TestData_day_t[Outcome].values.tolist()
    predicted=TestData_day_t['Predicted (Imbalanced Data Model)'].values.tolist()
    # Calculate overall model evaluation measures
    if sum(real)>4:
        CM=confusion_matrix(real, predicted)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # F1 score
        F1=2*PPV*TPR/(PPV+TPR)
        DailyResults.append([now_date, roc_auc_score(real, predicted), TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, F1,TestData_day_t.shape[0], ActualOutcomeNumber, ExpectedOutcomeNumber])
    else:
        DailyResults.append([now_date, '', '', '', '', '', '', '', '', '', '',TestData_day_t.shape[0], ActualOutcomeNumber, ExpectedOutcomeNumber])   
    # Calculate daily accuracy
    if TestData_day_t.shape[0]>0:
        if ActualOutcomeNumber>0 and (np.round(ExpectedOutcomeNumber, 0)<2*ActualOutcomeNumber):
            dayaccuracy=np.round((1-np.abs(np.round(ExpectedOutcomeNumber, 0)-ActualOutcomeNumber)/ActualOutcomeNumber)*100,2)
            DailyAccuracy.append([now_date, dayaccuracy])
        elif ActualOutcomeNumber==np.round(ExpectedOutcomeNumber, 0) and np.round(ExpectedOutcomeNumber, 0)==0:
            DailyAccuracy.append([now_date, 100])
        else:
            DailyAccuracy.append([now_date, 0.00])
    else:
        if ActualOutcomeNumber==np.round(ExpectedOutcomeNumber, 0) and np.round(ExpectedOutcomeNumber, 0)==0:
            DailyAccuracy.append([now_date, 100])
        else:
            DailyAccuracy.append([now_date, np.nan])
DailyResults=pd.DataFrame(DailyResults)
DailyResults.columns=['Index date', 'ROC AUC score', 'Sensitivity, hit rate, recall, or true positive rate',
                      'Specificity or true negative rate','Precision or positive predictive value',
                      'Negative predictive value','Fall out or false positive rate',
                     'False negative rate','False discovery rate','Overall accuracy','F1 score','Number of patients', 'Actual number ('+Outcome+')', 'Expected number ('+Outcome+')']
DailyAccuracy=pd.DataFrame(DailyAccuracy)   
DailyAccuracy.columns=['Index date', 'Daily prediction accuracy (%)']
DailyResults['Actual daily probability ('+Outcome+')']=DailyResults['Actual number ('+Outcome+')']/DailyResults['Number of patients']
DailyResults['Predicted daily probability ('+Outcome+')']=DailyResults['Expected number ('+Outcome+')']/DailyResults['Number of patients']
DailyResults['Actual daily probability ('+Outcome+')']=DailyResults['Actual daily probability ('+Outcome+')'].replace(np.nan, 0)
DailyResults['Predicted daily probability ('+Outcome+')']=DailyResults['Predicted daily probability ('+Outcome+')'].replace(np.nan, 0)
DailyResults['Daily residual ('+Outcome+')']=DailyResults['Expected number ('+Outcome+')']-DailyResults['Actual number ('+Outcome+')']
DailyResults['Daily prediction accuracy (%)']=DailyAccuracy['Daily prediction accuracy (%)']
print(DailyResults)
########################### Results visualization
# 1. DailyMetrics histograms with best daily probability cutoff
fig = plt.figure(figsize=(14,7), dpi=300)
DailyResults['Daily prediction accuracy (%)']=DailyResults['Daily prediction accuracy (%)']
ax=DailyResults[['ROC AUC score', 'Sensitivity, hit rate, recall, or true positive rate',
                      'Specificity or true negative rate','Precision or positive predictive value',
                      'Negative predictive value', 'Overall accuracy','F1 score', 'Daily prediction accuracy (%)']].apply(pd.to_numeric, errors='coerce').hist(figsize=(15,9),bins=100, grid=True,color='green', zorder=2, rwidth=0.9, ec="#eeeeee", alpha=0.9)
plt.tight_layout()
plt.show()
fig = ax[0][0].get_figure()
fig.savefig(Path+'DailyMetrics histograms by equal classification approach '+' for cutoff '+filesavename+'.svg', bbox_inches='tight', transparent=False, dpi=300)
# 2. Daily predicion comparision with best daily probability cutoff
plotData=DailyResults[['Number of patients',
       'Expected number (Imminent discharge within 24 hours)',
       'Actual number (Imminent discharge within 24 hours)']]
plotData.columns=['Emergency patients in hospital',
       'Model predicted 24-hour discharges',
       'Actual 24-hour discharges']
index = pd.date_range(start = "2019-02-01", end = "2020-01-31", freq = "D")
index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in index]
plotData.index=index
fig, ax = plt.subplots(figsize=(25, 10),dpi=300)
plotData.plot(label='plotData', color=['red', 'blue','orange'], lw=1, style="-o", ms=4, grid=False, fontsize=14, ax=ax)
ax.set_xticks(plotData.index)
# ax.set_title("Predicting 24-hour discharge for emergency patients \n (Aggregating patients being in hospital for different times)", fontweight="bold", fontsize=16)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.gcf().autofmt_xdate()
ax.figure.autofmt_xdate(rotation=45, ha='center')
plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.16),fontsize=14)
plt.xlabel("Time from 2019/2/1-2020/1/31", fontsize=16)
plt.ylabel("Number of patients", fontsize=18)
plt.show()
fig.savefig(Path+'Daily predicion comparision with by equal classification cutoff approach '+filesavename+'.svg', dpi=300, bbox_inches='tight',transparent=True)
# 3. Daily calibration validation with best cutoff
plotData=DailyResults
plotData['Ordered day in 2019/02/01-2020/01/31']=DailyResults['Actual daily probability ('+Outcome+')'].rank(method='dense', ascending=True).tolist()  
plotData.sort_values(by=['Ordered day in 2019/02/01-2020/01/31'], inplace=True, ascending=True)
plotData['Ordered day in 2019/02/01-2020/01/31']=np.arange(plotData.shape[0])
plotData.to_csv(Path+'Combined Daily prediction summary '+' '+Outcome+' by F1 score cutoff approach '+filesavename+'.csv', index=False)
# plot prediction gap histograms
plot=plotData.hvplot.scatter(legend='top_left',width=800, height=400,
                    stacked=True,cmap='Category20', title='XGBoost model to predict '+Outcome+' for emergency patients being in hospital on each index date',
                    x="Ordered day in 2019/02/01-2020/01/31", 
                    y=['Actual daily probability ('+Outcome+')', 
                       'Predicted daily probability ('+Outcome+')'])
hv.extension('bokeh')
# going to use show() to open plot in browser
from bokeh.plotting import show
show(hv.render(plot))
renderer = hv.renderer('bokeh')
renderer.save(plot, 'Daily prediction summary '+filesavename+' '+Outcome+' by equal classification cutoff approach')


# Precision-recall cut-off probability approach
from sklearn.metrics import confusion_matrix
# daily model measure confidence interval summary        
DailyResults=[]
DailyAccuracy=[]
for now_date in daterange(start_date, end_date):
    TestData_day_t=TestData[TestData['Index date']==now_date.strftime("%Y-%m-%d")]
    ExpectedOutcomeNumber=TestData_day_t['Predicted probability (Imbalanced Data Model)'].sum()
    ActualOutcomeNumber=TestData_day_t[Outcome].sum()
    real=TestData_day_t[Outcome].values.tolist()
    predicted=TestData_day_t['Predicted with F1-score-optimized cutoff'].values.tolist()
    # Calculate overall model evaluation measures
    if sum(real)>1:
        CM=confusion_matrix(real, predicted)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # F1 score
        F1=2*PPV*TPR/(PPV+TPR)
        DailyResults.append([now_date, roc_auc_score(real, predicted), average_precision_score(real, predicted), TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, F1,TestData_day_t.shape[0], ActualOutcomeNumber, ExpectedOutcomeNumber])
    else:
        DailyResults.append([now_date, '', '', '', '', '', '', '', '', '', '', '',TestData_day_t.shape[0], ActualOutcomeNumber, ExpectedOutcomeNumber])   
    # Calculate daily accuracy
    if TestData_day_t.shape[0]>0:
        if ActualOutcomeNumber>0 and (np.round(ExpectedOutcomeNumber, 0)<2*ActualOutcomeNumber):
            dayaccuracy=np.round((1-np.abs(np.round(ExpectedOutcomeNumber, 0)-ActualOutcomeNumber)/ActualOutcomeNumber)*100,2)
            DailyAccuracy.append([now_date, dayaccuracy])
        elif ActualOutcomeNumber==np.round(ExpectedOutcomeNumber, 0) and np.round(ExpectedOutcomeNumber, 0)==0:
            DailyAccuracy.append([now_date, 100])
        else:
            DailyAccuracy.append([now_date, 0.00])
    else:
        if ActualOutcomeNumber==np.round(ExpectedOutcomeNumber, 0) and np.round(ExpectedOutcomeNumber, 0)==0:
            DailyAccuracy.append([now_date, 100])
        else:
            DailyAccuracy.append([now_date, np.nan])

DailyResults=pd.DataFrame(DailyResults)
DailyResults.columns=['Index date', 'ROC AUC score', 'AUPRC score', 'Sensitivity, hit rate, recall, or true positive rate',
                      'Specificity or true negative rate','Precision or positive predictive value',
                      'Negative predictive value','Fall out or false positive rate',
                     'False negative rate','False discovery rate','Overall accuracy','F1 score','Number of patients', 'Actual number ('+Outcome+')', 'Expected number ('+Outcome+')']
DailyAccuracy=pd.DataFrame(DailyAccuracy)   
DailyAccuracy.columns=['Index date', 'Daily prediction accuracy (%)']
DailyResults['Actual daily probability ('+Outcome+')']=DailyResults['Actual number ('+Outcome+')']/DailyResults['Number of patients']
DailyResults['Predicted daily probability ('+Outcome+')']=DailyResults['Expected number ('+Outcome+')']/DailyResults['Number of patients']
DailyResults['Actual daily probability ('+Outcome+')']=DailyResults['Actual daily probability ('+Outcome+')'].replace(np.nan, 0)
DailyResults['Predicted daily probability ('+Outcome+')']=DailyResults['Predicted daily probability ('+Outcome+')'].replace(np.nan, 0)
DailyResults['Daily residual ('+Outcome+')']=DailyResults['Expected number ('+Outcome+')']-DailyResults['Actual number ('+Outcome+')']
DailyResults['Daily prediction accuracy (%)']=DailyAccuracy['Daily prediction accuracy (%)']
DailyResults.to_csv(Path+"Aggregated Daily Results Emergency.csv", index=False)

########################### Results visualization
# 1. DailyMetrics histograms with best daily probability cutoff
fig = plt.figure(figsize=(14,7), dpi=300)
DailyResults['Daily prediction accuracy (%)']=DailyResults['Daily prediction accuracy (%)']
ax=DailyResults[['ROC AUC score', 'AUPRC score', 'Sensitivity, hit rate, recall, or true positive rate',
                      'Specificity or true negative rate','Precision or positive predictive value',
                      'Negative predictive value', 'Overall accuracy','F1 score']].apply(pd.to_numeric, errors='coerce').hist(figsize=(15,9),bins=100, grid=True,color='green', zorder=2, rwidth=0.9, ec="#eeeeee", alpha=0.9)
plt.tight_layout()
plt.show()
fig = ax[0][0].get_figure()
fig.savefig(Path+'DailyMetrics histograms by F1 score approach '+' for cutoff '+filesavename+'.svg', bbox_inches='tight', transparent=False, dpi=300)
fig.savefig(Path+'DailyMetrics histograms by F1 score approach '+' for cutoff '+filesavename+'.pdf', bbox_inches='tight', transparent=False, dpi=300)

# 2. Daily predicion comparision with best daily probability cutoff
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager

# 2. Daily predicion comparision with best daily probability cutoff
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager

# Daily predicion comparision with best daily probability cutoff
display(DailyResults)
plotData=DailyResults[['Number of patients',
       'Expected number (Imminent discharge within 24 hours)',
       'Actual number (Imminent discharge within 24 hours)']]
plotData.columns=['Emergency patients in hospital',
       'Model predicted 24-hour discharges',
       'Actual 24-hour discharges']
index = pd.date_range(start = "2019-02-02", end = "2020-01-31", freq = "D")
index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in index]
plotData.index=index
fig, ax = plt.subplots(figsize=(20, 10),dpi=300)
plotData.plot(label='plotData', color=['red', 'blue','orange'], lw=1, style="-o", ms=4, grid=False, fontsize=14, ax=ax)
ax.set_xticks(plotData.index)
# ax.set_title("Predicting 24-hour discharge for emergency patients \n (Aggregating patients being in hospital for different times)", fontweight="bold", fontsize=16)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontweight('bold') for label in labels]
plt.gcf().autofmt_xdate()
ax.figure.autofmt_xdate(rotation=45, ha='center')
font = font_manager.FontProperties(weight='bold',
                                   style='normal', size=24)
plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.18),prop=font)
plt.xlabel("1 Feb 2019 - 31 Jan 2020", fontsize=24,fontweight='bold')
plt.ylabel("Number of patients", fontsize=26,fontweight='bold')
plt.show()
fig.savefig(Path+'Daily predicion comparision with by F1 score cutoff approach '+filesavename+'.pdf', dpi=300, bbox_inches='tight',transparent=True)
fig.savefig(Path+'Daily predicion comparision with by F1 score cutoff approach '+filesavename+'.svg', dpi=300, bbox_inches='tight',transparent=True)

# 3. Daily calibration validation with best cutoff
plotData=DailyResults
plotData['Ordered day in 2019/02/01-2020/01/31']=DailyResults['Actual daily probability ('+Outcome+')'].rank(method='dense', ascending=True).tolist()  
plotData.sort_values(by=['Ordered day in 2019/02/01-2020/01/31'], inplace=True, ascending=True)
plotData['Ordered day in 2019/02/01-2020/01/31']=np.arange(plotData.shape[0])
plotData.to_csv(Path+'Combined Daily prediction summary '+' '+Outcome+' by F1 score cutoff approach '+filesavename+'.csv', index=False)
# plot prediction gap histograms
plot=plotData.hvplot.scatter(legend='top_left',width=800, height=400,
                    stacked=True,cmap='Category20', title='XGBoost model to predict '+Outcome+' for emergency patients being in hospital on each index date',
                    x="Ordered day in 2019/02/01-2020/01/31", 
                    y=['Actual number ('+Outcome+')', 
                       'Expected number ('+Outcome+')'])
hv.extension('bokeh')
# going to use show() to open plot in browser
from bokeh.plotting import show
show(hv.render(plot))
renderer = hv.renderer('bokeh')
renderer.save(plot, 'Daily prediction summary '+filesavename+' '+Outcome+' by F1 score cutoff approach')
# 4. Weekday model performance by daily calibration validation
DailyResults['Index date']=pd.to_datetime(DailyResults['Index date'], dayfirst=True).dt.tz_localize(None)
DailyResults['Weekday of index date']=DailyResults['Index date'].dt.day_name().tolist()
DailyResults['Daily prediction accuracy']=DailyResults['Daily prediction accuracy (%)']/100
plotData=DailyResults[['Weekday of index date','ROC AUC score','AUPRC score',
       'Specificity or true negative rate',
       'Precision or positive predictive value', 'Negative predictive value',
       'Overall accuracy', 'F1 score',
       'Daily prediction accuracy']]
plotData.columns=['Weekday of index date','ROC AUC score','AUPRC score',
       'Specificity or true negative rate',
       'Precision or positive predictive value', 'Negative predictive value',
       'Overall accuracy', 'F1 score',
       'Daily prediction accuracy']
plotDataMelt=pd.melt(plotData,id_vars=['Weekday of index date'],value_vars=['ROC AUC score','AUPRC score',
       'Specificity or true negative rate',
       'Precision or positive predictive value', 'Negative predictive value',
       'Overall accuracy', 'F1 score',
       'Daily prediction accuracy'],var_name='Measure')
plotDataMelt['Weekday']=plotDataMelt['Weekday of index date']
plotDataMelt=plotDataMelt.replace({'Weekday' : { 'Monday' : 1, 'Tuesday' : 2, 
                                   'Wednesday' : 3, 'Thursday': 4, 
                                  'Friday': 5, 'Saturday': 6,
                                  'Sunday': 7}})
plotDataMelt.sort_values(by=['Weekday'], inplace=True, ascending=True)
# plot boxplots
plt.figure(figsize=(14,7), dpi=200)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
bplot = sns.boxplot(showfliers=False, y='value', x='Weekday of index date', 
                 data=plotDataMelt, 
                 palette="colorblind", 
                  hue='Measure')
# make grouped stripplot and save it in a variable
bplot = sns.stripplot(y='value', x='Weekday of index date', 
                   data=plotDataMelt, 
                   jitter=True,
                   dodge=True, 
                   marker='.', 
                   alpha=0.4,
                   hue='Measure',
                   color='grey')
bplot.axes.set_title("Model performance to predict 24-hour discharge for emergency patients being in hospital on each index date \n (F1-score optimized cut-off)", 
                     weight='bold', fontsize=16)
bplot.set_xlabel("Weekday of index date", fontsize=14,weight='bold')
bplot.set_ylabel("Measure value", fontsize=14,weight='bold')
bplot.tick_params(labelsize=10)
# get legend information from the plot object
handles, labels = bplot.get_legend_handles_labels()
l = plt.legend(handles[0:7], labels[0:7])
bplot.set_ylim(0.5,1.02)
# save as jpeg
bplot.figure.savefig(Path+"Weekday Performance "+filesavename+".svg", format='svg', dpi=300)








