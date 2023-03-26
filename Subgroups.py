from numpy import argmax
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)      
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error as mae
from matplotlib import pyplot as plt
import os
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

SelectedVariables=['Index date', 'Subcohort', 'filename', 'ClusterID', 'AdmissionDate',
       'SpellID', 'DischargeDate', 'EpisodeID', 'EpisodeStartDate',
       'EpisodeEndDate', 'LinkedBirthMonth', 'LinkedSex',
       'ConsultantMainSpecialtyCode', 'EthnicGroupCode', 'PostcodeStub',
       'LinkedDeathdate', 'SpineCheckDate', 'ConsultantCodeAnon',
       'AdmissionSourceCode', 'AdmissionMethodCode',
       'DischargeDestinationCode', 'DischargeMethodCode',
       'Latest admission to decision day, days', 'Adm2index, hours',
       'ToStayHourDuration', 
       'Imminent discharge within 48 hours', '7-days longer stay',
       '7-days stay or less', '7-14 days stay', '14-30 days stay',
       '30 days stay or more', 'Future hospital stay after decision day, days',
       'ToMortalityHourDuration', '30-day mortality','Age', 'Sex', 'BMI',
       'Height, cm', 'Weight, kg', 'Ethnic: White (Yes/No)',
       'Ethnic: Mixed (Yes/No)', 'Ethnic: Asian or Asian British (Yes/No)',
       'Ethnic: Black or Black British (Yes/No)',
       'Ethnic: Other groups (Yes/No)',
       'Ethnic: Not stated/Not known (Yes/No)', 'IMD score',
       'Swindon SN (Yes/No)', 'Chinnor (Yes/No)', 'Abingdon (Yes/No)',
       'Bampton, Burford, Carterton (Yes/No)', 'Didcot (Yes/No)',
       'Milton Keynes (Yes/No)', 'Oxford (Yes/No)', 'Witney (Yes/No)',
       'Other postcode region  (Yes/No)', 'Bicester (Yes/No)',
       'Wantage (Yes/No)', 'Slough (Yes/No)', 'Banbury (Yes/No)',
       'Wallingford (Yes/No)', 'Woodstock (Yes/No)', 'Watlington (Yes/No)',
       'Chipping Norton (Yes/No)', 'Reading (Yes/No)', 'Northampton (Yes/No)',
       'Kidlington (Yes/No)', 'Hemel Hempstead (Yes/No)', 'Thame (Yes/No)',
       'Charlson comorbidity index', 'Elixhauser comorbidity index',
       'Admission source: Usual place of residence',
       'Admission source: Non-NHS institutional care',
       'Admission source: Other NHS Provider', 
       'Current admission includes admission under surgical subspecialty at or before index date (Yes/No)',
       'Current admission includes admission under other specialty at or before index date (Yes/No)',
       'Current admission includes admission under medical subspecialty at or before index date (Yes/No)',
       'Current admission includes admission under acute and general surgery at or before index date (Yes/No)',
       'Current admission includes admission under acute and general medicine at or before index date (Yes/No)',
       'Actual LOS, day', 'Mean LOS, hour', 'Median LOS, hour', 'LOS SD', 'Index day of year',
       'Weekday of index date', 'Index date public holiday (Yes/No)','Imminent discharge within 24 hours',
       'Predicted (Imbalanced Data Model)',
       'Predicted probability (Imbalanced Data Model)',
       'Predicted (Balanced Data Model)',
       'Predicted probability (Balanced Data Model)']

Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
files=sorted(os.listdir(Path), key=str.lower)
if '.DS_Store' in files:
    files.remove('.DS_Store')   
AggregatedEmergencyPredictedFiles=[]
for file in [x for x in files if 'Predicted Emergency' in x and 'testing (Imminent discharge within 24 hours)' in x]:
    Data=pd.read_csv(Path+file)
    Data['AdmissionDate']=pd.to_datetime(Data['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
    Data['DischargeDate']=pd.to_datetime(Data['DischargeDate'], dayfirst=True).dt.tz_localize(None)
    Data['Actual LOS, day']=[(Data['DischargeDate'].iloc[x]-Data['AdmissionDate'].iloc[x]).total_seconds()/3600/24 for x in range(Data.shape[0])]
    Data=Data[SelectedVariables]
    print(file, Data.shape)    
    AggregatedEmergencyPredictedFiles.append(Data)
SaveAggregatedEmergencyPredictedFiles=pd.concat(AggregatedEmergencyPredictedFiles, axis=0)
print(SaveAggregatedEmergencyPredictedFiles.drop_duplicates().shape)
print(SaveAggregatedEmergencyPredictedFiles.head(100))
SaveAggregatedEmergencyPredictedFiles.to_csv(Path+"AggregatedEmergencyPredicted.csv", index=False)

Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
files=sorted(os.listdir(Path), key=str.lower)
if '.DS_Store' in files:
    files.remove('.DS_Store')   
AggregatedPlanedPredictedFiles=[]
for file in [x for x in files if 'Predicted Planed' in x and 'testing (Imminent discharge within 24 hours)' in x]:
    Data=pd.read_csv(Path+file)
    Data['AdmissionDate']=pd.to_datetime(Data['AdmissionDate'], dayfirst=True).dt.tz_localize(None)
    Data['DischargeDate']=pd.to_datetime(Data['DischargeDate'], dayfirst=True).dt.tz_localize(None)
    Data['Actual LOS, day']=[(Data['DischargeDate'].iloc[x]-Data['AdmissionDate'].iloc[x]).total_seconds()/3600/24 for x in range(Data.shape[0])]
    Data=Data[SelectedVariables]
    print(file, Data.shape)    
    AggregatedPlanedPredictedFiles.append(Data)
SaveAggregatedPlanedPredictedFiles=pd.concat(AggregatedPlanedPredictedFiles, axis=0)
print(SaveAggregatedPlanedPredictedFiles.drop_duplicates().shape)
print(SaveAggregatedPlanedPredictedFiles.head(100))
SaveAggregatedPlanedPredictedFiles.to_csv(Path+"AggregatedPlannedPredicted.csv", index=False)


from sklearn.metrics import roc_curve, auc
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']) 

### Patient-level: Calculate AUROC, AUPRC, PPV, NPV, F1 score, etc
TestData=SaveAggregatedEmergencyPredictedFiles
#TestData=TestData[TestData['Adm2index, hours']>28]
#TestData=TestData[(TestData['Adm2index, hours']<=13*24) & (TestData['Adm2index, hours']<=14*24)]
Outcome='Imminent discharge within 24 hours'
# J-statistic cutoff optimizing approach
if TestData[Outcome].sum()>0:
    threshold = Find_Optimal_Cutoff(TestData[Outcome], TestData['Predicted probability (Imbalanced Data Model)'])
else:
    threshold=[1]   
TestData['Predicted with YoudenJ-optimized cutoff']=TestData['Predicted probability (Imbalanced Data Model)'].map(lambda x: 1 if x > threshold[0] else 0)        
# F1 score cutoff optimizing approach
if TestData[Outcome].sum()>2:
    precision, recall, thresholds = precision_recall_curve(TestData[Outcome], TestData['Predicted probability (Imbalanced Data Model)'])
    fscore = (2 * precision * recall) / (precision + recall)
    threshold = thresholds[argmax(fscore)]
else:
    threshold=1
TestData['Predicted with F1-score-optimized cutoff']=TestData['Predicted probability (Imbalanced Data Model)'].map(lambda x: 1 if x >threshold  else 0)
    
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
real=TestData[Outcome].values.tolist()
predictedProb=TestData['Predicted probability (Imbalanced Data Model)'].values.tolist()
cutoffMethod='Predicted with F1-score-optimized cutoff'
predicted=TestData[cutoffMethod].values.tolist()
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
Evaluations=pd.DataFrame()
Evaluations['Measure']=[cutoffMethod, 'AUROC', 'AUPRC','PPV', 'NPV', 'Sensitivity', 'Specificity', 'F1']
Evaluations['Value']=['', roc_auc_score(real, predicted), average_precision_score(real, predictedProb),PPV, NPV, TPR, TNR, F1]
print(Evaluations)


### Hospital-level: Calculate MAE% after aggregating probabilities per index date
# subgroup analysis
Database=TestData
# sex
Database['Male gender (Yes/No)']=np.where(Database['LinkedSex']=='M', 1, 0)
Database['Female gender (Yes/No)']=np.where(Database['LinkedSex']=='F', 1, 0)
# age
Database['16-40'] = np.where( (Database['Age']>16) & (Database['Age']<=40), 1, 0)
Database['40-60'] = np.where( (Database['Age']>40) & (Database['Age']<=60), 1, 0)
Database['60-80'] = np.where( (Database['Age']>60) & (Database['Age']<=80), 1, 0)
Database['>80'] = np.where(Database['Age']>80, 1, 0)
# IMDScore
Database['IMD<8.5'] = np.where(Database['IMD score']<8.5, 1, 0)
Database['IMD 8.5-13.8'] = np.where( (Database['IMD score']>=8.5) & (Database['IMD score']<13.8), 1, 0)
Database['IMD 13.8-21.36'] = np.where( (Database['IMD score']>=13.8) & (Database['IMD score']<21.36), 1, 0)
Database['IMD 21.36-34.18'] = np.where( (Database['IMD score']>=21.36) & (Database['IMD score']<=34.18), 1, 0)
Database['IMD>34.18'] = np.where(Database['IMD score']>34.18, 1, 0)
# comorbidity index
Database['Charlson=0'] = np.where(Database['Charlson comorbidity index']==0, 1, 0)
Database['Charlson mild 1-2'] = np.where( (Database['Charlson comorbidity index']>=1) & (Database['Charlson comorbidity index']<2), 1, 0)
Database['Charlson moderate 3-4'] = np.where( (Database['Charlson comorbidity index']>=3) & (Database['Charlson comorbidity index']<=4), 1, 0)
Database['Charlson severe 5+'] = np.where(Database['Charlson comorbidity index']>5, 1, 0)
# weekday
Database['Weekday of admission date']=Database['AdmissionDate'].dt.dayofweek
Database['Admission_Week_Day']=Database['AdmissionDate'].dt.day_name().tolist()
Database['Monday admission'] = np.where(Database['Admission_Week_Day']=='Monday', 1, 0)
Database['Tuesday admission'] = np.where(Database['Admission_Week_Day']=='Tuesday', 1, 0)
Database['Wednesday admission'] = np.where(Database['Admission_Week_Day']=='Wednesday', 1, 0)
Database['Thursday admission'] = np.where(Database['Admission_Week_Day']=='Thursday', 1, 0)
Database['Friday admission'] = np.where(Database['Admission_Week_Day']=='Friday', 1, 0)
Database['Saturday admission'] = np.where(Database['Admission_Week_Day']=='Saturday', 1, 0)
Database['Sunday admission'] = np.where(Database['Admission_Week_Day']=='Sunday', 1, 0)
# LOS
Database['Actual LOS less than 7 days'] = np.where(Database['Actual LOS, day']<=7, 1, 0)
Database['Mean diagnosis LOS less than 7 days'] = np.where(Database['Mean LOS, hour']<=7*24, 1, 0)

subgroups=['Male gender (Yes/No)','Female gender (Yes/No)', 
'16-40', '40-60', '60-80', '>80',
 'Ethnic: White (Yes/No)',
       'Ethnic: Mixed (Yes/No)', 'Ethnic: Asian or Asian British (Yes/No)',
       'Ethnic: Black or Black British (Yes/No)',
       'Ethnic: Other groups (Yes/No)',
       'Ethnic: Not stated/Not known (Yes/No)', 
 'IMD<8.5', 'IMD 8.5-13.8','IMD 13.8-21.36', 'IMD 21.36-34.18', 'IMD>34.18',
           'Charlson=0','Charlson mild 1-2','Charlson moderate 3-4','Charlson severe 5+',
 'Monday admission','Tuesday admission', 'Wednesday admission', 'Thursday admission','Friday admission', 'Saturday admission', 'Sunday admission', 
 'Admission source: Usual place of residence',
'Admission source: Non-NHS institutional care',
'Admission source: Other NHS Provider', 
 'Current admission includes admission under surgical subspecialty at or before index date (Yes/No)','Current admission includes admission under other specialty at or before index date (Yes/No)',
 'Current admission includes admission under medical subspecialty at or before index date (Yes/No)',
 'Current admission includes admission under acute and general surgery at or before index date (Yes/No)',
 'Current admission includes admission under acute and general medicine at or before index date (Yes/No)']
columnVariables=subgroups+['Index date', 'Actual LOS less than 7 days','Mean diagnosis LOS less than 7 days',
       'Imminent discharge within 24 hours',
       'Predicted (Imbalanced Data Model)',
       'Predicted probability (Imbalanced Data Model)',
       'Predicted (Balanced Data Model)',
       'Predicted probability (Balanced Data Model)',
       'Predicted with YoudenJ-optimized cutoff',
       'Predicted with F1-score-optimized cutoff']

TestDataSubgroup=Database[columnVariables]
TestDataSubgroup=TestDataSubgroup.set_index("Index date")

# Overall analysis
TestDataHospitalLevel=TestDataSubgroup.groupby('Index date').sum().reset_index()
print(TestDataHospitalLevel)
actualNum=TestDataHospitalLevel['Predicted (Imbalanced Data Model)'].values.tolist()
predNum=TestDataHospitalLevel['Predicted with F1-score-optimized cutoff'].values.tolist()
print('MAE='+str(mae(actualNum, predNum)),'MAE%='+str(np.round(mae(actualNum, predNum)/np.mean(actualNum)*100,2))+'%')

# Subgroup analysis
SubgroupMAE=[]
for group in subgroups:
    TestDataHospitalLevel_Group=TestDataSubgroup[TestDataSubgroup[group]==1].groupby('Index date').sum().reset_index()
    actualNum=TestDataHospitalLevel_Group['Predicted (Imbalanced Data Model)'].values.tolist()
    predNum=TestDataHospitalLevel_Group['Predicted with F1-score-optimized cutoff'].values.tolist()
    SubgroupMAE.append([group,TestDataSubgroup[group].sum(), mae(actualNum, predNum),np.round(mae(actualNum, predNum)/np.mean(actualNum)*100,2)])
    
SubgroupMAE=pd.DataFrame(SubgroupMAE)
SubgroupMAE.columns=['Subgroup', 'Number of day-observations', 'MAE', 'MAE%']
print(SubgroupMAE)


# Subgroup analysis restricting actual LOS no more than 7 days
SubgroupMAE=[]
for group in subgroups:
    TestDataHospitalLevel_Group=TestDataSubgroup[(TestDataSubgroup[group]==1) & (TestDataSubgroup['Actual LOS less than 7 days']==1)].groupby('Index date').sum().reset_index()
    actualNum=TestDataHospitalLevel_Group['Predicted (Imbalanced Data Model)'].values.tolist()
    predNum=TestDataHospitalLevel_Group['Predicted with F1-score-optimized cutoff'].values.tolist()
    SubgroupMAE.append([group,TestDataSubgroup[TestDataSubgroup['Actual LOS less than 7 days']==1][group].sum(), mae(actualNum, predNum),np.round(mae(actualNum, predNum)/np.mean(actualNum)*100,2)])
    
SubgroupMAE=pd.DataFrame(SubgroupMAE)
SubgroupMAE.columns=['Subgroup', 'Number of day-observations', 'MAE', 'MAE%']
print(SubgroupMAE)



Path="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/24hourDischargeSubgroupAnalysis/DataSummary/"
files=sorted(os.listdir(Path), key=str.lower)
if '.DS_Store' in files:
    files.remove('.DS_Store')
if 'StatsResults.csv' in files:
    files.remove('StatsResults.csv')    
Results=[]
for file in files:
    print(file)
    Data=pd.read_csv(Path+file)
    Data=pd.read_csv(Path+file)
    lis=Data.iloc[0:Data.shape[0]-1,1].values.tolist()
    lis.extend([file])
    Results.append(lis)
Results=pd.DataFrame(Results)
Results.columns=['Test PPV','Test NPV', 'Test sensitivity',
 'Test specificity', 'Test precision', 'Test recall',
 'Test F1 score', 'Average daily accuracy (%)', 'Patient group']
Results.to_csv(Path+'StatsResults.csv', index=False)

Results['Group']=[x.split('by')[1].split('(')[0].split('.csv')[0] for x in Results['Patient group'].values.tolist()]
Results['Being in hospital']=[x.split('by')[0].split(' ')[1] for x in Results['Patient group'].values.tolist()]
Results['Group'].unique()   


def generatePlot(i):
    cohort="Emergency"+groups[i]
    BeingIn=cols[i]
    ResultsShow=Results[Results['Being in hospital']==cohort]
    ResultsShow=ResultsShow.replace(0,np.nan).dropna()
    del ResultsShow['Patient group']
    del ResultsShow['Being in hospital']
    ResultsShow.index=ResultsShow['Group']
    del ResultsShow['Group']
    ResultsShow.dtypes
    for x in ResultsShow.columns.tolist():
        ResultsShow[x] = ResultsShow[x].astype(float)
    print(ResultsShow)

    SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/24hourDischargeSubgroupAnalysis/DataSummaryPlot/"
    # set heatmap size
    plt.figure(figsize= (20,10), dpi=500) 
    # Standardize or Normalize every column in the figure
    # Standardize:
    # set heatmap size

    cm=sns.clustermap(ResultsShow, center=0, annot=True, fmt='.3f', annot_kws={'fontsize':8,'fontstyle':'italic', 'verticalalignment':'center'}, 
                      col_cluster=False, cmap='Blues', 
                      figsize= (15,10), cbar_pos=(1.05, .4, .03, .3), 
                      linewidths=0.004, yticklabels=True,xticklabels=True)
    cm.ax_heatmap.yaxis.set_ticks_position("right")
    ax = cm.ax_heatmap
    ax.tick_params(labelbottom=True,labeltop=False)
    #ax.set_xlabel("")
    ax.set_ylabel("Predicting 24-hour discharge in emergency patients \n being "+BeingIn)
    plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    #plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
    #plt.title("Predicting imminent discharge within 24 hours in planned patients", fontsize = 12)
    #plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.set_size_inches(15, fig.get_figheight(), forward=True)
    fig.savefig(SavePath+cohort+'_model_performance_by_subgroups.svg', format='svg', dpi=500)
    fig.savefig(SavePath+cohort+'_model_performance_by_subgroups.pdf', format='pdf', dpi=500)

groups=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7D10', 'D11D13', 'D14D20', 'D21D27', 'D28+']
cols=['in hospital for 0-24 hours', 'in hospital for 24-48 hours', 
      'in hospital for 48-72 hours', 'in hospital for 72-96 hours', 
      'in hospital for 96-120 hours', 'in hospital for 120-144 hours', 
      'in hospital for 144-264 hours', 'in hospital for 264-336 hours', 
      'in hospital for 336-504 hours', 'in hospital for 504-672 hours',
     'in hospital >672 hours']

def generatePlot(i):
    cohort="Emergency"+groups[i]
    BeingIn=cols[i]
    ResultsShow=Results[Results['Being in hospital']==cohort]
    ResultsShow=ResultsShow.replace(0,np.nan).dropna()
    print(set(ResultsShow['Group']))
    del ResultsShow['Patient group']
    del ResultsShow['Being in hospital']
    ResultsShow=ResultsShow[ResultsShow['Group'].isin([x for x in ResultsShow['Group'].tolist() if 'Age' in x])]
    ResultsShow.index=ResultsShow['Group']
    del ResultsShow['Group']
    ResultsShow.dtypes
    for x in ResultsShow.columns.tolist():
        ResultsShow[x] = ResultsShow[x].astype(float)
    print(ResultsShow)

    SavePath="../Patients flow prediction/Codes/Imbalance XGBoost/Visualization/24hourDischargeSubgroupAnalysis/DataSummaryPlot/"
    # set heatmap size
    # Standardize or Normalize every column in the figure
    # Standardize:
    # set heatmap size
    
    cm=sns.clustermap(ResultsShow, center=0, annot=True, fmt='.3f', annot_kws={'fontsize':8,'fontstyle':'italic', 'verticalalignment':'center'}, 
                      col_cluster=False, cmap='Blues', 
                      figsize= (15,3), cbar_pos=(1.05, .4, .03, .3), 
                      linewidths=0.004, yticklabels=True,xticklabels=True)
    cm.ax_heatmap.yaxis.set_ticks_position("right")
    cm.ax_heatmap.tick_params(labelbottom=True,labeltop=False)
    cm.ax_heatmap.set_xlabel("Predicting 24-hour discharge in emergency patients being "+BeingIn, fontsize=7)
    cm.ax_heatmap.set_ylabel("")
    plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    #plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
    #plt.title("Predicting imminent discharge within 24 hours in planned patients", fontsize = 12)
    #plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
    plt.tight_layout()
    fig = cm.ax_heatmap.get_figure()
    fig.set_size_inches(15, fig.get_figheight(), forward=True)
    fig.savefig(SavePath+cohort+'_model_performance_by_subgroups.svg', format='svg', dpi=500)
    fig.savefig(SavePath+cohort+'_model_performance_by_subgroups.pdf', format='pdf', dpi=500)
    

groups=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7D10', 'D11D13', 'D14D20', 'D21D27', 'D28+']
cols=['in hospital for 0-24 hours', 'in hospital for 24-48 hours', 
      'in hospital for 48-72 hours', 'in hospital for 72-96 hours', 
      'in hospital for 96-120 hours', 'in hospital for 120-144 hours', 
      'in hospital for 144-264 hours', 'in hospital for 264-336 hours', 
      'in hospital for 336-504 hours', 'in hospital for 504-672 hours',
     'in hospital >672 hours']
    

generatePlot(0)
generatePlot(1)
generatePlot(2)
generatePlot(3)
generatePlot(4)
generatePlot(5)
generatePlot(6)
generatePlot(7)
generatePlot(9)
generatePlot(10)

