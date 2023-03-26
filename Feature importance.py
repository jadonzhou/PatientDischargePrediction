import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, timedelta
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
              
NewData=pd.DataFrame()
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
files=sorted(os.listdir(Path), key=str.lower)
if '.DS_Store' in files:
    files.remove('.DS_Store')
fileSet=[x for x in files if 'DatabaseEmergency' in x and 'feature importance Balanced Data' in x]
features=[]
for file in fileSet:
    Data=pd.read_csv(Path+file)
    #Data=Data.iloc[0:400,:]
    features.extend(Data['Features'].values.tolist())
features=np.unique(features)
NewData['Features']=features
# generate feature rankings for visualization
for file in fileSet:
    Data=pd.read_csv(Path+file)
    Data['Rank']=[603-x for x in range(Data.shape[0])]
    Data['Rank']=np.where(Data['Importance']==0, 0, Data['Rank'])
    Data['Rank']=Data['Rank']/(603/100)
    print(Data.shape)
    fileRank=[]
    for var in NewData['Features'].values.tolist():
        varTemp=Data['Rank'][Data['Features']==var]
        if varTemp.shape[0]:
            fileRank.append(varTemp.tolist()[0])
        else:
            fileRank.append(Data['Rank'].min())
    NewData[file.split(' ')[0]]=fileRank
NewData = NewData.sort_values(by=NewData.columns[1], ascending=False)
NewData.to_csv(Path+"24-hour discharge emergency feature importance statistics.csv", index=False)


NewData=pd.DataFrame()
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
files=sorted(os.listdir(Path), key=str.lower)
if '.DS_Store' in files:
    files.remove('.DS_Store')
fileSet=[x for x in files if 'DatabasePlaned' in x and 'feature importance Balanced Data' in x]
features=[]
for file in fileSet:
    Data=pd.read_csv(Path+file)
    #Data=Data.iloc[0:400,:]
    features.extend(Data['Features'].values.tolist())
features=np.unique(features)
NewData['Features']=features
# generate feature rankings for visualization
for file in fileSet:
    Data=pd.read_csv(Path+file)
    Data['Rank']=[400-x for x in range(Data.shape[0])]
    Data['Rank']=np.where(Data['Importance']==0, 0, Data['Rank'])
    Data['Rank']=Data['Rank']/4
    print(Data.shape)
    fileRank=[]
    for var in NewData['Features'].values.tolist():
        varTemp=Data['Rank'][Data['Features']==var]
        if varTemp.shape[0]:
            fileRank.append(varTemp.tolist()[0])
        else:
            fileRank.append(Data['Rank'].min())
    NewData[file.split(' ')[0]]=fileRank
NewData = NewData.sort_values(by=NewData.columns[1], ascending=False)
NewData.to_csv(Path+"24-hour discharge planed feature importance statistics.csv", index=False)


# import libraries
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
import pandas as pd # for data analysis
# load dataset and create heatmap for individual feature importance for both emergency and planned
emergencyData = pd.read_csv(Path+"24-hour discharge emergency feature importance statistics.csv")
emergencyData = emergencyData.set_index('Features')
emergencyData=emergencyData.replace(0,np.nan)
emergencyData=emergencyData[emergencyData!=0].dropna()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(80, 50), dpi=200)
plt.figure(figsize= (8,40)) 
cbar_kws = {"orientation":"vertical", 
            "shrink":0.5,
            "pad": 0.01,
            'extend':'min', 
            'extendfrac':0.1, 
            "ticks":np.arange(0,101), 
            "drawedges":True,
           } # color bar keyword arguments
sns.heatmap(emergencyData, vmin = 0, vmax = 100, cmap="Blues", annot = True, annot_kws={"fontsize":8}, cbar = True, linewidth = 2, cbar_kws=cbar_kws, ax=ax1)
ax1.set_ylabel('')
ax1.set_title("Predicting imminent discharge within 24 hours in emergency patients", fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='left')
ax1.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
# load dataset and create heatmap
planedData = pd.read_csv(Path+"24-hour discharge planed feature importance statistics.csv")
planedData = planedData.set_index('Features')
planedData=planedData.replace(0,np.nan)
planedData=planedData[planedData!=0].dropna()
plt.figure(figsize= (8,40)) 
cbar_kws = {"orientation":"vertical", 
            "shrink":0.5,
            "pad": 0.01,
            'extend':'min', 
            'extendfrac':0.1, 
            "ticks":np.arange(0,101), 
            "drawedges":True,
           } # color bar keyword arguments
sns.heatmap(planedData, vmin = 0, vmax = 100, cmap="OrRd", annot = True, annot_kws={"fontsize":8}, cbar = True, linewidth = 2, cbar_kws=cbar_kws, ax=ax2)
ax2.set_ylabel('')
ax2.set_title("Predicting imminent discharge within 24 hours in planed patients", fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='left')
ax2.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
# save
fig.savefig(Path+'Feature importance.svg', format='svg', dpi=300)
fig.savefig(Path+'Feature importance.pdf', format='pdf', dpi=300)




# import libraries
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
import pandas as pd # for data analysis
# load dataset and create heatmap for individual feature importance for planned
Data = pd.read_csv(Path+"24-hour discharge planed feature importance statistics.csv")
Data = Data.set_index('Features')
Data=Data[['DatabasePlanedD1', 'DatabasePlanedD2', 'DatabasePlanedD3', 'DatabasePlanedD4', 'DatabasePlanedD5',
       'DatabasePlanedD6', 'DatabasePlanedD7D10','DatabasePlanedD11D13', 
           'DatabasePlanedD14D20','DatabasePlanedD21D27', 'DatabasePlanedD28+']]
Data=Data.replace(0,np.nan)
Data=Data[Data!=0].dropna()
#print(globalWarming_df)
# set heatmap size
plt.figure(figsize= (8,50), dpi=400) 
# create heatmap seaborn
cbar_kws = {"orientation":"vertical", 
            "shrink":1,
            'extend':'min', 
            'extendfrac':0.1, 
            "ticks":np.arange(0,101), 
            "drawedges":True,
           } # color bar keyword arguments
chart=sns.heatmap(Data, vmin = 0, vmax = 100, cmap="Blues", fmt='.0f', annot = True, annot_kws={"fontsize":8}, cbar = True, linewidth = 2, cbar_kws=cbar_kws)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='left')
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.title("Predicting imminent discharge within 24 hours in planned patients", fontsize = 12)
plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
plt.ylabel("Features", fontsize = 12, va="center")
plt.show()
fig = chart.get_figure()
fig.set_size_inches(75, fig.get_figheight(), forward=True)
fig.savefig(Path+'Feature importance for planned patients.svg', format='svg', dpi=300)
fig.savefig(Path+'Feature importance for planned patients.pdf', format='pdf', dpi=300)


import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
# load dataset and create DataFrame ready to create heatmap for emergency
Data = pd.read_csv(Path+"24-hour discharge emergency feature importance statistics.csv")
Data = Data.set_index('Features')
Data=Data[['DatabaseEmergencyD1', 'DatabaseEmergencyD2',
       'DatabaseEmergencyD3', 'DatabaseEmergencyD4', 'DatabaseEmergencyD5',
       'DatabaseEmergencyD6', 'DatabaseEmergencyD7D10','DatabaseEmergencyD11D13',
       'DatabaseEmergencyD14D20', 'DatabaseEmergencyD21D27', 'DatabaseEmergencyD28+']]
mask_na=0.00006
Data=Data.fillna(mask_na)
#Data=Data.replace(0,np.nan)
#Data=Data[Data!=0].dropna()
#print(globalWarming_df)

# set heatmap size
plt.figure(figsize= (15,120), dpi=500) 
 
# Standardize or Normalize every column in the figure
# Standardize:
# set heatmap size
cm=sns.clustermap(Data, center=0, annot=True, annot_kws={"size": 8}, 
                  col_cluster=False, cmap='Blues', 
                  figsize= (15,120), cbar_pos=(1.05, .2, .03, .4), 
                  linewidths=0.004, yticklabels=True,xticklabels=True)
cm.ax_heatmap.yaxis.set_ticks_position("right")
ax = cm.ax_heatmap
ax.tick_params(labelbottom=False,labeltop=True)
#ax.set_xlabel("")
ax.set_ylabel("Predicting imminent discharge within 24 hours in emergency patients")
plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
#plt.title("Predicting imminent discharge within 24 hours in planned patients", fontsize = 12)
#plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
plt.tight_layout()
plt.show()
fig = ax.get_figure()
fig.set_size_inches(60, fig.get_figheight(), forward=True)
fig.savefig(Path+'Feature importance clustermap for emergency patients.svg', format='svg', dpi=300)
fig.savefig(Path+'Feature importance clustermap for emergency patients.pdf', format='pdf', dpi=300)


import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
# load dataset and create DataFrame ready to create heatmap for feature category importance for emergency
Data = pd.read_csv(Path+"24-hour discharge emergency feature importance statistics.csv")
Data=Data[['Features','DatabaseEmergencyD1', 'DatabaseEmergencyD2',
       'DatabaseEmergencyD3', 'DatabaseEmergencyD4', 'DatabaseEmergencyD5',
       'DatabaseEmergencyD6', 'DatabaseEmergencyD7D10','DatabaseEmergencyD11D13',
       'DatabaseEmergencyD14D20', 'DatabaseEmergencyD21D27', 'DatabaseEmergencyD28+']]
featureMapping=pd.read_csv("../Data/iord_lookups/FeatureCategoryMapping.csv")
Data=pd.merge(Data, featureMapping, on='Features', how='left').groupby(['Category']).mean()
# set heatmap size
plt.figure(figsize= (15,10), dpi=500) 
 
# Standardize or Normalize every column in the figure
# Standardize:
# set heatmap size
DataPlot=Data
DataPlot.columns=['In hospital 1 day', 'In hospital 2 days', 'In hospital 3 days', 'In hospital 4 days', 'In hospital 5 days', 'In hospital 6 days', 'In hospital 7-10 days',
                  'In hospital 11-13 days', 'In hospital 14-20 days', 'In hospital 21-27 days', 'In hospital 28 days longer']
# standard_scale=1, 
cm=sns.clustermap(DataPlot, center=0, annot=True, fmt='.0f', annot_kws={'fontsize':8,'fontstyle':'italic', 'verticalalignment':'center'}, 
                  col_cluster=False, cmap='Blues', 
                  figsize= (15,10), cbar_pos=(1.05, .4, .03, .3), 
                  linewidths=0.004, yticklabels=True,xticklabels=True)
cm.ax_heatmap.yaxis.set_ticks_position("right")
ax = cm.ax_heatmap
ax.tick_params(labelbottom=True,labeltop=False)
#ax.set_xlabel("")
#ax.set_ylabel("Predicting imminent discharge within 24 hours in emergency patients")
plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
#plt.title("Predicting 24-hour discharge \n in emergency patients", fontsize = 12)
#plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
plt.tight_layout()
fig = ax.get_figure()
fig.set_size_inches(15, fig.get_figheight(), forward=True)
fig.savefig(Path+'Feature category importance clustermap for emergency patients.svg', format='svg', dpi=500)
fig.savefig(Path+'Feature categoryimportance clustermap for emergency patients.pdf', format='pdf', dpi=500)
DataPlot.to_csv(Path+'Feature category importance clustermap for emergency patients.csv')



import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
# load dataset and create DataFrame ready to create heatmap for feture category importance for planned
Data = pd.read_csv(Path+"24-hour discharge planed feature importance statistics.csv")
Data=Data[['Features','DatabasePlanedD1', 'DatabasePlanedD2', 'DatabasePlanedD3', 'DatabasePlanedD4', 'DatabasePlanedD5',
       'DatabasePlanedD6', 'DatabasePlanedD7D10','DatabasePlanedD11D13', 
           'DatabasePlanedD14D20','DatabasePlanedD21D27', 'DatabasePlanedD28+']]
featureMapping=pd.read_csv("../Data/iord_lookups/FeatureCategoryMapping.csv")
Data=pd.merge(Data, featureMapping, on='Features', how='left').groupby(['Category']).mean()
# set heatmap size
plt.figure(figsize= (15,10), dpi=500) 
# Standardize or Normalize every column in the figure
# Standardize:
# set heatmap size
DataPlot=Data
DataPlot.columns=['In hospital 1 day', 'In hospital 2 days', 'In hospital 3 days', 'In hospital 4 days', 'In hospital 5 days', 'In hospital 6 days', 'In hospital 7-10 days',
                  'In hospital 11-13 days', 'In hospital 14-20 days', 'In hospital 21-27 days', 'In hospital 28 days longer']
# standard_scale=1,
cm=sns.clustermap(DataPlot, center=0, annot=True, fmt='.0f', annot_kws={"size": 8}, 
                  col_cluster=False,  cmap='OrRd', 
                  figsize= (15,10), cbar_pos=(1.05, .2, .03, .4), 
                  linewidths=0.004, yticklabels=True,xticklabels=True)
cm.ax_heatmap.yaxis.set_ticks_position("right")
ax = cm.ax_heatmap
ax.tick_params(labelbottom=True,labeltop=False)
#ax.set_xlabel("")
#ax.set_ylabel("Predicting imminent discharge within 24 hours in planned patients")
plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
#plt.title("Predicting imminent discharge within 24 hours in planned patients", fontsize = 12)
#plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
plt.tight_layout()
plt.show()
fig = ax.get_figure()
fig.set_size_inches(15, fig.get_figheight(), forward=True)
fig.savefig(Path+'Feature category importance clustermap for planed patients.svg', format='svg', dpi=300)
fig.savefig(Path+'Feature category importance clustermap for planed patients.pdf', format='pdf', dpi=300)
DataPlot.to_csv(Path+'Feature category importance clustermap for planed patients.csv')




import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
Data = pd.read_csv(Path+"Feature category importance clustermap for emergency and planned patients.csv")
Data=Data.set_index('Category')
print(Data)
# set heatmap size
plt.figure(figsize= (30,12), dpi=500) 

# Standardize or Normalize every column in the figure
# Standardize:
# set heatmap size
DataPlot=Data
cm=sns.clustermap(DataPlot, center=0, annot=True, fmt='.0f', annot_kws={'fontsize':8,'fontstyle':'italic', 'verticalalignment':'center'}, 
                  col_cluster=False, cmap='Blues', 
                  figsize= (40,10), cbar_pos=(1.05, .4, .03, .3), 
                  linewidths=0.004, yticklabels=True,xticklabels=True)
cm.ax_heatmap.yaxis.set_ticks_position("right")
ax = cm.ax_heatmap
ax.tick_params(labelbottom=True,labeltop=False)
#ax.set_xlabel("")
#ax.set_ylabel("Predicting imminent discharge within 24 hours in emergency patients")
plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
#plt.title("Predicting 24-hour discharge \n in emergency and planned patients", fontsize = 12)
#plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
plt.tight_layout()
fig = ax.get_figure()
fig.set_size_inches(15, fig.get_figheight(), forward=True)
fig.savefig(Path+'Feature category importance clustermap for emergency and planned patients.svg', format='svg', dpi=500)
fig.savefig(Path+'Feature categoryimportance clustermap for emergency and planned patients.pdf', format='pdf', dpi=500)
#DataPlot.to_csv(Path+'Feature category importance clustermap for emergency and planned patients.csv')




import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
Data = pd.read_csv(Path+"Feature category importance clustermap for emergency and planned patients.csv")
#Data['Category']=Data['Category'].replace("Blood/liver/renal/cardiac/clotting/inflammatory/acid-base tests","Laboratory tests")
Data=Data.set_index('Category')
print(Data)

# set heatmap size
plt.figure(figsize= (30,10), dpi=500) 
 
# Standardize or Normalize every column in the figure
# Standardize:
# set heatmap size

DataPlot=Data
cm=sns.clustermap(DataPlot, center=0, annot=True, fmt='.0f', annot_kws={'fontsize':8,'fontstyle':'italic', 'verticalalignment':'center'}, 
                  col_cluster=False, cmap='OrRd', 
                  figsize= (40,10), cbar_pos=(1.05, .4, .03, .3), 
                  linewidths=0.004, yticklabels=True,xticklabels=True)
cm.ax_heatmap.yaxis.set_ticks_position("right")
ax = cm.ax_heatmap
ax.tick_params(labelbottom=True,labeltop=False)
#ax.set_xlabel("")
#ax.set_ylabel("Predicting imminent discharge within 24 hours in emergency patients")
plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
#plt.title("Predicting 24-hour discharge \n in emergency and planned patients", fontsize = 12)
#plt.xlabel("Patient groups by duration from most recent admission date", fontsize = 12)
plt.tight_layout()
fig = ax.get_figure()
fig.set_size_inches(15, fig.get_figheight(), forward=True)
fig.savefig(Path+'Feature category importance clustermap for emergency and planned patients new.svg', format='svg', dpi=500)
fig.savefig(Path+'Feature categoryimportance clustermap for emergency and planned patients new.pdf', format='pdf', dpi=500)
#DataPlot.to_csv(Path+'Feature category importance clustermap for emergency and planned patients.csv')



# Wordcloud plot for emergency feature importance
import wordcloud
from wordcloud.wordcloud import WordCloud
from PIL import Image,ImageSequence
from wordcloud import WordCloud,ImageColorGenerator
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
Data = pd.read_csv(Path+"24-hour discharge emergency feature importance statistics.csv")
Datafp=pd.DataFrame()
Datafp['Features']=Data['Features']
Datafp['Mean importance']=pd.DataFrame(Data.mean(axis=1))
Datafp.columns=['name','val']
fp=Datafp
name = list(fp.name)#词
value = fp.val#词的频率
for i in range(len(name)):
    name[i] = str(name[i])
    #name[i] = name[i].decode('gb2312')
dic = dict(zip(name, value))#词频以字典形式存储

#参数分别是指定字体、背景颜色、最大的词的大小、使用给定图作为背景形状 #font_path="TimesNewRoman.ttf",
image = Image.open('../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/bg.png')#作为背景形状的图
graph = np.array(image)
wc = wordcloud.WordCloud(background_color="white", scale=9,relative_scaling=.5,mask=graph, max_words = 600,font_path = '/System/Library/Fonts/STHeiti Light.ttc')
wc.generate_from_frequencies(dic)#根据给定词频生成词云
image_color = ImageColorGenerator(graph)
fig = plt.figure(figsize=(20,10), dpi=150)
plt.tight_layout(pad=0)
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.show()
fig.savefig(Path+'WordCloud of aggregated individual importance for emergency patients.svg', dpi = 300)
fig.savefig(Path+'WordCloud of aggregated individual importance for emergency patients.pdf', dpi = 300)



# Wordcloud plot for emergency feature importance
import wordcloud
Path="../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/"
Data = pd.read_csv(Path+"24-hour discharge planed feature importance statistics.csv")
Datafp=pd.DataFrame()
Datafp['Features']=Data['Features']
Datafp['Mean importance']=pd.DataFrame(Data.mean(axis=1))
Datafp.columns=['name','val']
fp=Datafp
name = list(fp.name)#词
value = fp.val#词的频率
for i in range(len(name)):
    name[i] = str(name[i])
    #name[i] = name[i].decode('gb2312')
dic = dict(zip(name, value))#词频以字典形式存储

#参数分别是指定字体、背景颜色、最大的词的大小、使用给定图作为背景形状 #font_path="TimesNewRoman.ttf",
image = Image.open('../Patients flow prediction/Codes/Imbalance XGBoost/24HourDischarge/bg.png')#作为背景形状的图
graph = np.array(image)
wc = wordcloud.WordCloud(background_color="white", scale=8,relative_scaling=.5,mask=graph, max_words = 600,font_path = '/System/Library/Fonts/STHeiti Light.ttc')
wc.generate_from_frequencies(dic)#根据给定词频生成词云
image_color = ImageColorGenerator(graph)
fig = plt.figure(figsize=(20,10), dpi=150)
plt.tight_layout(pad=0)
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.show()
fig.savefig(Path+'WordCloud of aggregated individual importance for planned patients.svg', dpi = 300)
fig.savefig(Path+'WordCloud of aggregated individual importance for planned patients.pdf', dpi = 300)









