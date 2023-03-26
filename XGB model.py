import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)    
import seaborn as sns
from xgboost import XGBClassifier
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt import fmin
from matplotlib import pyplot as plt


# read data
Path="../iord_extract_20220325/Databases/"
file='DatabaseEmergencyD1'
# testing data from 2017/2/1-2019/1/31
TrainData=pd.read_csv(Path+file+"Training.csv")
# testing data from 2019/2/1-2020/1/31
TestData=pd.read_csv(Path+file+"Testing.csv")

print('Imminent discharge within 24 hours in training: ', TrainData['Imminent discharge within 24 hours'].sum())
print('Imminent discharge within 24 hours in testing: ', TestData['Imminent discharge within 24 hours'].sum())
Outcome='Imminent discharge within 24 hours'
TrainY=pd.DataFrame(TrainData[Outcome])
cols=TrainData.columns[TrainData.columns.tolist().index("Age"):TrainData.shape[1]].values.tolist()
TrainX=TrainData[cols]
Train=TrainX
Train['target']=TrainY
print(Train.shape)
print(Train.columns)
test_x=TestData[cols]
test_x=np.array(test_x)
TestY=pd.DataFrame(TestData[Outcome])
TestY.columns=['target']
test_y=TestY.values
test_y=np.array(test_y)
print(test_y.shape)

# Feature selection via ReliefF
from ReliefF import ReliefF
Train.target.value_counts()
Train = Train.replace((np.inf, -np.inf), 0)
fs = ReliefF(n_neighbors=1, n_features_to_keep=150)
TrainSelected = fs.fit_transform(np.array(Train[cols]), np.array(Train['target']))
print(pd.DataFrame(TrainSelected).head())
print("--------------")
print("(No. of tuples, No. of Columns before ReliefF) : "+str(Train.shape)+ "\n(No. of tuples , No. of Columns after ReliefF) : "+str(TrainSelected.shape))


# Descriptive statistics
df=TrainSelected
description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])
numerical = []
categorical = []
# Construct a dataframe of Santander metadata
for col in df.columns:
    obs = df[col].size
    p_nan = round(df[col].isna().sum()/obs, 2)
    num_nan = f'{p_nan}% ({df[col].isna().sum()}/{obs})'
    dtype = 'categorical' if df[col].dtype == object else 'numerical'
    numerical.append(col) if dtype == 'numerical' else categorical.append(col)
    rng = f'{len(df[col].unique())} labels' if dtype == 'categorical' else f'{df[col].min()}-{df[col].max()}'
    description[col] = [obs, num_nan, dtype, rng]

pd.set_option('display.max_columns', 150)
print(description)
print(df.shape)


# Outcome distribution
n=df.shape[0]
sample = df.sample(n)
class_1 = len(sample[sample.target == 1])
class_0 = len(sample[sample.target== 0])
plt.bar(['Negative', 'Positive'], [class_0, class_1])
plt.title('Size' + str(n) + 'Sample Distribution')
plt.xlabel('Target Class Label')
plt.ylabel('Count of Customers')
plt.show()
print(f'Negative: {class_0} \n Positive: {class_1}')


# Organizes XGB results and extracts metadata from Trials object
def org_results(trials, hyperparams, ratio, model_name):
    fit_idx = -1
    for idx, fit  in enumerate(trials):
        hyp = fit['misc']['vals']
        xgb_hyp = {key:[val] for key, val in hyperparams.items()}
        if hyp == xgb_hyp:
            fit_idx = idx
            break
            
    train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])
    acc = round(trials[fit_idx]['result']['accuracy'], 3)
    train_auc = round(trials[fit_idx]['result']['train auc'], 3)
    test_auc = round(trials[fit_idx]['result']['test auc'], 3)
    conf_matrix = trials[fit_idx]['result']['conf matrix']

    results = {
        'model': model_name,
        'ratio': ratio,
        'parameter search time': train_time,
        'accuracy': acc,
        'test auc score': test_auc,
        'training auc score': train_auc,
        'confusion matrix': conf_matrix,
        'parameters': hyperparams
    }
    return results
def data_ratio(y):
    unique, count = np.unique(y, return_counts=True)
    ratio = round(count[0]/count[1], 2)
    return f'{ratio}:1 ({count[0]}/{count[1]})'

# balanced training data
batch_size = df.shape[0]
xgb_df = df.sample(batch_size)
print('balanced', df.shape, xgb_df.shape)
y = xgb_df['target'].reset_index(drop=True)
x = xgb_df.drop(columns=['target'])
from imblearn.combine import SMOTETomek 
from imblearn.combine import SMOTEENN
smotomek = SMOTETomek(random_state=42)
bal_x, bal_y= smotomek.fit_resample(x, y)
bal_train_x = np.array(bal_x)
bal_train_y = np.array(bal_y)
print('balanced', bal_x.shape, bal_y.shape)
print('balanced: Number of positive events', bal_y.sum())
# imbalanced training data
samp_len = len(bal_y)
xgb_df2 = df.sample(samp_len - batch_size)
xgb_df = pd.concat([xgb_df, xgb_df2])
print('imbalanced', df.shape, xgb_df.shape, xgb_df2.shape)
imb_y = xgb_df['target'].reset_index(drop=True)
imb_x = xgb_df.drop(columns=['target'])
imb_train_x = np.array(imb_x)
imb_train_y = np.array(imb_y)
print('imbalanced', imb_x.shape, imb_y.shape)
print('imbalanced: Number of positive events', imb_y.sum())

# xgb model train
def xgb_train(data_y, md_name,train_x, test_x, train_y, test_y):
    print(test_x.shape, test_y.shape)
    ratio = data_ratio(data_y)
    def xgb_objective(space, early_stopping_rounds=50):
        model = XGBClassifier(
            learning_rate = space['learning_rate'], 
            n_estimators = int(space['n_estimators']), 
            max_depth = int(space['max_depth']), 
            min_child_weight = space['m_child_weight'], 
            gamma = space['gamma'], 
            subsample = space['subsample'], 
            colsample_bytree = space['colsample_bytree'],
            objective = 'binary:logistic')
        model.fit(train_x, train_y, 
                  eval_set = [(train_x, train_y), (test_x, test_y)],
                  eval_metric = 'auc',
                  early_stopping_rounds = early_stopping_rounds,
                  verbose = False)
        # variable importance ranking
        n_top_features = 400
        sorted_idx = model.feature_importances_.argsort()[::-1]
        #plt.barh(imb_x.columns[sorted_idx][:n_top_features], model.feature_importances_[sorted_idx][:n_top_features])
        #plt.show()
        fImportance=pd.DataFrame(np.zeros((n_top_features,2)))
        fImportance.columns=['Features', 'Importance']
        fImportance['Features']=imb_x.columns[sorted_idx][:n_top_features].tolist()
        fImportance['Importance']=model.feature_importances_[sorted_idx][:n_top_features].tolist()
        fImportance.to_csv(file+' feature importance '+md_name+'.csv', index=False)
        # predictions
        predictions = model.predict(test_x)
        TestData['Predicted ('+md_name+' Model)']=predictions
        test_preds = model.predict_proba(test_x)[:,1]
        TestData['Predicted probability ('+md_name+' Model)']=test_preds
        train_preds = model.predict_proba(train_x)[:,1]
        xgb_booster = model.get_booster()
        train_auc = roc_auc_score(train_y, train_preds)
        test_auc = roc_auc_score(test_y, test_preds)
        accuracy = accuracy_score(test_y, predictions) 
        conf_matrix = confusion_matrix(test_y, predictions)
        print('Confusion matrix: ', conf_matrix)
        print('Classification report: ', classification_report(test_y,predictions))
        return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,
                'test auc': test_auc, 'train auc': train_auc, 'conf matrix': conf_matrix}
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),
        'max_depth': hp.quniform('max_depth', 1, 12, 1),
        'm_child_weight': hp.quniform('m_child_weight', 1, 6, 1),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'learning_rate': hp.loguniform('learning_rate', np.log(.001), np.log(.3)),
        'colsample_bytree': hp.quniform('colsample_bytree', .5, 1, .1)}
    trials = Trials()
    xgb_hyperparams = fmin(fn = xgb_objective, 
                     max_evals = 25, 
                     trials = trials,
                     algo = tpe.suggest,
                     space = space) 
    results = org_results(trials.trials, xgb_hyperparams, ratio, md_name)
    return results
imb_results = xgb_train(imb_y, 'Imbalanced Data', imb_train_x, test_x, imb_train_y, test_y)
bal_results = xgb_train(bal_y, 'Balanced Data', bal_train_x, test_x, bal_train_y, test_y)
print(bal_results)
print(imb_results)

# save predicted probabilities for testing data
print(TestData.columns)
TestData.to_csv("../Predicted EmergencyD1 testing ("+Outcome+").csv", index=False)
# confusion matrix
bal_confusion = bal_results.pop('confusion matrix')
print(bal_confusion)
imb_confusion = imb_results.pop('confusion matrix')


# Calculate evaluation metrics
def CalculateF1(smat): 
    tn = smat[0][0] 
    fp = smat[0][1] 
    fn = smat[1][0] 
    tp = smat[1][1] 
    PPV=tp/(tp+fp)
    NPV=tn/(tn+fn)
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    F1=2*(precision*recall)/(precision+recall)
    return [np.round(PPV, 3), np.round(NPV, 3), np.round(sensitivity, 3), np.round(specificity, 3), 
            np.round(precision, 3), np.round(recall, 3), np.round(F1,3)]
result=pd.DataFrame(np.zeros((2,8)))
result.columns=['test PPV','test NPV','test sensitivity','test specificity',
                'test precision', 'test recall', 'test F1 score', 'Average daily accuracy (%)']
result['model']=['Balanced','Imbalanced']
baF1=CalculateF1(bal_confusion)
result.iloc[0,0]=baF1[0]
result.iloc[0,1]=baF1[1]
result.iloc[0,2]=baF1[2]
result.iloc[0,3]=baF1[3]
result.iloc[0,4]=baF1[4]
result.iloc[0,5]=baF1[5]
result.iloc[0,6]=baF1[6]
ibaF1=CalculateF1(imb_confusion)
result.iloc[1,0]=ibaF1[0]
result.iloc[1,1]=ibaF1[1]
result.iloc[1,2]=ibaF1[2]
result.iloc[1,3]=ibaF1[3]
result.iloc[1,4]=ibaF1[4]
result.iloc[1,5]=ibaF1[5]
result.iloc[1,6]=ibaF1[6]
print(result)

# report prediciton performance metrics
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks
    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks
    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks
    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""
    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')
    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False
    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    if xyplotlabels:
        plt.ylabel('True label')
    else:
        plt.xlabel(stats_text)   
    if title:
        plt.title(title)
    plt.savefig(''.join(title.split('\n'))+'.png', dpi=300)
        
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Zero', 'One']
make_confusion_matrix(cf=bal_confusion, 
                      group_names=labels,
                      categories=categories, 
                      cmap='viridis_r', title=file+'\n '+Outcome+'\n Balanced Dataset')
make_confusion_matrix(cf=imb_confusion, 
                      group_names=labels,
                      categories=categories, 
                      cmap='viridis_r', title=file+'\n '+Outcome+'\n Imbalanced Dataset')

# summarize results
final_results = pd.DataFrame([bal_results, imb_results])
for i in range(result.shape[1]):
    final_results[result.columns[i]]=result[result.columns[i]]
col = final_results.pop("parameters")
final_results.insert(14, col.name, col)
final_results.index=['Balanced', 'Imbalanced']
del final_results['model']
final_results=final_results.T
print(final_results) 
final_results.to_csv('Summary '+file+'.csv')







