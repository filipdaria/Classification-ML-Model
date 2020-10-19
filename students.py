
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report, recall_score


df = pd.read_csv('all_stud.csv')
pd.set_option("display.max_rows", None, 'display.max_columns', None)
fig, ax = plt.subplots()

#removing outliers

df['punishment'].value_counts().plot(ax=ax, kind='bar')
df = df[df.punishment != 'why do you ask']
df = df[df.punishment != 'smily face']
df['target_var'].value_counts().plot(ax=ax, kind='bar')
index1 = df[(df['target_var'] != "deadlock") & (df['target_var'] !='challenge') & (df['target_var'] !='way for improvement')].index
df.drop(index1, inplace=True)
df['target_var'].replace('way for improvement','challenge', inplace=True)

# encode input variables

to_convert = ['school_attention', 'involvement', 'answers' ,'punishment']
df[to_convert] = df[to_convert].astype('category')
df['school_attention'] = df['school_attention'].cat.codes
df['involvement']=df['involvement'].cat.codes
df['answers']=df['answers'].cat.codes
df['punishment'] = df['punishment'].cat.codes

# encode target variable

label_encoder = LabelEncoder()
target = label_encoder.fit_transform(df['target_var'])
df['target_var'] = target

#correlation between features and target variable

print(df[df.columns[1:]].corr()['target_var'][:])
print(df[df.columns[1:]].corr()['involvement'][:])
print(df[[ 'school_enjoyment', 'school_attention', 'involvement']].corr()['school_attention'][:])

# feature importance

X =np.array(df.loc[:, df.columns != 'target_var'])
y = np.array(df.target_var)
clf = LassoCV().fit(X, y)
importance = np.abs(clf.coef_)
print(importance)
idx_fifth = importance.argsort()[-5]
threshold = importance[idx_fifth] + 0.01
feature_names = list(df.columns)
idx_features = (-importance).argsort()[:4]
name_features = np.array(feature_names)[idx_features]
print('Selected features: {}'.format(name_features))
df = df[name_features]
sfm = SelectFromModel(clf, threshold=threshold)
sfm.fit(X, y)
X_transform = sfm.transform(X)
n_features = sfm.transform(X).shape[1]
X_new = np.array(df.loc[:])

#Cross Validation

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cnt = 1
for train_index, test_index in kf.split(X_new, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
    
score = cross_val_score(LogisticRegression(random_state= 0), X_new, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold with Logistic Regression are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

score = cross_val_score(DecisionTreeClassifier(random_state= 0), X_new, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold with DecisionTreeClassifier are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

score = cross_val_score(RandomForestClassifier(random_state= 0), X_new, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold with RandomForrestClassifier are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

score = cross_val_score(SVC(random_state= 0), X_new, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold with SVC are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

#make predictions

X_train, X_test,Y_train, Y_test, = train_test_split(X_new,y,test_size=0.3, random_state=0)
model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

#evaluate predictions
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

#parameter tuning

gbc=GradientBoostingClassifier()
n_estimators=[100,300,400]
learning_rate=[0.01,0.1,1]
max_depth=[2,3,5]
params2={'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depth}
grid_gbc=GridSearchCV(gbc,param_grid=params2)
grid_gbc.fit(X_train,Y_train)
print(grid_gbc.best_params_)
gbclf=GradientBoostingClassifier(learning_rate=0.01,max_depth=5,n_estimators=500)
gbclf.fit(X_new,y)
print(gbclf.score(X_new,y))
