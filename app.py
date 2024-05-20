import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report


st.title('Manufacturing Classification System')  # 타이틀명 지정



#표준화 변환 및 csv로 변환 (MCS_dataset_std_240503_FT4_rawdata.csv 있으면 생략가능)

#loading database and 전처리
filename = 'FT4_DB_Sep2023+sorbitol_mean1.csv'
df = pd.read_csv(filename)
df = df.set_index('Material')

#표준화 변환 및 csv로 변환
df1=df.iloc[:, 2:]
x1 = df1.values  # 독립변인들의 value값만 추출
x1 = StandardScaler().fit_transform(x1)  # x객체에 x를 표준화한 데이터를 저장
features = df1.columns
z1 = pd.DataFrame(x1, columns=features, index=df1.index)


# 데이터 불러오기
data = pd.read_csv('MCS_dataset_std_240503_FT4_rawdata.csv')

# feature와 target 나누기
X = data.iloc[:, :-1]
y = data.iloc[:, -1] -1

# train, test 데이터셋 나누기
rs = st.number_input('머신러닝을 위한 무작위 숫자 입력', 1030)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rs)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

models = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'XGBoost']
accuracies = [lr_acc, svc_acc, dt_acc, rf_acc, xgb_acc]

fig1=plt.figure()
plt.bar(models, accuracies)
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
st.pyplot(fig1)
st.write('Logistic Regression Accuracy:', lr_acc)
st.write('Support Vector Machine:', svc_acc)
st.write('Decision Tree Accuracy:', dt_acc)
st.write('Random Forest Accuracy:', rf_acc)
st.write('XGBoost Accuracy:', xgb_acc)


model = st.selectbox('Please select a model', models)
st.write('You selected:', model)


# 모델에서 각 독립변수의 중요도 추출
importance = model.feature_importances_

# 중요도를 데이터프레임으로 변환
df_importance = pd.DataFrame({'feature': data.columns[:-1], 'importance': importance})

# 중요도를 내림차순으로 정렬
df_importance = df_importance.sort_values('importance', ascending=False)

# 중요도 시각화
fig2=plt.figure()
plt.bar(df_importance['feature'], df_importance['importance'])
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
st.pyplot(fig2)



API_df = df[(df['Function']=='API')]
API_list = API_df.index.to_list()
API_name = st.selectbox('Select API', API_list)

API_content = st.text_input('API_content (%)')

Excipient_df = df[(df['Function']!='API')]
Excipient_list = Excipient_df.index.to_list()
Excipient_1_name = st.selectbox(
    'Select Excipient_1',
    Excipient_list)

Excipient_1_content = st.text_input('Excipient_1_content (%)')

Excipient_2_name = st.selectbox(
    'Select Excipient_2',
    Excipient_list)

Excipient_2_content = st.text_input('Excipient_2_content (%)')

Excipient_3_name = st.selectbox(
    'Select Excipient_3',
    Excipient_list)

Excipient_3_content = st.text_input('Excipient_3_content (%)')

Excipient_4_name = st.selectbox(
    'Select Excipient_4',
    Excipient_list)

Excipient_4_content = st.text_input('Excipient_4_content (%)')

mixture_data = z1.loc[API_name]*API_content/100 + z1.loc[Excipient1_name]*Excipient1_content/100 + z1.loc[Excipient2_name]*Excipient2_content/100 + z1.loc[Excipient3_name]*Excipient3_content/100 + z1.loc[Excipient4_name]*Excipient4_content/100
mixture_df = mixture_data.to_frame()
mixture_df = mixture_df.transpose()	#행 열 전환

## Buttons
if st.button("Predict"):
    #Best model로 예측하기
    pred = model.predict(mixture_df)+1
    st.write("Mixture Class = " + str(pred[0]))   
    st.write("Class 1 : Direct Compression")
    st.write("Class 2 : Dry Granulation")
    st.write("Class 3 : Wet Granulation")
    st.write("Class 4 : Other Technology")
