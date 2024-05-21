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

#n= x1.shape[1]
#for i in range(1,n):
#    pca = PCA(n_components=i)
#    pca.fit_transform(x1)
#    print(sum(pca.explained_variance_ratio_))


pca = PCA(n_components=9) # 주성분을 몇개로 할지 결정
principalComponents = pca.fit_transform(x1)
col_pc = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9']
principalDf = pd.DataFrame(data=principalComponents, columns = col_pc, index=df1.index)

# 데이터 불러오기
tt = pd.read_csv('train_test_set_template.csv')

# tt 파일에서 material 이름과 함량 추출하기
materials = tt.iloc[:, [0, 2, 4, 6, 8]].values
amounts = tt.iloc[:, [1, 3, 5, 7, 9]].astype(float).values

# tt 파일에서 Class 값 추출하기
classes = tt.iloc[:, -1].values

# 각 material에 대해 feature값과 함량을 곱한 뒤 더하기
features = []
for i in range(len(materials)):
    feature = np.zeros(9)
    for j in range(5):
        if pd.notnull(materials[i][j]):
            material_name = materials[i][j]
            amount = amounts[i][j]
            material_features = principalDf.loc[principalDf.index == material_name].iloc[:, :].values
            if len(material_features) > 0:
                material_feature = material_features[0]
                feature += material_feature * amount
            else:
                pass
            
    features.append(feature)

tt2 = pd.DataFrame(data = features, columns = col_pc)
tt2["Class"] = tt["Class"]


# 데이터 불러오기
#data = pd.read_csv('MCS_dataset_std_240503_FT4_rawdata.csv')

# feature와 target 나누기
X = tt2.iloc[:, :-1]
y = tt2.iloc[:, -1] -1

# train, test 데이터셋 나누기
rs = st.number_input('머신러닝을 위한 무작위 숫자 입력', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rs)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)


API_df = df[(df['Function']=='API')]
API_list = API_df.index.to_list()
API_name = st.selectbox('Select API', API_list)

API_content = st.text_input('API_content (%)')


Excipient_df = df[(df['Function']!='API')]
Excipient_list = Excipient_df.index.to_list()
Excipient1_name = st.selectbox(
    'Select Excipient_1',
    Excipient_list)

Excipient1_content = st.text_input('Excipient1_content (%)')


Excipient2_name = st.selectbox(
    'Select Excipient_2',
    Excipient_list)

Excipient2_content = st.text_input('Excipient2_content (%)')


Excipient3_name = st.selectbox(
    'Select Excipient_3',
    Excipient_list)

Excipient3_content = st.text_input('Excipient3_content (%)')


Excipient4_name = st.selectbox(
    'Select Excipient_4',
    Excipient_list)

Excipient4_content = st.text_input('Excipient4_content (%)')

models = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'XGBoost']  #'Decision Tree'
select = st.selectbox('Please select a model', models)

## Buttons
if st.button("Predict"):
    svc = SVC()
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)
    svc_acc = accuracy_score(y_test, svc_pred)

    #dt = DecisionTreeClassifier()
    #dt.fit(X_train, y_train)
    #dt_pred = dt.predict(X_test)
    #dt_acc = accuracy_score(y_test, dt_pred)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    accuracies = [lr_acc, svc_acc, rf_acc, xgb_acc]  #dt_acc

    fig1=plt.figure()
    plt.bar(models, accuracies)
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    model_acc = pd.DataFrame(data = accuracies, index = models, columns = ['Accuracy'])
    st.write(model_acc)



    if select == 'Logistic Regression':
        model = lr
    elif select == 'Support Vector Machine':
        model = svc
    #elif select == 'Decision Tree':
    #    model = dt
    elif select == 'Random Forest':
        model = rf
    elif select == 'XGBoost':
        model = xgb_model
    
    #if (model == dt or model == rf or model == xgb_model):
    #    # 모델에서 각 독립변수의 중요도 추출
    #    importance = model.feature_importances_
    #    # 중요도를 데이터프레임으로 변환
    #    df_importance = pd.DataFrame({'feature': data.columns[:-1], 'importance': importance})
    #    # 중요도를 내림차순으로 정렬
    #    df_importance = df_importance.sort_values('importance', ascending=False)
    #    # 중요도 시각화
    #    fig2=plt.figure()
    #    plt.bar(df_importance['feature'], df_importance['importance'])
    #    plt.xticks(rotation=45)
    #    plt.xlabel('Features')
    #    plt.ylabel('Importance')
    #    plt.title('Feature Importance')
    #    st.pyplot(fig2)

    API_content_f = float(API_content)
    Excipient1_content_f = float(Excipient1_content)
    Excipient2_content_f = float(Excipient2_content)
    Excipient3_content_f = float(Excipient3_content)
    Excipient4_content_f = float(Excipient4_content)
    mixture_data = z1.loc[API_name]*API_content_f/100 + z1.loc[Excipient1_name]*Excipient1_content_f/100 + z1.loc[Excipient2_name]*Excipient2_content_f/100 + z1.loc[Excipient3_name]*Excipient3_content_f/100 + z1.loc[Excipient4_name]*Excipient4_content_f/100
    mixture_df = mixture_data.to_frame()
    mixture_df = mixture_df.transpose()	#행 열 전환
    #Best model로 예측하기
    pred = model.predict(mixture_df)+1
    st.write("Mixture Class = " + str(pred[0]))   
    st.write("Class 1 : Direct Compression")
    st.write("Class 2 : Dry Granulation")
    st.write("Class 3 : Wet Granulation")
    st.write("Class 4 : Other Technology")
