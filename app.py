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
from mpl_toolkits.mplot3d import axes3d


st.header('Manufacturing Classification System')  # 타이틀명 지정



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

st.subheader(" ")
st.subheader('Formulation Design')

API_df = df[(df['Function']=='API')]
API_list = API_df.index.to_list()
API_name = st.selectbox('Select API', API_list, index = 52)

API_content = st.number_input('API_content (%)', 0, 100, value = 30)


Filler_df =  df[df['Function'].isin(['Filler_DC', 'Filler_WG'])]
Filler_list = Filler_df.index.to_list()

Binder_df =  df[(df['Function']=='Binder')]
Binder_list = Binder_df.index.to_list()

Disintegrant_df =  df[(df['Function']=='Disintegrant')]
Disintegrant_list = Disintegrant_df.index.to_list()

Excipient1_name = st.selectbox(
    'Select Filler 1',
    Filler_list, index = 22)

Excipient1_content = st.number_input('Filler 1 content (%)',0, 100, value = 30 )


Excipient2_name = st.selectbox(
    'Select Filler 2',
    Filler_list,index = 23)

Excipient2_content = st.number_input('Filler 2 content (%)',0, 100, value = 30 )


Excipient3_name = st.selectbox(
    'Select Binder',
    Binder_list, index = 2)

Excipient3_content = st.number_input('Binder content (%)',0, 100, value = 5)


Excipient4_name = st.selectbox(
    'Select Disintegrant',
    Disintegrant_list, index = 3)

Excipient4_content = st.number_input('Disintegrant content (%)',0, 100, value = 5)


st.subheader(" ")
st.subheader('Analytical Condition')

n= x1.shape[1]

exp_vr = []
comp = []

for i in range(1,n+1):
    pca = PCA(n_components=i)
    pca.fit_transform(x1)
    exp_vr.append(sum(pca.explained_variance_ratio_))
    comp.append(str(i))

explained_vraiance_ratio = pd.DataFrame(data= exp_vr, columns = ["Explained Variance Ratio of PCA"])
n_components = pd.DataFrame(data= comp, columns = ["n_components"])
evr = pd.concat([explained_vraiance_ratio, n_components], axis = 1)
evr = evr.set_index('n_components')


num_pc = st.number_input('Set the number of princial components', 1, 21, value = 9) 
pca = PCA(n_components=num_pc) # 주성분을 몇개로 할지 결정
principalComponents = pca.fit_transform(x1)

col_pc = []
for i in range(1,num_pc+1):
    col_pc.append("pc"+str(i))
    
principalDf = pd.DataFrame(data=principalComponents, columns = col_pc, index=df1.index)

with st.expander('num_PC vs Explained Variance Ratio'):
      st.write(evr)

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
    feature = np.zeros(num_pc)
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
rs = st.number_input('Input a random seed for machine learning', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rs)

models = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'XGBoost']  #'Decision Tree'
select = st.selectbox('Please select a model', models)

lr = LogisticRegression()
#lr.fit(X_train, y_train)
#lr_pred = lr.predict(X_test)
#lr_acc = accuracy_score(y_test, lr_pred)
    
svc = SVC()
#svc.fit(X_train, y_train)
#svc_pred = svc.predict(X_test)
#svc_acc = accuracy_score(y_test, svc_pred)

#dt = DecisionTreeClassifier()
#dt.fit(X_train, y_train)
#dt_pred = dt.predict(X_test)
#dt_acc = accuracy_score(y_test, dt_pred)

rf = RandomForestClassifier()
#rf.fit(X_train, y_train)
#rf_pred = rf.predict(X_test)
#rf_acc = accuracy_score(y_test, rf_pred)

xgb_model = xgb.XGBClassifier()
#xgb_model.fit(X_train, y_train)
#xgb_pred = xgb_model.predict(X_test)
#xgb_acc = accuracy_score(y_test, xgb_pred)
    
#accuracies = [lr_acc, svc_acc, rf_acc, xgb_acc]  #dt_acc

#fig1=plt.figure()
#plt.bar(models, accuracies)
#plt.ylim([0, 1])
#plt.ylabel('Accuracy')
#plt.xticks(rotation=45)
#st.pyplot(fig1)
#model_acc = pd.DataFrame(data = accuracies, index = models, columns = ['Accuracy'])
#st.write(model_acc)



st.subheader(" ")
## Buttons
if st.button("Predict"):
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

    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_acc = round(accuracy_score(y_test, model_pred), 4)
    



    
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
    mixture_data = principalDf.loc[API_name]*API_content_f/100 + principalDf.loc[Excipient1_name]*Excipient1_content_f/100 + principalDf.loc[Excipient2_name]*Excipient2_content_f/100 + principalDf.loc[Excipient3_name]*Excipient3_content_f/100 + principalDf.loc[Excipient4_name]*Excipient4_content_f/100
    mixture_df = mixture_data.to_frame()
    mixture_df = mixture_df.transpose()	#행 열 전환
    
    #Best model로 예측하기
    pred = model.predict(mixture_df)+1
    
    st.write("Predicted Manfacturing Class = " + str(pred[0]))
    st.write("Model Accuracy : " + str(model_acc))
    st.write("Class 1 : Direct Compression")
    st.write("Class 2 : Dry Granulation")
    st.write("Class 3 : Wet Granulation")
    st.write("Class 4 : Other Technology")
    if num_pc >= 3:
        fig = plt.figure(constrained_layout=True, figsize=(12,9))
        ax = fig.add_subplot(221, projection='3d')
        ax_xy = fig.add_subplot(222, projection='3d')
        ax_yz= fig.add_subplot(223, projection='3d')
        ax_zx = fig.add_subplot(224, projection='3d')
        
        x1 = tt2[tt2["Class"] == 1]["pc1"]
        y1 = tt2[tt2["Class"] == 1]["pc2"]
        z1 = tt2[tt2["Class"] == 1]["pc3"]

        x2 = tt2[tt2["Class"] == 2]["pc1"]
        y2 = tt2[tt2["Class"] == 2]["pc2"]
        z2 = tt2[tt2["Class"] == 2]["pc3"]

        x3 = tt2[tt2["Class"] == 3]["pc1"]
        y3 = tt2[tt2["Class"] == 3]["pc2"]
        z3 = tt2[tt2["Class"] == 3]["pc3"]

        x4 = tt2[tt2["Class"] == 4]["pc1"]
        y4 = tt2[tt2["Class"] == 4]["pc2"]
        z4 = tt2[tt2["Class"] == 4]["pc3"]

        xm = mixture_df["pc1"]
        ym = mixture_df["pc2"]
        zm = mixture_df["pc3"]
        xm_f = round(float(xm), 1)
        ym_f = round(float(ym), 1)
        zm_f = round(float(zm), 1)

        
        ax.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax.scatter(x3, y3, z3, color = 'r', alpha = 0.5, label = 'Class 3')
        ax.scatter(x4, y4, z4, color = 'gray', alpha = 0.5, label = 'Class 4')
        ax.scatter(xm, ym, zm , s=100, color = 'black', alpha = 0.5, marker='*', label = 'Mixture')
        ax.text(xm_f, ym_f, zm_f, f'({xm_f}, {ym_f}, {zm_f})', color='black')


        ax_xy.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax_xy.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax_xy.scatter(x3, y3, z3, color = 'r', alpha = 0.5, label = 'Class 3')
        ax_xy.scatter(x4, y4, z4, color = 'gray', alpha = 0.5, label = 'Class 4')
        ax_xy.scatter(xm, ym, zm , s=100, color = 'black', alpha = 0.5, marker='*', label = 'Mixture')
        ax_xy.text(xm_f, ym_f, zm_f, f'({xm_f}, {ym_f}, {zm_f})', color='black')

        ax_yz.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax_yz.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax_yz.scatter(x3, y3, z3, color = 'r', alpha = 0.5, label = 'Class 3')
        ax_yz.scatter(x4, y4, z4, color = 'gray', alpha = 0.5, label = 'Class 4')
        ax_yz.scatter(xm, ym, zm , s=100, color = 'black', alpha = 0.5, marker='*', label = 'Mixture')
        ax_yz.text(xm_f, ym_f, zm_f, f'({xm_f}, {ym_f}, {zm_f})', color='black')

        ax_zx.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax_zx.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax_zx.scatter(x3, y3, z3, color = 'r', alpha = 0.5, label = 'Class 3')
        ax_zx.scatter(x4, y4, z4, color = 'gray', alpha = 0.5, label = 'Class 4')
        ax_zx.scatter(xm, ym, zm , s=100, color = 'black', alpha = 0.5, marker='*', label = 'Mixture')
        ax_zx.text(xm_f, ym_f, zm_f, f'({xm_f}, {ym_f}, {zm_f})', color='black')

        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        #ax.view_init(20,60)
        
        ax_xy.set_xlabel('pc1')
        ax_xy.set_ylabel('pc2')
        ax_xy.set_zlabel('pc3')
        ax_xy.view_init(100,0)

        ax_yz.set_xlabel('pc1')
        ax_yz.set_ylabel('pc2')
        ax_yz.set_zlabel('pc3')
        ax_yz.view_init(0,0)

        ax_zx.set_xlabel('pc1')
        ax_zx.set_ylabel('pc2')
        ax_zx.set_zlabel('pc3')
        ax_zx.view_init(0,100)
        


    elif num_pc == 2:
        fig = plt.figure(constrained_layout=True, figsize=(12,9))
                      
        x1 = tt2[tt2["Class"] == 1]["pc1"]
        y1 = tt2[tt2["Class"] == 1]["pc2"]
        
        x2 = tt2[tt2["Class"] == 2]["pc1"]
        y2 = tt2[tt2["Class"] == 2]["pc2"]
       
        x3 = tt2[tt2["Class"] == 3]["pc1"]
        y3 = tt2[tt2["Class"] == 3]["pc2"]
       
        x4 = tt2[tt2["Class"] == 4]["pc1"]
        y4 = tt2[tt2["Class"] == 4]["pc2"]
        
        xm = mixture_df["pc1"]
        ym = mixture_df["pc2"]
       
        xm_f = round(float(xm), 1)
        ym_f = round(float(ym), 1)
        
        plt.scatter(x1, y1, color = 'b', alpha = 0.5, label = 'Class 1')
        plt.scatter(x2, y2, color = 'g', alpha = 0.5, label = 'Class 2')
        plt.scatter(x3, y3, color = 'r', alpha = 0.5, label = 'Class 3')
        plt.scatter(x4, y4, color = 'gray', alpha = 0.5, label = 'Class 4')
        plt.scatter(xm, ym, s=100, color = 'black', alpha = 0.5, marker='*', label = 'Mixture')
        plt.text(xm_f, ym_f, f'({xm_f}, {ym_f})', color='black')


        plt.xlabel('pc1')
        plt.ylabel('pc2')
        
    elif num_pc == 1:
        fig = plt.figure(constrained_layout=True, figsize=(12,9))
                      
        x1 = tt2[tt2["Class"] == 1]["pc1"]               
        x2 = tt2[tt2["Class"] == 2]["pc1"]              
        x3 = tt2[tt2["Class"] == 3]["pc1"]              
        x4 = tt2[tt2["Class"] == 4]["pc1"]
               
        xm = mixture_df["pc1"]
              
        xm_f = round(float(xm), 1)
               
        plt.scatter(tt2.index, x1, color = 'b', alpha = 0.5, label = 'Class 1')
        plt.scatter(tt2.index, x2, color = 'g', alpha = 0.5, label = 'Class 2')
        plt.scatter(tt2.index, x3, color = 'r', alpha = 0.5, label = 'Class 3')
        plt.scatter(tt2.index, x4, color = 'gray', alpha = 0.5, label = 'Class 4')
        plt.scatter(0, xm_f, s=100, color = 'black', alpha = 0.5, marker='*', label = 'Mixture')
        #plt.text(0, xm_f, f'({xm_f})', color='black')


        plt.xlabel('pc1')        
        

    plt.legend()
        
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
