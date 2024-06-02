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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
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


pca_1 = PCA(n_components=6) # 주성분을 몇개로 할지 결정
principalComponents_1 = pca_1.fit_transform(x1)

col_pc_1 = []
for i in range(1,7):
    col_pc_1.append("pc"+str(i))
    
principalDf_1 = pd.DataFrame(data=principalComponents_1, columns = col_pc_1, index=df1.index)


# 데이터 불러오기
tt = pd.read_csv('train_test_set_template_raw.csv', dtype={"Class":object})

# tt 파일에서 material 이름과 함량 추출하기
materials = tt.iloc[:, [0, 2, 4, 6, 8]].values
amounts = tt.iloc[:, [1, 3, 5, 7, 9]].astype(float).values

# tt 파일에서 Class 값 추출하기
classes = tt.iloc[:, -1].values

# 각 material에 대해 feature값과 함량을 곱한 뒤 더하기
features_1 = []
for i in range(len(materials)):
    feature_1 = np.zeros(6)
    for j in range(5):
        if pd.notnull(materials[i][j]):
            material_name = materials[i][j]
            amount = amounts[i][j]
            material_features_1 = principalDf_1.loc[principalDf_1.index == material_name].iloc[:, :].values
            if len(material_features_1) > 0:
                material_feature_1 = material_features_1[0]
                feature_1 += material_feature_1 * amount
            else:
                pass
            
    features_1.append(feature_1)

tt2_1 = pd.DataFrame(data = features_1, columns = col_pc_1)
tt2_1["Class"] = tt["Class"]




# feature와 target 나누기
X_1 = tt2_1.iloc[:, :-1]
y_1 = tt2_1.iloc[:, -1]

# train, test 데이터셋 나누기
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2)

rf_1 = RandomForestClassifier()
rf_1.fit(X_train_1, y_train_1)
rf_pred_1 = rf_1.predict(X_test_1)
rf_acc_1 = accuracy_score(y_test_1, rf_pred_1)



st.subheader(" ")
st.subheader('Formulation Design')

API_df = df[(df['Function']=='API')]
API_list = API_df.index.to_list()
API_name = st.selectbox('API', API_list, index = 34)

API_pc = principalDf_1.loc[principalDf_1.index == API_name].iloc[:, :].values
API_class = rf_1.predict(API_pc)
st.write("API only : Class "+API_class[0])
strength = st.number_input('API content (mg)', 1, 1000, value = 100)
tablet_wt = st.number_input('Tablet weight (mg)', strength, 1000, value = 200)
API_content = round(strength/tablet_wt*100, 1)
st.write("API content (%): "+str(API_content))
st.write("")


Filler_df =  df[df['Function'].isin(['Filler_DC', 'Filler_WG'])]
Filler_list = Filler_df.index.to_list()

Binder_df =  df[(df['Function']=='Binder')]
Binder_list = Binder_df.index.to_list()

Disintegrant_df =  df[(df['Function']=='Disintegrant')]
Disintegrant_list = Disintegrant_df.index.to_list()


#sample formulations
sample_f =st.radio(label = 'Select a preset formulation', options = ['DC Formulation 1', 'DC Formulation 2','DG Formulation','WG Formulation 1 (FBG)', 'WG Formulation 2 (HSG)'])
if sample_f == 'DC Formulation 1':   
    index_ex1 = 22
    value_ex1 = 0
    index_ex2 = 23
    index_ex3 = 2
    value_ex3 = 0
    index_ex4 = 1
    value_ex4 = 5
elif sample_f =='DC Formulation 2':
    index_ex1 = 4
    value_ex1 = 20
    index_ex2 = 33
    index_ex3 = 2
    value_ex3 = 0
    index_ex4 = 3
    value_ex4 = 3
elif sample_f =='DG Formulation':
    index_ex1 = 22
    value_ex1 = 0
    index_ex2 = 24
    index_ex3 = 8
    value_ex3 = 10
    index_ex4 = 0
    value_ex4 = 2
elif sample_f =='WG Formulation 1 (FBG)':
    index_ex1 = 22
    value_ex1 = 0
    index_ex2 = 22
    index_ex3 = 8
    value_ex3 = 3
    index_ex4 = 0
    value_ex4 = 5
elif sample_f =='WG Formulation 2 (HSG)':
    index_ex1 = 22
    value_ex1 = 0
    index_ex2 = 22
    index_ex3 = 2
    value_ex3 = 3
    index_ex4 = 2
    value_ex4 = 5

Excipient3_name = st.selectbox(
    'Binder',
    Binder_list, index = index_ex3)

Excipient3_content = st.number_input('Binder content (%)',0, 100, value = value_ex3 )
Excipient3_content_f = float(Excipient3_content)
st.write("")

Excipient4_name = st.selectbox(
    'Disintegrant',
    Disintegrant_list, index = index_ex4)

Excipient4_content = st.number_input('Disintegrant content (%)',0, 100, value = value_ex4)
Excipient4_content_f = float(Excipient4_content)
st.write("")

Excipient1_name = st.selectbox(
    'Filler 1',
    Filler_list, index = index_ex1)

Excipient1_content = st.number_input('Filler 1 content (%)',0, 100, value = value_ex1)
Excipient1_content_f = float(Excipient1_content)
st.write("")


Excipient2_name = st.selectbox(
    'Filler 2',
    Filler_list, index = index_ex2)

#Excipient2_content = st.number_input('Filler 2 content (%)',0, 100, value = value_ex3, label_visibility = "collapsed" )
Excipient2_content_f = 100 -API_content -Excipient1_content_f -Excipient3_content_f -Excipient4_content_f
Excipient2_content = str(Excipient2_content_f)
st.write("Filler 2 content (%): "+Excipient2_content)
if float(Excipient2_content) <0:
    st.write(":red[Error: Please adjust contents (Filler 2 content is negative)]")

st.write("")


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

explained_vraiance_ratio = pd.DataFrame(data= exp_vr, columns = ["Explained Variance Ratio"])
n_components = pd.DataFrame(data= comp, columns = ["n_comp"])
evr = pd.concat([explained_vraiance_ratio, n_components], axis = 1)
evr = evr.set_index('n_comp')


num_pc = st.number_input('Set the number of princial components', 1, 21, value = 6) 
pca = PCA(n_components=num_pc) # 주성분을 몇개로 할지 결정
principalComponents = pca.fit_transform(x1)

col_pc = []
for i in range(1,num_pc+1):
    col_pc.append("pc"+str(i))
    
principalDf = pd.DataFrame(data=principalComponents, columns = col_pc, index=df1.index)

with st.expander('n_comp vs Explained Variance Ratio'):
      st.write(evr)

#fig = plt.figure(constrained_layout=True, figsize=(6,4))

#plt.plot(comp, exp_vr, c = 'blue', linestyle = '-', marker = 'o', markersize = 5)
#plt.xlabel('Number of PC (Princial Component)')
#plt.ylabel('Explained Variance Ratio')
#plt.grid()
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot()

# 데이터 불러오기
tt = pd.read_csv('train_test_set_template_raw.csv', dtype={"Class":object})

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




# feature와 target 나누기
X = tt2.iloc[:, :-1]
y = tt2.iloc[:, -1]

st.write("")
rs = st.number_input('Set a seed for machine learning', 1)




st.subheader(" ")
## Buttons
if st.button("Predict"):
#    if select == 'Logistic Regression':
#        model = lr
#    elif select == 'Support Vector Machine':
#        model = svc
#    #elif select == 'Decision Tree':
#    #    model = dt
#    elif select == 'Random Forest':
#        model = rf
#    elif select == 'XGBoost':
#        model = xgb_model

#    model.fit(X_train, y_train)
#    model_pred = model.predict(X_test)
#    model_acc = round(accuracy_score(y_test, model_pred), 4)
    

    # train, test 데이터셋 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rs)

    models = ['Random Forest', 'Logistic Regression', 'Support Vector Machine','k-NN', 'LightGBM' ]  #'Decision Tree', 'XGBoost'
    #select = st.selectbox('Please select a model', models)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    
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

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)


    lgbm = LGBMClassifier(n_estimators=30)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    lgbm_acc = accuracy_score(y_test, lgbm_pred)

    #gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    #gbm.fit(X_train, y_train)
    #gbm_pred = gbm.predict(X_test)
    #gbm_acc = accuracy_score(y_test, gbm_pred)


    #xgb_model = xgb.XGBClassifier()
    #xgb_model.fit(X_train, y_train)
    #xgb_pred = xgb_model.predict(X_test)
    #xgb_acc = accuracy_score(y_test, xgb_pred)
    
    accuracies = [lr_acc, svc_acc, rf_acc, knn_acc, lgbm_acc ]  #dt_acc, xgb_acc

    
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
    #pred = model.predict(mixture_df)

    pred_lr = lr.predict(mixture_df)
    pred_svc = svc.predict(mixture_df)
    pred_rf = rf.predict(mixture_df)
    pred_knn = knn.predict(mixture_df)
    pred_lgbm = lgbm.predict(mixture_df)
    pred_all = [pred_lr, pred_svc, pred_rf, pred_knn, pred_lgbm ]
   
    result = pd.DataFrame({'Model': models, 'MCS Class': pred_all, 'Accuracy': accuracies})
    result = result.set_index('Model')
    
    st.write(result)
   
    #st.write("Predicted Manfacturing Class = " + str(pred[0]))
    #st.write("Model Accuracy : " + str(model_acc))
    st.write("Class 1 : Direct Compression")
    st.write("Class 2 : Dry Granulation")
    st.write("Class 3.1 : Wet Granulation (Fluid Bed Granulation)")
    st.write("Class 3.2 : Wet Granulation (Low Shear Granulation)")
    st.write("Class 3.3 : Wet Granulation (High Shear Granulation)")
    st.write("Class 3.4 : Wet Granulation (Melt Granulation)")
    st.write("Class 4 : Other Technology")
    if num_pc >= 3:
        fig = plt.figure(constrained_layout=True, figsize=(12,9))
        ax = fig.add_subplot(221, projection='3d')
        ax_xy = fig.add_subplot(222, projection='3d')
        ax_yz= fig.add_subplot(223, projection='3d')
        ax_zx = fig.add_subplot(224, projection='3d')
        
        x1 = tt2[tt2["Class"] == "1"]["pc1"]
        y1 = tt2[tt2["Class"] == "1"]["pc2"]
        z1 = tt2[tt2["Class"] == "1"]["pc3"]

        x2 = tt2[tt2["Class"] == "2"]["pc1"]
        y2 = tt2[tt2["Class"] == "2"]["pc2"]
        z2 = tt2[tt2["Class"] == "2"]["pc3"]

        x31 = tt2[tt2["Class"] == "3.1"]["pc1"]
        y31 = tt2[tt2["Class"] == "3.1"]["pc2"]
        z31 = tt2[tt2["Class"] == "3.1"]["pc3"]

        x32 = tt2[tt2["Class"] == "3.2"]["pc1"]
        y32 = tt2[tt2["Class"] == "3.2"]["pc2"]
        z32 = tt2[tt2["Class"] == "3.2"]["pc3"]

        x33 = tt2[tt2["Class"] == "3.3"]["pc1"]
        y33 = tt2[tt2["Class"] == "3.3"]["pc2"]
        z33 = tt2[tt2["Class"] == "3.3"]["pc3"]

        x34 = tt2[tt2["Class"] == "3.4"]["pc1"]
        y34 = tt2[tt2["Class"] == "3.4"]["pc2"]
        z34 = tt2[tt2["Class"] == "3.4"]["pc3"]        
        
        x4 = tt2[tt2["Class"] == "4"]["pc1"]
        y4 = tt2[tt2["Class"] == "4"]["pc2"]
        z4 = tt2[tt2["Class"] == "4"]["pc3"]

        x_api = principalDf.loc[API_name]["pc1"]
        y_api = principalDf.loc[API_name]["pc2"]
        z_api = principalDf.loc[API_name]["pc3"]

        x_ex1 = principalDf.loc[Excipient1_name]["pc1"]
        y_ex1 = principalDf.loc[Excipient1_name]["pc2"]
        z_ex1 = principalDf.loc[Excipient1_name]["pc3"]

        x_ex2 = principalDf.loc[Excipient2_name]["pc1"]
        y_ex2 = principalDf.loc[Excipient2_name]["pc2"]
        z_ex2 = principalDf.loc[Excipient2_name]["pc3"]        

        x_ex3 = principalDf.loc[Excipient3_name]["pc1"]
        y_ex3 = principalDf.loc[Excipient3_name]["pc2"]
        z_ex3 = principalDf.loc[Excipient3_name]["pc3"]      

        x_ex4 = principalDf.loc[Excipient4_name]["pc1"]
        y_ex4 = principalDf.loc[Excipient4_name]["pc2"]
        z_ex4 = principalDf.loc[Excipient4_name]["pc3"]            
        
        xm = mixture_df["pc1"]
        ym = mixture_df["pc2"]
        zm = mixture_df["pc3"]
        xm_f = round(float(xm), 1)
        ym_f = round(float(ym), 1)
        zm_f = round(float(zm), 1)

        al_ex1 = 1
        al_ex2 = 1
        al_ex3 = 1
        al_ex4 = 1
        if Excipient1_content == 0:
            al_ex1 = 0.1
        if Excipient2_content == 0:
            al_ex2 = 0.1
        if Excipient3_content == 0:
            al_ex3 = 0.1
        if Excipient4_content == 0:
            al_ex4 = 0.1

        
        ax.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax.scatter(x31, y31, z31, color = 'gold', alpha = 0.5, label = 'Class 3.1')
        ax.scatter(x32, y32, z32, color = 'orange', alpha = 0.5, label = 'Class 3.2')
        ax.scatter(x33, y33, z33, color = 'firebrick', alpha = 0.5, label = 'Class 3.3')
        ax.scatter(x34, y34, z34, color = 'red', alpha = 0.5, label = 'Class 3.4')
        ax.scatter(x4, y4, z4, color = 'magenta', alpha = 0.5, label = 'Class 4')
        ax.scatter(x_api, y_api, z_api, color = 'black', alpha = 1 ,  marker='^' )
        ax.scatter(x_ex1, y_ex1, z_ex1,color = 'black', alpha = al_ex1, marker='^')
        ax.scatter(x_ex2, y_ex2, z_ex2, color = 'black', alpha = al_ex2,marker='^' )
        ax.scatter(x_ex3, y_ex3, z_ex3, color = 'black', alpha = al_ex3, marker='^')
        ax.scatter(x_ex4, y_ex4, z_ex4,color = 'black', alpha = al_ex4,marker='^' )
        ax.scatter(xm, ym, zm , s=100, color = 'black', alpha = 1,  marker='*', label = 'Mixture')
        ax.text(x_api, y_api, z_api, 'API', color='black')
        ax.text(xm_f, ym_f, zm_f, f'Mixture({xm_f}, {ym_f}, {zm_f})', color='black')
        ax.text(x_ex1, y_ex1, z_ex1, 'Filler 1', alpha= al_ex1, color='black')
        ax.text(x_ex2, y_ex2, z_ex2, 'Filler 2', alpha= al_ex2, color='black')
        ax.text(x_ex3, y_ex3, z_ex3, 'Binder', alpha= al_ex3, color='black')
        ax.text(x_ex4, y_ex4, z_ex4, 'Disintegrant',alpha= al_ex4, color='black')


        ax_xy.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax_xy.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax_xy.scatter(x31, y31, z31, color = 'gold', alpha = 0.5, label = 'Class 3.1')
        ax_xy.scatter(x32, y32, z32, color = 'orange', alpha = 0.5, label = 'Class 3.2')
        ax_xy.scatter(x33, y33, z33, color = 'firebrick', alpha = 0.5, label = 'Class 3.3')
        ax_xy.scatter(x34, y34, z34, color = 'red', alpha = 0.5, label = 'Class 3.4')
        ax_xy.scatter(x4, y4, z4, color = 'magenta', alpha = 0.5, label = 'Class 4')
        ax_xy.scatter(x_api, y_api, z_api, color = 'black', alpha = 1, marker='^')
        ax_xy.scatter(x_ex1, y_ex1, z_ex1, color = 'black', alpha = al_ex1, marker='^')
        ax_xy.scatter(x_ex2, y_ex2, z_ex2, color = 'black', alpha = al_ex2, marker='^')
        ax_xy.scatter(x_ex3, y_ex3, z_ex3, color = 'black', alpha = al_ex3, marker='^')
        ax_xy.scatter(x_ex4, y_ex4, z_ex4, color = 'black', alpha = al_ex4, marker='^')
        ax_xy.scatter(xm, ym, zm , s=100, color = 'black', alpha = 1, marker='*', label = 'Mixture')
        ax_xy.text(xm_f, ym_f, zm_f, f'Mixture({xm_f}, {ym_f}, {zm_f})', color='black')
        ax_xy.text(x_api, y_api, z_api, 'API', color='black')
        ax_xy.text(x_ex1, y_ex1, z_ex1, 'Filler 1', alpha=al_ex1, color='black')
        ax_xy.text(x_ex2, y_ex2, z_ex2, 'Filler 2', alpha=al_ex2,color='black')
        ax_xy.text(x_ex3, y_ex3, z_ex3, 'Binder', alpha=al_ex3,color='black')
        ax_xy.text(x_ex4, y_ex4, z_ex4, 'Disintegrant', alpha=al_ex4,color='black')
        

        ax_yz.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax_yz.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax_yz.scatter(x31, y31, z31, color = 'gold', alpha = 0.5, label = 'Class 3.1')
        ax_yz.scatter(x32, y32, z32, color = 'orange', alpha = 0.5, label = 'Class 3.2')
        ax_yz.scatter(x33, y33, z33, color = 'firebrick', alpha = 0.5, label = 'Class 3.3')
        ax_yz.scatter(x34, y34, z34, color = 'red', alpha = 0.5, label = 'Class 3.4')
        ax_yz.scatter(x4, y4, z4, color = 'magenta', alpha = 0.5, label = 'Class 4')
        ax_yz.scatter(x_api, y_api, z_api, color = 'black', alpha = 1, marker='^')
        ax_yz.scatter(x_ex1, y_ex1, z_ex1, color = 'black', alpha = al_ex1, marker='^')
        ax_yz.scatter(x_ex2, y_ex2, z_ex2, color = 'black', alpha = al_ex2, marker='^')
        ax_yz.scatter(x_ex3, y_ex3, z_ex3, color = 'black', alpha = al_ex3, marker='^')
        ax_yz.scatter(x_ex4, y_ex4, z_ex4, color = 'black', alpha = al_ex4, marker='^')
        ax_yz.scatter(xm, ym, zm , s=100, color = 'black', alpha = 1, marker='*', label = 'Mixture')
        ax_yz.text(xm_f, ym_f, zm_f, f'Mixture({xm_f}, {ym_f}, {zm_f})', color='black')
        ax_yz.text(x_api, y_api, z_api, 'API', color='black')
        ax_yz.text(x_ex1, y_ex1, z_ex1, 'Filler 1', alpha=al_ex1,color='black')
        ax_yz.text(x_ex2, y_ex2, z_ex2, 'Filler 2', alpha=al_ex2,color='black')
        ax_yz.text(x_ex3, y_ex3, z_ex3, 'Binder',alpha=al_ex3, color='black')
        ax_yz.text(x_ex4, y_ex4, z_ex4, 'Disintegrant', alpha=al_ex4,color='black')

        ax_zx.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1')
        ax_zx.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2')
        ax_zx.scatter(x31, y31, z31, color = 'gold', alpha = 0.5, label = 'Class 3.1')
        ax_zx.scatter(x32, y32, z32, color = 'orange', alpha = 0.5, label = 'Class 3.2')
        ax_zx.scatter(x33, y33, z33, color = 'firebrick', alpha = 0.5, label = 'Class 3.3')
        ax_zx.scatter(x34, y34, z34, color = 'red', alpha = 0.5, label = 'Class 3.4')
        ax_zx.scatter(x4, y4, z4, color = 'magenta', alpha = 0.5, label = 'Class 4')
        ax_zx.scatter(x_api, y_api, z_api, color = 'black', alpha = 1, marker='^',label = 'Selected Material')
        ax_zx.scatter(x_ex1, y_ex1, z_ex1, color = 'black', alpha = al_ex1, marker='^')
        ax_zx.scatter(x_ex2, y_ex2, z_ex2, color = 'black', alpha = al_ex2, marker='^')
        ax_zx.scatter(x_ex3, y_ex3, z_ex3, color = 'black', alpha = al_ex3, marker='^')
        ax_zx.scatter(x_ex4, y_ex4, z_ex4, color = 'black', alpha = al_ex4, marker='^')
        ax_zx.scatter(xm, ym, zm , s=100, color = 'black', alpha = 1, marker='*', label = 'Mixture')
        ax_zx.text(x_api, y_api, z_api, 'API', color='black')
        ax_zx.text(xm_f, ym_f, zm_f, f'Mixture({xm_f}, {ym_f}, {zm_f})', color='black')
        ax_zx.text(x_ex1, y_ex1, z_ex1, 'Filler 1', alpha=al_ex1,color='black')
        ax_zx.text(x_ex2, y_ex2, z_ex2, 'Filler 2', alpha=al_ex2,color='black')
        ax_zx.text(x_ex3, y_ex3, z_ex3, 'Binder', alpha=al_ex3,color='black')
        ax_zx.text(x_ex4, y_ex4, z_ex4, 'Disintegrant', alpha=al_ex4,color='black')

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
                      
        x1 = tt2[tt2["Class"] == "1"]["pc1"]
        y1 = tt2[tt2["Class"] == "1"]["pc2"]
        
        x2 = tt2[tt2["Class"] == "2"]["pc1"]
        y2 = tt2[tt2["Class"] == "2"]["pc2"]
       
        x31 = tt2[tt2["Class"] == "3.1"]["pc1"]
        y31 = tt2[tt2["Class"] == "3a"]["pc2"]

        x32 = tt2[tt2["Class"] == "3.2"]["pc1"]
        y32 = tt2[tt2["Class"] == "3.2"]["pc2"]

        x33 = tt2[tt2["Class"] == "3.3"]["pc1"]
        y33 = tt2[tt2["Class"] == "3.3"]["pc2"]

        x34 = tt2[tt2["Class"] == "3.4"]["pc1"]
        y34 = tt2[tt2["Class"] == "3.4"]["pc2"]
       
        x4 = tt2[tt2["Class"] == 4]["pc1"]
        y4 = tt2[tt2["Class"] == 4]["pc2"]

        x_api = principalDf.loc[API_name]["pc1"]
        y_api = principalDf.loc[API_name]["pc2"]
        
        x_ex1 = principalDf.loc[Excipient1_name]["pc1"]
        y_ex1 = principalDf.loc[Excipient1_name]["pc2"]
        
        x_ex2 = principalDf.loc[Excipient2_name]["pc1"]
        y_ex2 = principalDf.loc[Excipient2_name]["pc2"]
        
        x_ex3 = principalDf.loc[Excipient3_name]["pc1"]
        y_ex3 = principalDf.loc[Excipient3_name]["pc2"]    

        x_ex4 = principalDf.loc[Excipient4_name]["pc1"]
        y_ex4 = principalDf.loc[Excipient4_name]["pc2"]        
        
        xm = mixture_df["pc1"]
        ym = mixture_df["pc2"]
       
        xm_f = round(float(xm), 1)
        ym_f = round(float(ym), 1)
        
        plt.scatter(x1, y1, color = 'b', alpha = 0.5, label = 'Class 1')
        plt.scatter(x2, y2, color = 'g', alpha = 0.5, label = 'Class 2')
        plt.scatter(x31, y31, z31, color = 'gold', alpha = 0.5, label = 'Class 3.1')
        plt.scatter(x32, y32, z32, color = 'orange', alpha = 0.5, label = 'Class 3.2')
        plt.scatter(x33, y33, z33, color = 'firebrick', alpha = 0.5, label = 'Class 3.3')
        plt.scatter(x34, y34, z34, color = 'red', alpha = 0.5, label = 'Class 3.4')
        plt.scatter(x4, y4, z4, color = 'magenta', alpha = 0.5, label = 'Class 4')
        plt.scatter(x_api, y_api, color = 'black', marker='^', alpha = 1)
        plt.scatter(x_ex1, y_ex1, color = 'black',  marker='^', alpha = al_ex1)
        plt.scatter(x_ex2, y_ex2, color = 'black',  marker='^', alpha = al_ex2)
        plt.scatter(x_ex3, y_ex3, color = 'black', marker='^',  alpha = al_ex3)
        plt.scatter(x_ex4, y_ex4, color = 'black', marker='^',  alpha = al_ex4)
        plt.scatter(xm, ym, s=100, color = 'black', alpha = 1, marker='*', label = 'Mixture')
        plt.text(xm_f, ym_f, f'Mixture({xm_f}, {ym_f})', color='black')
        plt.text(x_ex1, y_ex1, 'Filler 1', alpha=al_ex1,color='black')
        plt.text(x_ex2, y_ex2, 'Filler 2', alpha=al_ex2,color='black')
        plt.text(x_ex3, y_ex3, 'Binder', alpha=al_ex3,color='black')
        plt.text(x_ex4, y_ex4, 'Disintegrant', alpha=al_ex4,color='black')


        plt.xlabel('pc1')
        plt.ylabel('pc2')
        
   
    plt.legend(loc='best', bbox_to_anchor=(1.0,0.5))
        
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
