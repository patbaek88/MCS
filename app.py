import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
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

n_components = 6
pca_1 = PCA(n_components=n_components) # 주성분을 몇개로 할지 결정
principalComponents_1 = pca_1.fit_transform(x1)

col_pc_1 = []
for i in range(1,n_components+1):
    col_pc_1.append("pc"+str(i))
    
principalDf_1 = pd.DataFrame(data=principalComponents_1, columns = col_pc_1, index=df1.index)


# 데이터 불러오기
tt = pd.read_csv('train_test_set_template_generated2.csv', dtype={"Class":object})

# tt 파일에서 material 이름과 함량 추출하기
materials = tt.iloc[:, [0, 2, 4, 6, 8]].values
amounts = tt.iloc[:, [1, 3, 5, 7, 9]].astype(float).values

# tt 파일에서 Class 값 추출하기
classes = tt.iloc[:, -1].values

# 각 material에 대해 feature값과 함량을 곱한 뒤 더하기
features_1 = []
for i in range(len(materials)):
    feature_1 = np.zeros(n_components)
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

#lgbm_1 = LGBMClassifier(n_estimators=30)
#lgbm_1.fit(X_train_1, y_train_1)
#lgbm_pred_1 = lgbm_1.predict(X_test_1)
#lgbm_acc_1 = accuracy_score(y_test_1, lgbm_pred_1)


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


num_pc = st.number_input('Number of princial components', 1, 21, value = 6) 
pca = PCA(n_components=num_pc) # 주성분을 몇개로 할지 결정
principalComponents = pca.fit_transform(x1)

col_pc = []
for i in range(1,num_pc+1):
    col_pc.append("pc"+str(i))
    
principalDf = pd.DataFrame(data=principalComponents, columns = col_pc, index=df1.index)


# 주성분 로딩 계산 (특징의 상관성)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# 로딩값을 데이터프레임으로 변환
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(num_pc)], index=df1.columns)

# 각 주성분(PC)과 가장 상관성이 높은 특징을 찾기
top_features = pd.DataFrame()
for i in range(loadings_df.shape[1]):
    pc = loadings_df.iloc[:, i]
    sorted_pc = pc.abs().sort_values(ascending=False)
    top_features[f'PC{i+1}'] = sorted_pc.index

# 각 주성분에 대해 로딩값이 0.5 이상인 특징과 해당 로딩값을 데이터프레임으로 정리
significant_loadings_df = pd.DataFrame()
for i in range(loadings_df.shape[1]):
    pc = loadings_df.iloc[:, i]
    significant_loadings = pc[pc.abs() >= 0.5]
    significant_loadings = significant_loadings.sort_values(key=abs, ascending=False)
    
    pc_df = pd.DataFrame(significant_loadings)
    pc_df.columns = [f'PC{i+1}']
    
    if significant_loadings_df.empty:
        significant_loadings_df = pc_df
    else:
        significant_loadings_df = pd.concat([significant_loadings_df, pc_df], axis=1)


ft4_features = 'FT4_features.csv'
df_ft4_features = pd.read_csv(ft4_features)
# 결과 출력


fig = plt.figure(constrained_layout=True, figsize=(12,9))
ax1 = fig.add_subplot(221, projection='3d')
ax1_xy = fig.add_subplot(222, projection='3d')
ax1_yz= fig.add_subplot(223, projection='3d')
ax1_zx = fig.add_subplot(224, projection='3d')

function_df = pd.DataFrame(data = df["Function"])
principalDf_2 = pd.concat([principalDf, function_df], axis = 1) 

x_apis = principalDf_2[principalDf_2["Function"] == "API"]["pc1"]
y_apis = principalDf_2[principalDf_2["Function"] == "API"]["pc2"]
z_apis = principalDf_2[principalDf_2["Function"] == "API"]["pc3"]

x_dc = principalDf_2[principalDf_2["Function"] == "Filler_DC"]["pc1"]
y_dc = principalDf_2[principalDf_2["Function"] == "Filler_DC"]["pc2"]
z_dc = principalDf_2[principalDf_2["Function"] == "Filler_DC"]["pc3"]

x_wg = principalDf_2[principalDf_2["Function"] == "Filler_WG"]["pc1"]
y_wg = principalDf_2[principalDf_2["Function"] == "Filler_WG"]["pc2"]
z_wg = principalDf_2[principalDf_2["Function"] == "Filler_WG"]["pc3"]

x_binder = principalDf_2[principalDf_2["Function"] == "Binder"]["pc1"]
y_binder = principalDf_2[principalDf_2["Function"] == "Binder"]["pc2"]
z_binder = principalDf_2[principalDf_2["Function"] == "Binder"]["pc3"]

x_disint = principalDf_2[principalDf_2["Function"] == "Disintegrant"]["pc1"]
y_disint = principalDf_2[principalDf_2["Function"] == "Disintegrant"]["pc2"]
z_disint = principalDf_2[principalDf_2["Function"] == "Disintegrant"]["pc3"]


ax1.scatter(x_apis, y_apis, z_apis, color = 'gray', alpha = 0.5, label = 'API')
ax1.scatter(x_dc, y_dc, z_dc, color = 'b', alpha = 0.5, label = 'Filler_DC')
ax1.scatter(x_wg, y_wg, z_wg, color = 'r', alpha = 0.5, label = 'Filler_WC')
ax1.scatter(x_binder, y_binder, z_binder, color = 'yellow', alpha = 0.5, label = 'Binder')
ax1.scatter(x_disint, y_disint, z_disint, color = 'orange', alpha = 0.5, label = 'Disintegrant')

ax1_xy.scatter(x_apis, y_apis, z_apis, color = 'gray', alpha = 0.5, label = 'API')
ax1_xy.scatter(x_dc, y_dc, z_dc, color = 'b', alpha = 0.5, label = 'Filler_DC')
ax1_xy.scatter(x_wg, y_wg, z_wg, color = 'r', alpha = 0.5, label = 'Filler_WC')
ax1_xy.scatter(x_binder, y_binder, z_binder, color = 'yellow', alpha = 0.5, label = 'Binder')
ax1_xy.scatter(x_disint, y_disint, z_disint, color = 'orange', alpha = 0.5, label = 'Disintegrant')

ax1_yz.scatter(x_apis, y_apis, z_apis, color = 'gray', alpha = 0.5, label = 'API')
ax1_yz.scatter(x_dc, y_dc, z_dc, color = 'b', alpha = 0.5, label = 'Filler_DC')
ax1_yz.scatter(x_wg, y_wg, z_wg, color = 'r', alpha = 0.5, label = 'Filler_WC')
ax1_yz.scatter(x_binder, y_binder, z_binder, color = 'yellow', alpha = 0.5, label = 'Binder')
ax1_yz.scatter(x_disint, y_disint, z_disint, color = 'orange', alpha = 0.5, label = 'Disintegrant')

ax1_zx.scatter(x_apis, y_apis, z_apis, color = 'gray', alpha = 0.5, label = 'API')
ax1_zx.scatter(x_dc, y_dc, z_dc, color = 'b', alpha = 0.5, label = 'Filler_DC')
ax1_zx.scatter(x_wg, y_wg, z_wg, color = 'r', alpha = 0.5, label = 'Filler_WC')
ax1_zx.scatter(x_binder, y_binder, z_binder, color = 'yellow', alpha = 0.5, label = 'Binder')
ax1_zx.scatter(x_disint, y_disint, z_disint, color = 'orange', alpha = 0.5, label = 'Disintegrant')

ax1.set_xlabel('PC1: compress(+), cohesion(+), flow(-)')
ax1.set_ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
ax1.set_zlabel('PC3: air sensitivity(+)')
#ax1.view_init(20,60)
        
ax1_xy.set_xlabel('PC1: compress(+), cohesion(+), flow(-)')
ax1_xy.set_ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
ax1_xy.set_zlabel('PC3')
ax1_xy.view_init(90, 270)

ax1_yz.set_xlabel('PC1')
ax1_yz.set_ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
ax1_yz.set_zlabel('PC3: air sensitivity(+)')
ax1_yz.view_init(0,0)

ax1_zx.set_xlabel('PC1: compress(+), cohesion(+), flow(-)')
ax1_zx.set_ylabel('PC2')
ax1_zx.set_zlabel('PC3: air sensitivity(+)')
ax1_zx.view_init(0,-90)



pc_meaning_df = pd.read_csv("pc_meaning.csv")
pc_meaning_df = pc_meaning_df.set_index('PC')

with st.expander('n_comp vs Explained Variance Ratio'):
      st.write(evr)

# 결과 출력
with st.expander('Interpretation of Principal Components'):
      #st.write(loadings_df)
      st.write("Significant Loadings DataFrame")
      st.write(significant_loadings_df)
      st.write('Features')
      #st.write(df_ft4_features)
      st.write(pc_meaning_df)
      st.write("PC1: compress(+), cohesion(+), flow(-)")
      st.write("PC2: cohesion(+), flow(-), compr. strength(+)")
      st.write("PC3: air sensitivity(+)")
      st.write("")
      st.write('3D Plot of Raw Materials PCA')
      plt.legend(loc='best', bbox_to_anchor=(1.0,0.75))
      plt.show()
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()
      


#fig = plt.figure(constrained_layout=True, figsize=(6,4))

#plt.plot(comp, exp_vr, c = 'blue', linestyle = '-', marker = 'o', markersize = 5)
#plt.xlabel('Number of PC (Princial Component)')
#plt.ylabel('Explained Variance Ratio')
#plt.grid()
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot()

# 데이터 불러오기
tt = pd.read_csv('train_test_set_template_generated2.csv', dtype={"Class":object})

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

#다중 클래스 레이블을 바이너리 형식으로 변환 (ROC AUC 계산을 위해)
y_bin = label_binarize(y, classes = np.unique(y))

#st.write("")
#rs = st.number_input('Set a seed for machine learning', 1)

st.write("")
attempts = st.number_input('Number of prediction attempts', 1, 100, value = 30)


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
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #random_state = rs

    #models = ['Random Forest', 'Logistic Regression', 'Support Vector Machine','k-NN', 'LightGBM' ]  #'Decision Tree', 'XGBoost'
    #select = st.selectbox('Please select a model', models)

    #lr = LogisticRegression()
    #lr.fit(X_train, y_train)
    #lr_pred = lr.predict(X_test)
    #lr_acc = accuracy_score(y_test, lr_pred)
    
    #svc = SVC()
    #svc.fit(X_train, y_train)
    #svc_pred = svc.predict(X_test)
    #svc_acc = accuracy_score(y_test, svc_pred)

    #dt = DecisionTreeClassifier()
    #dt.fit(X_train, y_train)
    #dt_pred = dt.predict(X_test)
    #dt_acc = accuracy_score(y_test, dt_pred)

    #rf = RandomForestClassifier()
    #rf.fit(X_train, y_train)
    #rf_pred = rf.predict(X_test)
    #rf_acc = accuracy_score(y_test, rf_pred)

    #knn = KNeighborsClassifier(n_neighbors=5)
    #knn.fit(X_train, y_train)
    #knn_pred = knn.predict(X_test)
    #knn_acc = accuracy_score(y_test, knn_pred)


    #lgbm = LGBMClassifier(n_estimators=30)
    #lgbm.fit(X_train, y_train)
    #lgbm_pred = lgbm.predict(X_test)
    #lgbm_acc = accuracy_score(y_test, lgbm_pred)

    #gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    #gbm.fit(X_train, y_train)
    #gbm_pred = gbm.predict(X_test)
    #gbm_acc = accuracy_score(y_test, gbm_pred)


    #xgb_model = xgb.XGBClassifier()
    #xgb_model.fit(X_train, y_train)
    #xgb_pred = xgb_model.predict(X_test)
    #xgb_acc = accuracy_score(y_test, xgb_pred)
    
    #accuracies = [lr_acc, svc_acc, rf_acc, knn_acc, lgbm_acc ]  #dt_acc, xgb_acc

    
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

    # 모델을 정의합니다.
    models = {
        'Logistic Regression': LogisticRegression(multi_class='ovr'),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'LightGBM': LGBMClassifier(n_estimators=30),
        'SGD': SGDClassifier(loss='log_loss', class_weight = 'balanced')
    }

    results = []

    # 각 모델에 대해
    for model_name, model in models.items():
        predictions = []
        accuracies = []
        precisions = []
        recalls = []
        f1scores = []
        rocaucs = []
        loglosses = []
        confusionmatrixes = []
    
        # 시드를 1부터 attempts까지 변경하면서
        for seed in range(1, attempts+1):
            np.random.seed(seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = seed)
            y_test_bin = label_binarize(y_test, classes=np.unique(y))
        
            # 모델 학습 및 예측
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test) if hasattr(model, "predict_proba") else None
            y_pred_proba = model.predict_proba(X_test)
            pred = model.predict(mixture_df)
        
            # 예측 결과 저장
            predictions.append(pred)
        
            # 정확도 계산 및 저장
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            precision = precision_score(y_test, y_pred, average='macro')
            precisions.append(precision)
            recall = recall_score(y_test, y_pred, average='macro')
            recalls.append(recall)
            f1score = f1_score(y_test, y_pred, average='macro')
            f1scores.append(f1score)
            rocauc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
            #if y_pred_proba is not None:
            #    if len(np.unique(y_test)) > 2:
            #        rocauc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            #    else:
            #        rocauc = roc_auc_score(y_test, y_pred_proba[:, 1])
            #rocaucs.append(rocauc)
            confusionmatrix = confusion_matrix(y_test, y_pred)
            confusionmatrixes.append(confusionmatrix)
    
        # 최빈값 계산
        predictions = np.array(predictions)
        most_common_predictions = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=0, arr=predictions)
    
        # 정확도의 평균 계산
        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1score = np.mean(f1scores)
        mean_rocauc = np.mean(rocauc)
        #mean_confusiommatrix = np.mean(confusionmatrixes)
    
        # 결과 저장
        results.append({
            'Model': model_name,
            'MCS Class': most_common_predictions,
            'Mean Accuracy': mean_accuracy,
            'Mean Precision': mean_precision,
            'Mean Recall': mean_recall,
            'Mean F1 Score': mean_f1score,
            'Mean ROC AUC': mean_rocauc
        })

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)
    #results_df = results_df.sort_values(by=["Mean Accuracy"], ascending=[False])
    results_df = results_df.set_index('Model')



    

    #pred_lr = lr.predict(mixture_df)
    #pred_svc = svc.predict(mixture_df)
    #pred_rf = rf.predict(mixture_df)
    #pred_knn = knn.predict(mixture_df)
    #pred_lgbm = lgbm.predict(mixture_df)
    #pred_all = [pred_lr, pred_svc, pred_rf, pred_knn, pred_lgbm ]
   
    #result = pd.DataFrame({'Model': models, 'MCS Class': pred_all, 'Accuracy': accuracies})
    #result = result.set_index('Model')
    
    #st.write(result)
    st.write("Prediction Results (attempts=" + str(attempts) +")")
    st.write(results_df)
   
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


        ax.scatter(x1, y1, z1, color = 'b', alpha = 0.5, label = 'Class 1', s= 5)
        ax.scatter(x2, y2, z2, color = 'g', alpha = 0.5, label = 'Class 2', s= 5)
        ax.scatter(x31, y31, z31, color = 'gold', alpha = 0.5, label = 'Class 3.1', s= 5)
        ax.scatter(x32, y32, z32, color = 'orange', alpha = 0.5, label = 'Class 3.2', s= 5)
        ax.scatter(x33, y33, z33, color = 'firebrick', alpha = 0.5, label = 'Class 3.3', s= 5)
        ax.scatter(x34, y34, z34, color = 'red', alpha = 0.5, label = 'Class 3.4', s= 5)
        ax.scatter(x4, y4, z4, color = 'magenta', alpha = 0.5, label = 'Class 4', s= 5)
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


        ax.set_xlabel('PC1: compress(+), cohesion(+), flow(-)')
        ax.set_ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
        ax.set_zlabel('PC3: air sensitivity(+)')
        #ax.view_init(20,60)
        
        ax_xy.set_xlabel('PC1: compress(+), cohesion(+), flow(-)')
        ax_xy.set_ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
        ax_xy.set_zlabel('PC3')
        ax_xy.view_init(90,270)

        ax_yz.set_xlabel('PC1')
        ax_yz.set_ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
        ax_yz.set_zlabel('PC3: air sensitivity(+)')
        ax_yz.view_init(0,0)

        ax_zx.set_xlabel('PC1: compress(+), cohesion(+), flow(-)')
        ax_zx.set_ylabel('PC2')
        ax_zx.set_zlabel('PC3: air sensitivity(+)')
        ax_zx.view_init(0,-90)
        


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


        plt.xlabel('PC1: compress(+), cohesion(+), flow(-)')
        plt.ylabel('PC2: cohesion(+), flow(-), compr. strength(+)')
        
   
    plt.legend(loc='best', bbox_to_anchor=(1.0,0.75))
        
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
