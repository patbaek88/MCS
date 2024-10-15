import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import axes3d

password_input = st.text_input("Please enter the password",type= "password")
password = "cmc"

if password_input == password:
    st.header('Manufacturing Classification System')  # 타이틀명 지정
    #database 불러오기 및 전처리
    filename = 'FT4_Database.csv'
    df_origin = pd.read_csv(filename)
    df = df_origin
    
    if st.checkbox('Add new data'):
        st.write("Current Database")
        st.dataframe(df_origin.set_index('Material'))
        st.write("New data")
        df_new = pd.DataFrame(columns = df_origin.columns)
        df_new = st.data_editor(df_new, num_rows="dynamic", hide_index=None)
        df = pd.concat([df_new, df_origin], ignore_index=True)
        if df_new.isnull().values.any():
            st.write(":red[Please fill all 'None' or uncheck 'Add new data']")
        else:
            df_csv = df.to_csv(index = False)
            st.download_button(label="Download New Database", data= df_csv, file_name = 'FT4_Database_New.csv', mime="text/csv")
            st.write("")
    
    df = df.set_index('Material')
    df1=df.iloc[:, 1:] #필요한 데이터만 필터
    x1 = df1.values  # 독립변인들의 value값만 추출
    x1 = StandardScaler().fit_transform(x1)  # x객체에 x를 표준화한 데이터를 저장
   
    n_components = 6
    pca_1 = PCA(n_components=n_components) # 주성분을 몇개로 할지 결정
    principalComponents_1 = pca_1.fit_transform(x1)
    
    col_pc_1 = []
    for i in range(1,n_components+1):
        col_pc_1.append("pc"+str(i))
        
    principalDf_1 = pd.DataFrame(data=principalComponents_1, columns = col_pc_1, index=df1.index)
    
    
    # 데이터 불러오기
    tt = pd.read_csv('train_test_set_template_final.csv', dtype={"Class":object})
    
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
    
    gbt_1 = GradientBoostingClassifier()
    gbt_1.fit(X_train_1, y_train_1)
    gbt_pred_1 = gbt_1.predict(X_test_1)
    gbt_acc_1 = accuracy_score(y_test_1, gbt_pred_1)

    
    #st.subheader(" ")
    st.subheader('Formulation Design')
    API_df = df[(df['Function']=='API')] #df에서 API만 필터
    API_list = API_df.index.to_list()
    API_name = st.selectbox('**API**', API_list) #API 선택메뉴
    API_pc = principalDf_1.loc[principalDf_1.index == API_name].iloc[:, :].values
    API_class = gbt_1.predict(API_pc)
    pred_prob_1 = gbt_1.predict_proba(API_pc)
    pred_prob_max_1  = float(pred_prob_1.max(axis=1))
    st.write("API only : Class "+API_class[0]+" (Model: Gradient Boosted Trees, Probability: "+str(pred_prob_max_1)+")")
    strength = st.number_input('API content (mg)', 1, 1000, value = 100)
    tablet_wt = st.number_input('Tablet weight (mg)', strength, 1000, value = 200)
    API_content = round(strength/tablet_wt*100, 1)
    st.write("API content (%): "+str(API_content))
    st.write("")
    
    Filler_df =  df[df['Function'].isin(['Filler_DC', 'Filler_WG', 'Filler'])] #df에서 Filler만 필터
    Filler_list = Filler_df.index.to_list()
    #Binder_df =  df[(df['Function']=='Binder')] #df에서 Binder만 필터
    #Binder_list = Binder_df.index.to_list()
    Disintegrant_df =  df[(df['Function']=='Disintegrant')] #df에서 Disintegrant만 필터
    Disintegrant_list = Disintegrant_df.index.to_list()
    All_exp_df = df[df['Function'] != 'API']
    All_exp_list = All_exp_df.index.to_list()
    
    Disintegrant_name = st.selectbox(
        '**Disintegrant**',
        Disintegrant_list)
    
    Disintegrant_content = st.number_input('Disintegrant content (%)',0, 100, value = 5)
    Disintegrant_content_f = float(Disitegrant_content)
    st.write("") 


    API_content_f = float(API_content)


    # Filler 조합과 함량을 생성하는 함수
    filler_combinations = []
    for combination in itertools.combinations_with_replacement(Filler_list, 2):
        for Filler1_content in range(0, 101 - API_content - Disintegrant_content_f):
            Filler2_content = 100 - API_content - Disintegrant_content_f - f1_amount
            if Filler2_content >= 0:
                filler_combinations.append((combination, Filler1_content, combination, Filler2_content))


    data = pd.DataFrame(combinations, columns=['Filler1', 'Filler1_Amount', 'Filler2', 'Filler2_Amount'])
    
    # 범주형 데이터를 수치형으로 변환
    data = pd.get_dummies(data, columns=['Filler1', 'Filler2'])
    
    # RandomForest 모델을 사용하여 확률 예측
    probabilities = gbt_1.predict_proba(data)[:, 1]
    
    # 상위 5% 조합 선택
    top_5_percent_index = np.argsort(probabilities)[-int(0.05 * len(probabilities)):]
    top_combinations = data.iloc[top_5_percent_index]



    mixture_data = principalDf.loc[API_name]*API_content_f/100 + principalDf.loc[Excipient1_name]*Excipient1_content_f/100 + principalDf.loc[Excipient2_name]*Excipient2_content_f/100 + principalDf.loc[Excipient3_name]*Excipient3_content_f/100 + principalDf.loc[Excipient4_name]*Excipient4_content_f/100
    mixture_df = mixture_data.to_frame()
    mixture_df = mixture_df.transpose()	#행 열 전환


    material_names = [API_name, Filler1_name, Filler2_name, Disintegrant_name]
    material_content_p = [API_content, Excipient2_content, Excipient1_content, Excipient3_content, Excipient4_content]
    material_content_mg = [strength, round(tablet_wt*Excipient2_content_f/100, 1), round(tablet_wt*Excipient1_content_f/100, 1), round(tablet_wt*Excipient3_content_f/100, 1), round(tablet_wt*Excipient4_content_f/100, 1) ]
    material_function = ["Drug Substance", "Filler", " ", "Binder", "Disintegrant"]
    formulation_df = pd.DataFrame({'Component': material_names, 'Function': material_function, 'Amount (mg/tablet)': material_content_mg, 'Amount (%/tablet)': material_content_p})
    formulation_df_cleaned = formulation_df[~(formulation_df ==0).any(axis=1)]
    st.write("")
    st.write("**[Components and Composition]**")
    st.dataframe(formulation_df_cleaned.set_index('Component'))
    st.write('Total tablet weight : '+str(tablet_wt)+' mg')
    st.subheader(" ")
    
    st.sidebar.title('Analytical Condition')
    n= x1.shape[1] #Feature 개수
    exp_vr = []
    comp = []
    for i in range(1,n+1): #1부터 feature개수까지 주성분개수 늘려가며 Explained Variance Ratio 확인하기
        pca = PCA(n_components=i)
        pca.fit_transform(x1)
        exp_vr.append(sum(pca.explained_variance_ratio_))
        comp.append(str(i))
    explained_vraiance_ratio = pd.DataFrame(data= exp_vr, columns = ["Explained Variance Ratio"])
    n_components = pd.DataFrame(data= comp, columns = ["n_comp"])
    evr = pd.concat([explained_vraiance_ratio, n_components], axis = 1)
    evr = evr.set_index('n_comp')

    st.sidebar.write("")
    num_pc = st.sidebar.number_input('Number of principal components', 1, 21, value = 6) #분석할 주성분 수 입력받기
    pca = PCA(n_components=num_pc)
    principalComponents = pca.fit_transform(x1)
    
    col_pc = []  #주성분 개수에 따라 컬럼명 생성하기
    for i in range(1,num_pc+1):
        col_pc.append("pc"+str(i))
    principalDf = pd.DataFrame(data=principalComponents, columns = col_pc, index=df1.index)

    # 주성분 로딩 계산 (특징의 상관성)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # 로딩값을 데이터프레임으로 변환
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(num_pc)], index=df1.columns)
    
    # 각 주성분(PC)과 가장 상관성이 높은 특징을 찾기
    #top_features = pd.DataFrame()
    #for i in range(loadings_df.shape[1]):
    #    pc = loadings_df.iloc[:, i]
    #    sorted_pc = pc.abs().sort_values(ascending=False)
    #    top_features[f'PC{i+1}'] = sorted_pc.index
    
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

    pc_meaning_df = pd.read_csv("pc_meaning.csv")
    pc_meaning_df = pc_meaning_df.set_index('PC')
    
    with st.sidebar.expander('n_comp vs Explained Variance Ratio'):
          st.write(evr)

    with st.sidebar.expander('Interpretation of Principal Components'):
        #st.write(loadings_df)
        st.write("Significant Loadings DataFrame")
        st.write(significant_loadings_df)
        st.write('Features')
        st.write(pc_meaning_df)
        st.write("PC1: compress(+), cohesion(+), flow(-)")
        st.write("PC2: cohesion(+), flow(-), compr. strength(+)")
        st.write("PC3: air sensitivity(+)")
        st.write("PC4: air permeability(-)")
        st.write("PC5: stability index")
        st.write("PC6: flow(+)")

    with st.sidebar.expander("Materials' PC values"):
        principalDf_add_f = pd.concat([df['Function'], principalDf], axis = 1) 
        filter_options = ["API", "Filler", "Filler_DC", "Filler_WG", "Binder", "Disintegrant"]
        material_filter = st.multiselect("Filter Option", filter_options, default=["API", "Filler", "Filler_DC", "Filler_WG", "Binder", "Disintegrant"])
        filtered_principalDf = principalDf_add_f[principalDf_add_f['Function'].isin(material_filter)]
        st.write(filtered_principalDf)
        


    # Formulation과 Class 정보 데이터 불러오기
    fci = pd.read_csv('train_test_set_template_final.csv', dtype={"Class":object})
    # fci 파일에서 material 이름과 함량 추출하기
    materials = fci.iloc[:, [0, 2, 4, 6, 8]].values
    amounts = fci.iloc[:, [1, 3, 5, 7, 9]].astype(float).values
    # fci 파일에서 Class 값 추출하기
    classes = fci.iloc[:, -1].values
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
    
    tt = pd.DataFrame(data = features, columns = col_pc) #train_test용 데이터프레임 만들기
    tt["Class"] = fci["Class"]
           
    # feature와 target 나누기
    X = tt.iloc[:, :-1]
    y = tt.iloc[:, -1]


    #다중 클래스 레이블을 바이너리 형식으로 변환 (ROC AUC 계산을 위해)
    y_bin = label_binarize(y, classes = np.unique(y))
        
    st.sidebar.write("")
    attempts = st.sidebar.number_input('Number of prediction attempts', 1, 100, value = 3)

    st.sidebar.write("")
    if num_pc >= 6:
        axis_options = ["pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]
    else:
        axis_options = [f"pc{i}" for i in range(1, num_pc+1)]
    
    if num_pc == 2:
        selected_axis = ["pc1", "pc2"]
    elif num_pc == 3:
        selected_axis = st.sidebar.multiselect(
            "Select 2 or 3 axis to plot", axis_options, default=["pc1", "pc2", "pc3"], key="axis"
        )
    elif num_pc >= 4:
        selected_axis = st.sidebar.multiselect(
            "Select 2 or 3 axis to plot", axis_options, default=["pc1", "pc3", "pc4"], key="axis"
        )
        
    
    mapping = {"pc1": "PC1: compress(+), cohesion(+), flow(-)", "pc2": "PC2: cohesion(+), flow(-), compr. strength(+)", "pc3": "PC3: air sensitivity(+)", "pc4": "PC4: air permeability(-)", "pc5": "PC5: stability index", "pc6": "PC6: flow(+)"}
    
    #st.subheader(" ")
    ## Buttons
    if st.button("Predict"):
        API_content_f = float(API_content)
        Excipient1_content_f = float(Excipient1_content)
        Excipient2_content_f = float(Excipient2_content)
        Excipient3_content_f = float(Excipient3_content)
        Excipient4_content_f = float(Excipient4_content)
        mixture_data = principalDf.loc[API_name]*API_content_f/100 + principalDf.loc[Excipient1_name]*Excipient1_content_f/100 + principalDf.loc[Excipient2_name]*Excipient2_content_f/100 + principalDf.loc[Excipient3_name]*Excipient3_content_f/100 + principalDf.loc[Excipient4_name]*Excipient4_content_f/100
        mixture_df = mixture_data.to_frame()
        mixture_df = mixture_df.transpose()	#행 열 전환    
            



  
                        

        
                  


                
                
               
       
elif password_input == "":
    st.write(" ")
else:
    st.write(":red[Incorrect password. Please try again.]")
