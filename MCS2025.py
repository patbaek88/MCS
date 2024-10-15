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
  
    Disintegrant_df =  df[(df['Function']=='Disintegrant')] #df에서 Disintegrant만 필터
    Disintegrant_list = Disintegrant_df.index.to_list()
    
    
    Disintegrant_name = st.selectbox(
        '**Disintegrant**',
        Disintegrant_list)
    
    Disintegrant_content = st.number_input('Disintegrant content (%)',0, 100, value = 5)
    Disintegrant_content_f = float(Disitegrant_content)
    st.write("") 


    API_content_f = float(API_content)


    
    



   
        
  
    #st.subheader(" ")
    ## Buttons
    if st.button("Optimize"):
        
        filler_list = [f'Filler_list{i}' for i in range(1, 37)]
        filler_combinations = []
        remaining_amount = 100 - API_content_f - Disintegrant_content_f

        for i in range(0, 36):
            Filler1_name = Filler_list{i}
            Filler1_content = remaining_amount
               
            mixture_data = principalDf.loc[API_name]*API_content_f/100 + principalDf.loc[Filler1_name]*Filler1_content_f/100 + principalDf.loc[Disintegrant_name]*Disintegrant_content_f/100
            mixture_df = mixture_data.to_frame()
            mixture_df = mixture_df.transpose()	#행 열 전환

            mixture_class = gbt_1.predict(mixture_df)
            if mixture_Class == 1
                pred_prob_1 = gbt_1.predict_proba(mixture_df)
                pred_prob_max_1  = float(pred_prob_1.max(axis=1))
            


    


  
                        
    # RandomForest 모델을 사용하여 확률 예측
    probabilities = gbt_1.predict_proba(data)[:, 1]
    
    # 상위 5% 조합 선택
    top_5_percent_index = np.argsort(probabilities)[-int(0.05 * len(probabilities)):]
    top_combinations = data.iloc[top_5_percent_index]
        
                  


                
                
               
       
elif password_input == "":
    st.write(" ")
else:
    st.write(":red[Incorrect password. Please try again.]")
