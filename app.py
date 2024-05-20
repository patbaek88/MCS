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


# In[24]:


#표준화 변환 및 csv로 변환 (MCS_dataset_std_240503_FT4_rawdata.csv 있으면 생략가능)

#loading database and 전처리
filename = 'D:\#.Secure Work Folder\역량강화_MCS\Database\FT4_DB_Sep2023+sorbitol_mean1.csv'

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 881030)

#best model 자동 선택

best_accuracy = 0

# 다수의 머신러닝 모델 비교
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier()]
for model in models:
    # 모델 학습
    model.fit(X_train, y_train)

    # 모델 예측
    y_pred = model.predict(X_test)
    # 모델 성능 평가
    accuracy = accuracy_score(y_test, y_pred)

    # 가장 성능이 좋은 모델과 k 선택
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
            

print(f"Best Model: {best_model}")
print(f"Best Accuracy: {best_accuracy}")


# In[14]:


selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X, y)

scores = -np.log10(selector.pvalues_)
plt.bar(range(X.shape[1]), scores)
plt.xticks(range(X.shape[1]), data.columns[:-1], rotation=90)
plt.show()


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 11)

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

print('Logistic Regression Accuracy:', lr_acc)
print('Support Vector Machine:', svc_acc)
print('Decision Tree Accuracy:', dt_acc)
print('Random Forest Accuracy:', rf_acc)
print('XGBoost Accuracy:', xgb_acc)

models = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'XGBoost']
accuracies = [lr_acc, svc_acc, dt_acc, rf_acc, xgb_acc]

plt.bar(models, accuracies)
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()


# In[7]:


model = xgb_model


# In[8]:


# 모델에서 각 독립변수의 중요도 추출
importance = model.feature_importances_

# 중요도를 데이터프레임으로 변환
df_importance = pd.DataFrame({'feature': data.columns[:-1], 'importance': importance})

# 중요도를 내림차순으로 정렬
df_importance = df_importance.sort_values('importance', ascending=False)

# 중요도 시각화
plt.bar(df_importance['feature'], df_importance['importance'])
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()


# In[55]:


def predict_class(mixture_data):
    # 모델로 예측 수행
    pred = model.predict(mixture_data)+1
    
    # 예측 결과 출력
    print('Predicted Class:', pred[0])


# In[72]:


import tkinter.ttk as ttk
from tkinter import *

root = Tk()
root.title("MCS Calaulator")  # 타이틀명 지정
root.geometry("640x640")  # 가로 x 세로의 크기 지정

API_df = df[(df['Function']=='API')]
API_list = API_df.index.to_list()
API_name_box = ttk.Combobox(root, width=40, height=5, values=API_list)
API_name_box.pack()
API_name_box.set("Select API")  # 최초 목록 제목 설정 / 버튼 클릭을 통한 값 설정도 가능

lb1 = Label(root)
lb1.configure(text = "Input API content(%)")
lb1.pack()
API_content_box = Entry(root, width=10)
API_content_box.pack()

Excipient_df = df[(df['Function']!='API')]
Excipient_list = Excipient_df.index.to_list()

Excipient1_name_box = ttk.Combobox(root, width=40, height=5, values=Excipient_list)
Excipient1_name_box.pack()
Excipient1_name_box.set("Select Excipient1")

lb2 = Label(root)
lb2.configure(text = "Input Excipient1 content(%)")
lb2.pack()
Excipient1_content_box = Entry(root, width=10)
Excipient1_content_box.pack()

Excipient2_name_box = ttk.Combobox(root, width=40, height=5, values=Excipient_list)
Excipient2_name_box.pack()
Excipient2_name_box.set("Select Excipient2")

lb3 = Label(root)
lb3.configure(text = "Input Excipient2 content(%)")
lb3.pack()
Excipient2_content_box = Entry(root, width=10)
Excipient2_content_box.pack()


Excipient3_name_box = ttk.Combobox(root, width=40, height=5, values=Excipient_list)
#Excipient_name_box.current(0)  # 0번째 인덱스 값 선택
Excipient3_name_box.pack()
Excipient3_name_box.set("Select Excipient3")

lb4 = Label(root)
lb4.configure(text = "Input Excipient3 content(%)")
lb4.pack()
Excipient3_content_box = Entry(root, width=10)
Excipient3_content_box.pack()
#Excipient3_content_box.insert(0, "Input Excipient3 content(%)")


Excipient4_name_box = ttk.Combobox(root, width=40, height=5, values=Excipient_list) #state="readonly")
#Excipient_name_box.current(0)  # 0번째 인덱스 값 선택
Excipient4_name_box.pack()
Excipient4_name_box.set("Select Excipient4")

lb5 = Label(root)
lb5.configure(text = "Input Excipient4 content(%)")
lb5.pack()
Excipient4_content_box = Entry(root, width=10)
Excipient4_content_box.pack()

lb6 = Label(root)
lb7 = Label(root)
#lb8 = Label(root)
#lb9 = Label(root)
#lb10 = Label(root)
#lb11 = Label(root)
#lb12 = Label(root)
lb13 = Label(root)
lb14 = Label(root)
lb15 = Label(root)
lb16 = Label(root)
lb17 = Label(root)
lb18 = Label(root)

def btncmd():
    API_name = API_name_box.get()
    API_content = float(API_content_box.get())
    Excipient1_name = Excipient1_name_box.get()
    Excipient1_content = float(Excipient1_content_box.get())
    Excipient2_name = Excipient2_name_box.get()
    Excipient2_content = float(Excipient2_content_box.get())
    Excipient3_name = Excipient3_name_box.get()
    Excipient3_content = float(Excipient3_content_box.get())
    Excipient4_name = Excipient4_name_box.get()
    Excipient4_content = float(Excipient4_content_box.get())


    mixture_data = z1.loc[API_name]*API_content/100 + z1.loc[Excipient1_name]*Excipient1_content/100 + z1.loc[Excipient2_name]*Excipient2_content/100 + z1.loc[Excipient3_name]*Excipient3_content/100 + z1.loc[Excipient4_name]*Excipient4_content/100
    mixture_df = mixture_data.to_frame()
    mixture_df = mixture_df.transpose()	#행 열 전환
    
    #Best model로 예측하기
    pred = best_model.predict(mixture_df)+1
    
    lb6.configure(text = "")
    lb7.configure(text = "Mixture Class = " + str(pred[0]))
       
   
    #lb8.configure(text = "--------------------Calculation detail--------------------")
    #lb9.configure(text = "API PC1 score = "+API_PC1_score_1+" / content = "+API_content_box.get()+" %")
    #lb10.configure(text = "Excipient1 PC1 score = "+Excipient1_PC1_score_1+" / content = "+Excipient1_content_box.get()+" %")
    #lb11.configure(text = "Excipient2 PC1 score = "+Excipient2_PC1_score_1+" / content = "+Excipient2_content_box.get()+" %")
    #lb12.configure(text = "Excipient3 PC1 score = "+Excipient3_PC1_score_1+" / content = "+Excipient3_content_box.get()+" %")
    #lb13.configure(text = "Excipient4 PC1 score = "+Excipient4_PC1_score_1+" / content = "+Excipient4_content_box.get()+" %")
    #lb14.configure(text = "MCS score = " +MCS_score_1)
    lb13.configure(text = "Class 1 : Direct Compression")
    lb14.configure(text = "Class 2 : Dry Granulation")
    lb15.configure(text = "Class 3 : Wet Granulation")
    lb16.configure(text = "Class 4 : Other Technology")
    
    lb17.configure(text = f"Best Model: "+str(best_model))
    lb18.configure(text = f"Accuracy: "+str(best_accuracy))


    #MCS_df = pd.DataFrame({'API_name': [API_name], 'API_content(%)': [API_content], 'Excipient1_name': [Excipient1_name], 'Excipient1_content(%)': [Excipient1_content], 'Excipient2_name': [Excipient2_name], 'Excipient2_content(%)': [Excipient2_content], 'Excipient3_name': [Excipient3_name], 'Excipient3_content(%)': [Excipient3_content], 'MCS_score': [MCS_score], 'MCS': [MCS]})
    #MCS_df



btn = Button(root, text="Calculate", command=btncmd)
btn.pack()


lb5.pack()
lb6.pack()
lb7.pack()
#lb8.pack()
#lb9.pack()
#lb10.pack()
#lb11.pack()
#lb12.pack()
lb13.pack()
lb14.pack()
lb15.pack()
lb16.pack()
lb17.pack()
lb18.pack()



root.mainloop()  # 창이 닫히지 않게 해주는 것

