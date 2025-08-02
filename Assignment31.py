import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def BreastCancerCaseStudy(datapath):
    
    df = pd.read_csv(datapath)
    
    print("Data loaded Successfully :")
    
    print(df.head())
    print(df.shape)
    
    print("Null Values in Data:\n", df.isnull().sum())
    
    df.drop(columns=['CodeNumber'], inplace=True)
    
    print(df.head())
    
    df.replace("?", np.nan, inplace=True)
    
    df['BareNuclei'] = pd.to_numeric(df['BareNuclei'])
    
    df.dropna(inplace=True)
    
    print(df.shape)
    
    x = df.drop(columns=['CancerType'])
    y = df['CancerType'].map({2: 0, 4: 1})
    
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    
    DecisionTree(x_scale,y,x)
    Randomforest(x_scale,y,x)
    

def DecisionTree(x_scale,y,x):
    
    print("\n Random Forest Classifier  Result :\n")
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = DecisionTreeClassifier(max_depth=7)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    y_proba = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("DecisionTree Classifier Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    roc_auc = roc_auc_score(y_test,y_proba)
    print ("ROC-AUC:", roc_auc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("DecisionTree Classifier Confusion Matrix ")
    plt.show() 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()
    importance = pd.Series(model.feature_importances_,index=x.columns)
    importance = importance.sort_values(ascending=False)
    
    importance.plot(kind='bar', figsize=(10,7), title="Features Importance")
    plt.show()
    
       
def Randomforest(x_scale,y,x):
    
    print("\n Random Forest Classifier  Result :\n")
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = RandomForestClassifier(n_estimators=100,max_depth=7, random_state=42)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    y_proba = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Random Forest Classifier Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    roc_auc = roc_auc_score(y_test,y_proba)
    print ("ROC-AUC:", roc_auc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Classifier Confusion Matrix ")
    plt.show() 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()
    importance = pd.Series(model.feature_importances_,index=x.columns)
    importance = importance.sort_values(ascending=False)
    
    importance.plot(kind='bar', figsize=(10,5), title="Features Importance")
    plt.show() 


def main():
    
    BreastCancerCaseStudy("breast-cancer-wisconsin.csv")
    
if __name__ == "__main__":
    main()
