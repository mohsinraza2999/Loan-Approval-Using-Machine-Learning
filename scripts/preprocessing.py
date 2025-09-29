import pandas as pd

def load_raw_data():
    df=pd.read_csv('data/raw/LoanApprovalPrediction.csv')
    return drop_unwanted(df)

def feature_engineering(df):
    df['Loan_Status']=df['Loan_Status'].replace({'Y':1,'N':0})

    df['Education']=df['Education'].replace({'Not Graduate':0,'Graduate':1})

    df['Property_Area']=df['Property_Area'].replace({'Rural':0,'Urban':1,'Semiurban':2})

    df['Self_Employed']=df['Self_Employed'].replace({'No':0,'Yes':1})

    return df

    
def drop_unwanted(df):
    df=df.drop(columns=['Loan_ID','Gender','Married'],axis=1)

    df=df.dropna(axis=0)
    return feature_engineering(df)

def clean_data():

    df=load_raw_data()

    name='preprocess_data'

    path='data/training_data/'

    df.to_csv(path+name+'.csv')