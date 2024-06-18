## Main Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
import missingno
warnings.filterwarnings('ignore')
## sklearn -- preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


##Data Notebook
TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'diabetes.csv')
df = pd.read_csv(TRAIN_DATA_PATH)

diabetes_data = df.copy(deep = True)
diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].mean(), inplace = True)
diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].mean(), inplace = True)
diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].median(), inplace = True)
diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(), inplace = True)
diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(), inplace = True)



normal_threshold = (diabetes_data['BloodPressure'] < 120) & (diabetes_data['BloodPressure'] < 80)
prehypertension_threshold = ((diabetes_data['BloodPressure'] >= 120) & (diabetes_data['BloodPressure'] <= 139)) | ((diabetes_data['BloodPressure'] >= 80) & (diabetes_data['BloodPressure'] <= 89))
hypertension_threshold = (diabetes_data['BloodPressure'] >= 140) | (diabetes_data['BloodPressure'] >= 90)

# Create a new categorical column based on the thresholds
diabetes_data['BloodPressureCategory'] = 'Normal'
diabetes_data.loc[prehypertension_threshold, 'BloodPressureCategory'] = 'Prehypertension'
diabetes_data.loc[hypertension_threshold, 'BloodPressureCategory'] = 'Hypertension'

# Display the updated DataFrame with the new categorical column




# Define the BMI ranges and categories
bmi_ranges = [0, 18.5, 24.9, 29.9, float('inf')]
bmi_categories = ['Underweight', 'Normal Weight', 'Overweight', 'Obese']

# Create a new categorical column based on BMI ranges
diabetes_data['BMICategory'] = pd.cut(diabetes_data['BMI'], bins=bmi_ranges, labels=bmi_categories, right=False)





# Apply a logarithmic transformation to the "Insulin" feature
diabetes_data['Log_Insulin'] = np.log1p(diabetes_data['Insulin'])

# Apply a logarithmic transformation to the "SkinThickness" feature
diabetes_data['Log_SkinThickness'] = np.log1p(diabetes_data['SkinThickness'])

## to features and target
X = diabetes_data.drop(columns=['Outcome'], axis=1)
y = diabetes_data['Outcome']


## split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)



# ##PIPELINE

num_cols = ['Age', 'Pregnancies', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Glucose','Log_Insulin','Log_SkinThickness']
categ_cols = X_train.select_dtypes(exclude='number').columns.tolist()

num_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(num_cols)),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

## Categorical
categ_pipline = Pipeline(steps=[
                 ('selector', DataFrameSelector(categ_cols)),
                 ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

all_pipeline = FeatureUnion(transformer_list=[
                        ('num', num_pipline),
                        ('categ', categ_pipline)
                    ])

## apply  
_ = all_pipeline.fit_transform(X_train)


def process_new(x_new):
    df_new=pd.DataFrame([x_new],columns=X_train.columns)
    

    ##Adjust the datatypes
    df_new['Pregnancies']=df_new['Pregnancies'].astype('int64')
    df_new['Glucose']=df_new['Glucose'].astype('float64')
    df_new['BloodPressure']=df_new['BloodPressure'].astype('float64')
    df_new['SkinThickness']=df_new['SkinThickness'].astype('float64')
    df_new['Insulin']=df_new['Insulin'].astype('float64')
    df_new['BMI']=df_new['BMI'].astype('float64')
    df_new['DiabetesPedigreeFunction']=df_new['DiabetesPedigreeFunction'].astype('float64')
    df_new['Age']=df_new['Age'].astype('int64')
    df_new['BloodPressureCategory']=df_new['BloodPressureCategory'].astype('str')
    df_new['BMICategory']=df_new['BMICategory'].astype('str')
    df_new['Log_Insulin']=df_new['Log_Insulin'].astype('float64')
    df_new['Log_SkinThickness']=df_new['Log_SkinThickness'].astype('float64')
    
    ## Feature
    df_new['Log_Insulin'] = np.log1p(df_new['Insulin'])
    df_new['Log_SkinThickness'] = np.log1p(df_new['SkinThickness'])

    ## Apply the pipeline
    X_processed=all_pipeline.transform(df_new)


    return X_processed


