import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.title('🤖 ThinkTankers ML App')

st.write('This app builds a machine learning model')

with st.expander('Data'):
    # Load the dataset
    df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
    st.write('Dataset:')
    st.dataframe(df)

    # Separate features (X) and target (Y)
    st.write('*X (Features)*')
    x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
    st.dataframe(x_raw)

    st.write('*Y (Target)*')
    y_raw = df['AtRiskOfOrExperiencingHomelessnessFlag']
    st.write(y_raw)

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='PeopleonApplication', y='FamilyType', color='AtRiskOfOrExperiencingHomelessnessFlag')

# Data prepration (user input)
with st.sidebar:
  st.header('Input features')
  Family = st.selectbox('Family Type', (
    'Single Person', 
    'Single Parent, 1 Child', 
    'Single Parent, 2 Children', 
    'Single Parent, >2 Children', 
    'Couple Only', 
    'Couple, 1 Child', 
    'Couple, 2 Children', 
    'Couple, >2 Children', 
    'Single Person Over 55', 
    'Couple Only Over 55', 
    'Other'
))
  DisabilityFlag = st.selectbox('Disability', ('Yes', 'No'))
  TotalPeople = st.slider('Total People on application', 1, 12, 2)
  TotalMonths = st.slider('Total months you have been register', 0, 239, 23)
  # 1.0 mininum, 12.0 maximum, 1.681 average (same as other)

# user input into dataframe
# Create a DataFrame for the input features
  data = {'FamilyType': Family,
          'MonthsonHousingRegister': TotalMonths,
          'DisabilityApplicationFlag': DisabilityFlag,
          'PeopleonApplication': TotalPeople,
         }
  input_df = pd.DataFrame(data, index=[0])
  input_details = pd.concat([input_df, x_raw], axis=0) #combine input features with housing features
  # input details is user input

with st.expander('Input features'):
  st.write('*Input User*')
  input_df
  st.write('*Combined Housing data*') #combine original dgn user input
  input_details

# Encode X 
encode = ['FamilyType', 'DisabilityApplicationFlag']
df_house = pd.get_dummies(input_details, prefix=encode)

x = df_house[1:] #ignore first row but use everything after first row
input_row = df_house[:1] #use only first row

# Encode y
target_mapper = {'Yes': 1,
                 'No': 0
                 }
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)
#y #ni yang encoded
#y_raw # original


with st.expander('Data preparation'):
  st.write('*Encoded X (input housing)*')
  input_row
  st.write('*Encoded y*')
  y
  # y_raw #compare dgn original nk tgk betul ke tak

# Object-Oriented Programming for Random Forest and XGBoost
class MLModel:
    def _init_(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)

    def predict(self, input_data):
        return self.model.predict(input_data), self.model.predict_proba(input_data)

# Procedural Programming for Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test, input_row):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    pred_label = model.predict(input_row)
    pred_prob = model.predict_proba(input_row)
    return accuracy, pred_label, pred_prob

# Model selection and training
with st.expander('Model Selection & Training'):
    model_option = st.selectbox('Choose a Model', ['Random Forest', 'Logistic Regression', 'XGBoost'])

    # Split the dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if model_option == 'Random Forest':
        rf_model = MLModel(RandomForestClassifier(random_state=42), 'Random Forest')
        rf_model.train(X_train, y_train)
        accuracy = rf_model.evaluate(X_test, y_test)
        prediction, prediction_proba = rf_model.predict(input_row)

    elif model_option == 'Logistic Regression':
        accuracy, prediction, prediction_proba = logistic_regression(X_train, y_train, X_test, y_test, input_row)

    elif model_option == 'XGBoost':
        xgb_model = MLModel(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 'XGBoost')
        xgb_model.train(X_train, y_train)
        accuracy = xgb_model.evaluate(X_test, y_test)
        prediction, prediction_proba = xgb_model.predict(input_row)

    st.write(f'Model Accuracy: {accuracy:.2f}')

# Deployment: Prediction
with st.expander('Risk Of or Experiencing Homelessness Prediction'):
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])
    st.write('Prediction Probabilities:')
    st.dataframe(df_prediction_proba.style.format({'No': '{:.2f}', 'Yes': '{:.2f}'}))

    homelessness = np.array(['No', 'Yes'])
    st.success(f'Prediction: {homelessness[prediction][0]}')
