import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Streamlit app title and intro
st.title('🤖 ThinkTankers ML App')
st.write('This app builds a machine learning model for homelessness risk prediction.')

# Data Loading
df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
y_raw = df['AtRiskOfOrExperiencingHomelessnessFlag']

# Encode target (y)
target_mapper = {'Yes': 1, 'No': 0}
y = y_raw.map(target_mapper)

# Encode features (X)
x = pd.get_dummies(x_raw, prefix=['FamilyType', 'DisabilityApplicationFlag'])

# User input handling
st.sidebar.header('Input Features')
input_data = {
    'FamilyType': st.sidebar.selectbox('Family Type', x_raw['FamilyType'].unique()),
    'MonthsonHousingRegister': st.sidebar.slider('Total months registered', 0, 239, 23),
    'DisabilityApplicationFlag': st.sidebar.selectbox('Disability', ['Yes', 'No']),
    'PeopleonApplication': st.sidebar.slider('People on Application', 1, 12, 2),
}
input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df, prefix=['FamilyType', 'DisabilityApplicationFlag'])
input_encoded = input_encoded.reindex(columns=x.columns, fill_value=0)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model selection and evaluation
st.sidebar.header('Model Selection')
model_choice = st.sidebar.radio('Choose a model:', ['Random Forest', 'Logistic Regression', 'XGBoost'])

if model_choice == 'Random Forest':
    model = RandomForestClassifier(random_state=42)
elif model_choice == 'Logistic Regression':
    model = LogisticRegression(max_iter=1000, random_state=42)
else:  # XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
prediction = model.predict(input_encoded)
prediction_proba = model.predict_proba(input_encoded)

st.write(f'**Model Accuracy:** {accuracy:.2f}')

# Results
st.subheader('Prediction Results')
pred_df = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])
st.write('Prediction Probabilities:')
st.dataframe(pred_df.style.format({'No': '{:.2f}', 'Yes': '{:.2f}'}))
result = 'At Risk' if prediction[0] == 1 else 'Not at Risk'
st.success(f'Prediction: {result}')
