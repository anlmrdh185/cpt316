import pandas as pd
import numpy as np
import streamlit as st

st.title('🤖 ThinkTankers ML App')

st.write('This app builds a machine learning model')

with st.expander('Data'):
    # Load the dataset
    df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
    st.write('Dataset:')
    st.dataframe(df)

    # Separate features (X) and target (Y)
    st.write('**X (Features)**')
    x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
    st.dataframe(x_raw)

    st.write('**Y (Target)**')
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
  input_details = pd.concat([input_df, x_raw], axis=0) #combine input features with penguin features
  # input details is user input

with st.expander('Input features'):
  st.write('**Input User**')
  input_df
  st.write('**Combined Housing data**') #combine original dgn user input
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
  st.write('**Encoded X (input housing)**')
  input_row
  st.write('**Encoded y**')
  y
  # y_raw #compare dgn original nk tgk betul ke tak
