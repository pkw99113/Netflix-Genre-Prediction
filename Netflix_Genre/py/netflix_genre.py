import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for machine learning
# pip install scikit-learn
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the dataset
df = pd.read_csv("netflix_titles.csv")
#print(df.head())

#print("\nShape of the dataset:", df.shape)

#print("\nColumn info:")
#print(df.dtypes)

# check for missing values
#print("\nMissing Values:")
#print(df.isnull().sum()) # shows how many missing values in each column 

#visualize the missing data - creating a heat map to see where values are missing
# yellow represent missing values # purple represents non-missing values 

# plt.figure(figsize=(10,6))
# sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
# plt.title("Missing Values Heatmap")
# plt.show()

# drop columns that are useless 
df = df.drop(['show_id', 'title', 'director', 'cast', 'description', 'date_added'], axis = 1)

#print("Shape after dropping columns:", df.shape)

#print(df.head())

# replace missing values
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown')
df['duration'] = df['duration'].fillna('Unknown')

# print("\nMissing Values After Cleaning:")
# print(df.isnull().sum())

# add new column of main genre
# drop the listed_in column

# take the first genre as the main genre
df['main_genre'] = df['listed_in'].apply(lambda x: x.split(",")[0])

# delete listed_in column
df = df.drop('listed_in', axis = 1)

# shows the genres of where the count is 1. Unique genre
rare_genre = df['main_genre'].value_counts()[df['main_genre'].value_counts()==1].index
df = df[~df['main_genre'].isin(rare_genre)]
#print("\nNon-Unique Main Genre Counts: \n", df['main_genre'].value_counts())

#print(df[['main_genre']].head())
#print(df.head())

# create a copy to be safe
df_model = df.copy()

# initialize label encoder
le = LabelEncoder()

# columns to encode characters to integers
cols_to_encode = ['type', 'country', 'rating', 'duration']

# encode each column
for col in cols_to_encode:
    df_model[col] = le.fit_transform(df_model[col])

#print(df_model.head())

