# -*- coding: utf-8 -*-
"""w14_edas.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JEBfOznDds4YyinxxdS5OUqUHJUwUCXE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

data = pd.read_csv("data_imp.csv").drop(columns=["need_statement"])

data.head()

data.not_fully_funded.value_counts()

data.describe()

numerical_cols = ['students_reached', 'fulfillment_labor_materials', 'title_word_count',
                  'description_word_count', 'need_word_count', 'essay_word_count']

readable_names = {
    'students_reached': 'Students Reached',
    'fulfillment_labor_materials': 'Fulfillment Cost',
    'title_word_count': 'Title Word Count',
    'description_word_count': 'Description Word Count',
    'need_word_count': 'Need Word Count',
    'essay_word_count': 'Essay Word Count'
}

# Rename columns in the dataframe for the correlation plot
data_readable = data.rename(columns=readable_names)

# Generate correlation heatmap with readable column names
plt.figure(figsize=(10, 6))
correlation_matrix = data_readable[list(readable_names.values())].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels 45 degrees
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(data=data, x='grade_level', y='essay_word_count')
plt.title('Average Essay Word Count by Grade Level')
plt.xlabel('Grade Level')
plt.ylabel('Average Essay Word Count')
plt.xticks(rotation=45)
plt.show()

numerical_cols = ['students_reached', 'fulfillment_labor_materials', 'title_word_count',
                  'description_word_count', 'need_word_count', 'essay_word_count']

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=False, bins=10)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 3. Count plots for categorical variables
categorical_cols = ['resource_type', 'school_state', 'grade_level']

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=data[col], order=data[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Distribution of Title Word Count
plt.figure(figsize=(8, 5))
sns.histplot(data['title_word_count'], bins=10)
plt.title('Distribution of Title Word Count')
plt.xlabel('Title Word Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of Description Word Count
plt.figure(figsize=(8, 5))
sns.histplot(data['description_word_count'], bins=10)
plt.title('Distribution of Description Word Count')
plt.xlabel('Description Word Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of Essay Word Count
plt.figure(figsize=(8, 5))
sns.histplot(data['essay_word_count'], bins=10)
plt.title('Distribution of Essay Word Count')
plt.xlabel('Essay Word Count')
plt.ylabel('Frequency')
plt.show()

# Count Plot of Resource Type
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='resource_type', order=data['resource_type'].value_counts().index)
plt.title('Count of Resource Types')
plt.xlabel('Resource Type')
plt.ylabel('Count')
plt.xticks(rotation=0)  # No rotation needed here
plt.show()

# Count Plot of School State
plt.figure(figsize=(12, 6))  # Increase figure width for better spacing
sns.countplot(data=data, x='school_state', order=data['school_state'].value_counts().index)
plt.title('Count of School States')
plt.xlabel('School State')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and adjust alignment
plt.tight_layout()  # Ensures labels fit within the figure
plt.show()

# Count Plot of Grade Level
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='grade_level', order=data['grade_level'].value_counts().index)
plt.title('Count of Grade Levels')
plt.xlabel('Grade Level')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')  # Slight rotation for spacing
plt.show()