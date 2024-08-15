# Text-MachineLearning
This project was created on Jupyter Notebook. In this project, we took a dataset composed of Movies created with their positive and negative reviews. Taking the reviews and vectorizing them, turning into a numerical representation and transforming them to features to train different machine learning models. Some of the Machine Learning models used are Logistic Regression, Decision Tree Classifier, Random Forest Classifier. We used the F1 Score to evaluate the machine learning models.

Libraries necessary for this project:  

import math  
from sklearn.dummy import DummyClassifier  
from sklearn.metrics import f1_score, mean_squared_error, classification_report  
import numpy as np  
import pandas as pd  
import nltk  
import re  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from nltk.corpus import stopwords  
import matplotlib  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
import matplotlib.dates as mdates  
import seaborn as sns  
import sklearn.metrics as metrics  
from tqdm.auto import tqdm  
from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.linear_model import LogisticRegression, LinearRegression  
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
import seaborn as sns  
