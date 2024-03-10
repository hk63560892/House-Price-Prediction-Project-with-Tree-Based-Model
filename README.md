the-simpsons-characters-recognition-challenge-iii-hk63560892 created by GitHub Classroom

# 房價 Regression 數據分析

## 專案描述 
這是一個使用了xgboost，catboost以及lgbm模型來進行房價 regression 分析的專案。

## 數據源 
[https://www.kaggle.com/competitions/machine-learning-2023-nycu-regression/data](https://www.kaggle.com/competitions/machine-learning-2023-nycu-regression/data)

## 安裝和設置 
以下是在這個分析中使用的主要函式庫：
- from sklearn.feature_selection import VarianceThreshold
- from sklearn.metrics import mean_absolute_error
- import pandas as pd
- from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
- import matplotlib.pyplot as plt
- import numpy as np
- from sklearn.feature_selection import SelectKBest, f_regression
- import seaborn as sns
- from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, SpectralClustering
- from sklearn.pipeline import Pipeline
- from sklearn.decomposition import PCA
- import sweetviz as sv
- from sklearn.model_selection import cross_val_score, KFold, train_test_split
- from catboost import CatBoostRegressor
- import featuretools as ft
- import lightgbm as lgb
- import xgboost as xgb
- from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

要運行此專案，您需要首先安裝這些函式庫。

## 使用方法 
- 匯入所需的函式庫
- 匯入數據
- 查看數據的前幾行
- 顯示數據類型的一般概念
- 數據集的摘要統計
- 數據可視化
- 特徵工程
- 特徵選擇：Correlation
- 特徵選擇：Variance Threshold
- 特徵選擇：selectKbest
- 不使用Normalization
- 生成測試特徵
- Light GBM
- CAT BOOST
- XG boost
- 通過grid search 和random search 找出最適合三個模型的超參數

## 結果和視覺化 
請參考 `house_price_pred.ipynb` notebook 以查看完整的分析結果和相關視覺化。

