# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

# Classifier metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

import seaborn as sns
from scipy import stats

dataset_path = "data/final_cars_datasets.csv"


def readDataset():
    cars_data = pd.read_csv(dataset_path)
    X = cars_data.iloc[:, :-1].values
    y = cars_data.iloc[:, -1].values
    cars_data.head()
    return cars_data


def groupVariables(cars_data):
    numerical_vars = ['price', 'year', 'mileage', 'engine_capacity', ]
    numerical_vars_idx = cars_data.columns.get_indexer(numerical_vars)
    cat_vars = ['mark', 'model', 'transmission', 'drive', 'hand_drive', 'fuel']
    cat_vars_idx = cars_data.columns.get_indexer(cat_vars)
    return [cat_vars, numerical_vars]


def createDataset():
    data = readDataset()
    df = pd.DataFrame(data)
    df = df.drop(df.columns[[0]], axis=1)
    variables = groupVariables(data)
    cat_vars = variables[0]
    numerical_vars = variables[1]
    return df, cat_vars, numerical_vars


def tolydiniuAnalize():
    global df_numerical
    print("Skaitinio tipo reiksmes")
    df_numerical = df.select_dtypes(include='int64')
    print(df_numerical)
    print("\nBendras reiksmiu skaicius: ")
    print(df_numerical.count())
    print("\nTrukstamu reiksmiu skaicius:")
    print(df_numerical.isnull().sum())
    print("\nKardinalumas:")
    print(df_numerical.nunique())
    print("\nMinimali verte:")
    print(df_numerical.min())
    print("\nMaksimali verte:")
    print(df_numerical.max())
    print("\nPirmas kvartilis:")
    print(df_numerical.quantile(.25))
    print("\nTrecias kvartilis:")
    print(df_numerical.quantile(.75))
    print("\nVidurkis:")
    print(df_numerical.mean())
    print("\nMediana:")
    print(df_numerical.median())
    print("\nStandartinis nuokrypis:")
    print(df_numerical.std())


def kategoriniuAnalize():
    global df_categorical, z
    print("Kategorinio tipo reiksmes")
    df_categorical = df.select_dtypes(include='object')
    print(df_categorical)
    print("\nBendras reiksmiu skaicius: ")
    print(df_categorical.count())
    print("\nTrukstamu reiksmiu skaicius:")
    print(df_categorical.isnull().sum())
    print("\nKardinalumas:")
    print(df_categorical.nunique())
    print("\nModa:")
    print(df_categorical.mode())
    print("\nModos daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in df_categorical.mode().iterrows()]).sum())
    print("\nModos procentinis daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in df_categorical.mode().iterrows()]).sum() / len(
        df_categorical.index) * 100)
    print("\nAntroji moda:")
    temp_df = df_categorical
    modas = []
    for x, y in df_categorical.mode().iterrows():
        for z in range(0, len(y)):
            modas.append(y[z])
    for x in modas:
        temp_df = temp_df.replace(to_replace=x, value=np.nan, regex=True)
    print(temp_df.mode())
    print("\nAntrosios modos daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in temp_df.mode().iterrows()]).sum())
    print("\nAntrosios modos procentinis daznumas:")
    print(
        pd.concat([df_categorical.eq(x) for _, x in temp_df.mode().iterrows()]).sum() / len(df_categorical.index) * 100)


def fixOutliers():
    global z, df_numerical
    print("\nOutliers aptikimas:")
    z = np.abs(stats.zscore(df_numerical))
    print(z)
    threshold = 3
    print(np.where(z > 3))
    df_numerical = df_numerical[(z < 3).all(axis=1)]
    print(df_numerical.count())


def scatterTolydinis():
    print("\nGrafikai su stipria tiesine priklausomybe: metai/rida, kaina/rida")
    ax1 = df_numerical.plot.scatter(x='year', y='mileage')
    ax2 = df_numerical.plot.scatter(x='price', y='mileage')
    print("\nGrafikai su silpna tiesine priklausomybe: metai/variklis, kaina/variklis")
    ax3 = df_numerical.plot.scatter(x='year', y='engine_capacity')
    ax4 = df_numerical.plot.scatter(x='mileage', y='engine_capacity')
    plt.show()


def fuelvsTransmissionGraph():
    c2 = df_categorical['fuel'].value_counts().plot(kind='bar')
    c1 = df_categorical.query('`transmission` == "at"')
    c1 = c1.groupby(['fuel']).size().reset_index(name='counts')
    ax5 = c1.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'transmission' == at")
    c3 = df_categorical.query('`transmission` == "mt"')
    c3 = c3.groupby(['fuel']).size().reset_index(name='counts')
    ax6 = c3.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'transmission' == mt")


def drivevsFuelGraph():
    c4 = df_categorical.query('`drive` == "2wd"')
    c4 = c4.groupby(['fuel']).size().reset_index(name='counts')
    ax10 = c4.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'Drive' == 2wd")
    c6 = df_categorical.query('`drive` == "4wd"')
    c6 = c6.groupby(['fuel']).size().reset_index(name='counts')
    ax11 = c6.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'Drive' == 4wd")
    c7 = df_categorical.query('`drive` == "awd"')
    c7 = c7.groupby(['fuel']).size().reset_index(name='counts')
    ax12 = c7.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'Drive' == awd")


def drawBoxplots():
    boxplot1 = df.boxplot(by='drive', column='price', grid=False)
    plt.show()
    boxplot2 = df.boxplot(by='mark', column='price', grid=False, rot=90)
    plt.show()
    boxplot3 = df.boxplot(by='fuel', column='engine_capacity', grid=False, rot=90)
    plt.show()


def covandcorr():
    global corr
    print("\nKovariacija:")
    print(df.cov())
    corr = df.corr()
    print("\nKoreliacija:")
    print(corr)
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True)
    plt.show()


def normalizeDf():
    normalized_df = (df_numerical - df_numerical.min()) / (df_numerical.max() - df_numerical.min())
    print("Normalizuotas duomenu rinkinys:")
    print(normalized_df)


def categoricalToNumerical():
    print("Kategoriniai i tolydinius")
    for col_name in df_categorical.columns:
        if df_categorical[col_name].dtype == 'object':
            df_categorical[col_name] = df_categorical[col_name].astype('category')
        df_categorical[col_name] = df_categorical[col_name].cat.codes
    print(df_categorical)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.close('all')
    # Read data
    df, cat_vars, numerical_vars = createDataset()

    print(cat_vars)
    print(numerical_vars)

    # 2 Skaitinio tipo
    tolydiniuAnalize()

    # 2 Kategorinio tipo
    kategoriniuAnalize()

    # 4
    df.hist()
    plt.show()
    # 5-6
    fixOutliers()

    # 7.1 Tolydinio tipo scatter plot
    scatterTolydinis()

    # 7.2 Scatter Plot Matrix
    pd.plotting.scatter_matrix(df_numerical, alpha=0.2)
    plt.show()

    # 7.3 Kategorinio tipo bar plot

    # Fuel vs transmission
    fuelvsTransmissionGraph()

    # Drive vs fuel

    drivevsFuelGraph()

    # 7.4 Kategoriniai ir tolydiniai

    # Histogramos
    #ax13 = df.groupby(['mark'])['price'].plot(kind='bar', stacked=True)
    #plt.title("Mark vs price")

    # Boxplot
    drawBoxplots()

    # 8 Kovariacija ir koreliacija
    covandcorr()

    # 9 Duomenu normalizacija [0;1]
    normalizeDf()

    # 10 Kategoriniai i tolydinius
    categoricalToNumerical()
