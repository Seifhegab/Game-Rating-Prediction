import numpy as np
import pandas as pd
import statistics
import sklearn.preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

import time, datetime

import warnings
warnings.filterwarnings("ignore")

gamedata = pd.read_csv("games-regression-dataset.csv");

X = gamedata.iloc[:, :-1]
y = gamedata.iloc[:, -1]

gamedata.drop_duplicates(inplace=True)

print(gamedata.index)

# Data Spliting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=10)

X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)



class PreProcessing:
    Xtrain = pd.DataFrame()
    Xtest = pd.DataFrame()
    trainFeatures = []
    langMod = 0
    scale = MinMaxScaler()
    selectedFeaturesNames = []
    
    ratingCountMean = 0
    priceMean = 0
    ageRatingMean = 0
    sizeMean = 0
    GenreMode = ""
    originalDateMean = 0
    currentVersionMean = 0

    def __init__(self, Xtrain, Xtest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest

    def removeNulls(self, data):
        nullCols = data.columns[data.isna().mean() > 0.3]
        data.drop(nullCols, axis=1, inplace=True)
        #self.data.dropna(axis=0, inplace=True)

    def fillNulls(self, data, feature, value):
        data[feature].fillna(value, inplace=True)
        
    def dropDuplicates(self, xdata, ydata):
        xdata.drop_duplicates(inplace=True)
        ydata = ydata[xdata.index]

    def dateEncoding(self, data):
#         releaseDate = pd.to_datetime(data["Original Release Date"], dayfirst=True)
#         releaseDatelist = pd.to_datetime(releaseDate).dt.year.astype(str).str[2:]
#         updateDate = pd.to_datetime(data["Current Version Release Date"], dayfirst=True)
#         updateDatelist = pd.to_datetime(updateDate).dt.year.astype(str).str[2:]

#         data["Original Release Date"] = releaseDatelist
#         data["Current Version Release Date"] = updateDatelist
        
        releaseDate = pd.to_datetime(data["Original Release Date"], dayfirst=True)
        release = pd.to_datetime(releaseDate).values.astype(np.int64) // 10 ** 9
        data["Original Release Date"] = release
        
        updateDate = pd.to_datetime(data["Current Version Release Date"], dayfirst=True)
        update = pd.to_datetime(updateDate).values.astype(np.int64) // 10 ** 9
        data["Current Version Release Date"] = update
        
        self.originalDateMean = data["Original Release Date"].mean()
        self.currentVersionMean = data["Current Version Release Date"].mean()
        
    def update_app_purchase(self, data):
        n = data["In-app Purchases"]
        n = np.array(n)
        for i in range(0, len(n)):
            if type(n[i]) == type(""):
                n[i] = n[i].split(",")
            n[i] = pd.to_numeric(n[i])
            n[i] = (np.rint(n[i])).astype(int)
            n[i] = statistics.mean(n[i])
        data["In-app Purchases"] = n

    def setLanguageNumbers(self, data):
        featuresLang = data["Languages"]

        featuresLang = featuresLang.str.replace(", ", "")
        featuresLang = featuresLang.str.len() / 2

        data["Languages"] = featuresLang

    def setGenereNumbers(self, data):
        featuresLang = data["Genres"]
        
        featuresLang = featuresLang.str.split(", ")
        featuresLang = featuresLang.str.len()

        data["Genres"] = featuresLang
        
    def processAgeRating(self, data):
        data["Age Rating"] = data["Age Rating"].str.replace("+", "")
        self.ageRatingMean = data["Age Rating"].mean()
        
    def calculateGameAge(self, data):
        from datetime import date
        today = date.today()
        current_year = datetime.datetime.strptime(str(today),"%Y-%m-%d")
        timestamp = time.mktime(current_year.timetuple())
        series1 = int(timestamp)
        series2 = data["Original Release Date"].astype(int)
        data["Game Age"] = series1 - series2
        #self.data.drop(axis=1, columns="Current Version Release Date", inplace=True)

    def encodeGeneres(self, data):
        
        dummy = pd.get_dummies(data['Primary Genre'], prefix='', prefix_sep='')
        for col in dummy.columns:
            data[col] = dummy[col]
        data.drop(axis=1, columns="Primary Genre", inplace=True)
        
    def make_dummy_Frames(self, data):
        # change to array
        m = data["Languages"].astype(str)
        m = np.array(m)
        # splitting the list in each row
        for i in range(0, len(m)):
            m[i] = m[i].split(", ")
        # change to dataframe
        m = pd.DataFrame(m, columns=['Lang'])
        # make dummy variables
        k = pd.get_dummies(m.explode(['Lang'])).groupby(level=0).sum()
        # reset this column to be 1 because it is repeated many times
        if(k.columns.__contains__("Lang_ZH")):
            k["Lang_ZH"] = k["Lang_ZH"].replace([2, 3, 4, 5, 6], 1)
        return k
    
    def makeGenereDummy(self, data):
        # change to array
        m = data["Genres"].astype(str)
        m = np.array(m)
        # splitting the list in each row
        for i in range(1, len(m)):
            m[i] = m[i].split(", ")
        # change to dataframe
        m = pd.DataFrame(m, columns=['Genres'])
        # make dummy variables
        k = pd.get_dummies(m.explode(['Genres'])).groupby(level=0).sum()
        for col in k.columns:
            data[col] = k[col].values
    
    
    def encodeDeveloper(self, data):
        dummy = pd.get_dummies(data['Developer'], prefix='', prefix_sep='')

        for col in dummy.columns:
            data[col] = dummy[col]
        data.drop(axis=1, columns="Developer", inplace=True)

    def dropZerosCols(self, data):
        cols = data.columns

        for col in cols:
            if (len(data) - (data[col] == 0).sum() < 15 ):
                data.drop(axis=1, columns=col, inplace=True)
                
    def dropUniqueCols(self, data):      
        cols = data.columns

        for col in cols:
            uniquePercent = (data[col].nunique()) / len(data)
            if (uniquePercent > 0.97 and col != "Size"):
                data.drop(axis=1, columns=col, inplace=True)
        
    def concatDummyVars(self, data, dummy):
        for col in dummy.columns:
            data[col] = dummy[col].values
        #data = pd.concat([self.Xtrain, dummy])
        
    def scalingFitTransform(self, data):
        cols = data.columns
        self.scale.fit(data)
        scaledData = self.scale.transform(data)
        scaled = pd.DataFrame(data=scaledData, columns=cols)
        return scaled
        
        
        
    def scalingTransform(self, test):
        cols = test.columns
        scaledTest = self.scale.transform(test)
        test = pd.DataFrame(data=scaledTest, columns=cols)
        return test
        
    def selectFeatures(self, data):
        featureSelector = SelectKBest(score_func=f_regression, k=38)
        topFeatures = featureSelector.fit_transform(data, y_train)
        selectedFeatures = featureSelector.get_support()
        for (s, a) in zip(selectedFeatures, self.trainFeatures):
            if(s == True):
                self.selectedFeaturesNames.append(a)
               

        topFeatures = pd.DataFrame(data=topFeatures)
        return topFeatures
    
    def mapSelectedFeatures(self, data):
        return data[self.selectedFeaturesNames]
        
        
    def handleMissingValues(self, data):
        self.fillNulls(data, "Price", self.priceMean)
        self.fillNulls(data, "Languages", self.langMod)
        self.fillNulls(data, "Primary Genre", self.GenreMode)
        self.fillNulls(data, "User Rating Count", self.ratingCountMean)
        self.fillNulls(data, "Current Version Release Date", self.currentVersionMean)
        self.fillNulls(data, "Original Release Date", self.ageRatingMean)
        self.fillNulls(data, "Size", self.sizeMean)
        
    def calculateCorrelaction(self, data, cols):
        mydata = pd.DataFrame(data=data, columns=cols)
        corrdata = pd.DataFrame.copy(mydata, deep=True)
        corrdata["Average User Rating"] = y_train
        
        corelation = corrdata.corr()
        plt.figure(figsize=(39, 39))
        sns.heatmap(corelation, cmap=plt.cm.Reds, annot=True)
        plt.show()
        return corelation
    
    def selectTopFeatures(self, data):
        corrfeat = self.calculateCorrelaction(data)
        topfeatures = corrfeat.index[abs(corrfeat["Average User Rating"]) > 0.05]
        topfeatures = topfeatures.delete(-1)
        print(topfeatures)
        return topfeatures
        
    def cleanTrainData(self):
        ########
        print("-----------------------------------")
        print("PreProcessing Train")
        print("-----------------------------------")
        
        dummy_frames = self.make_dummy_Frames(self.Xtrain)
        self.concatDummyVars(self.Xtrain, dummy_frames)
        if(self.Xtrain.columns.__contains__("Lang_nan")):
            self.Xtrain.drop(columns=["Lang_nan"], inplace=True, axis=1)
        # print(type(dummy_frames))
        self.fillNulls(self.Xtrain, "In-app Purchases", '0')
        self.removeNulls(self.Xtrain)
        self.dateEncoding(self.Xtrain)
        self.encodeDeveloper(self.Xtrain)
        self.update_app_purchase(self.Xtrain)
        self.setLanguageNumbers(self.Xtrain)
        self.processAgeRating(self.Xtrain)
        print(self.Xtrain["Primary Genre"].mode())
        self.GenreMode = self.Xtrain["Primary Genre"].mode()[0]
        self.priceMean = self.Xtrain["Price"].mean()
        self.sizeMean = self.Xtrain["Size"].mean()
        self.ratingCountMean = self.Xtrain["User Rating Count"].mean()
        self.langMod = self.Xtrain["Languages"].mode()[0]
        
        self.handleMissingValues(self.Xtrain)
        
        self.calculateGameAge(self.Xtrain)
        self.encodeGeneres(self.Xtrain)
        self.makeGenereDummy(self.Xtrain)
        self.setGenereNumbers(self.Xtrain)
        
        #self.dropZerosCols(self.Xtrain)
        self.dropUniqueCols(self.Xtrain)
        #self.Xtest["Languages"].fillna(axis=0, inplace=True, value=self.langMod[0])
        self.trainFeatures = self.Xtrain.columns
        
        scaledTrain = self.scalingFitTransform(self.Xtrain)
        selected = self.selectFeatures(scaledTrain)
        self.calculateCorrelaction(selected, self.selectedFeaturesNames)
        #corrTopSelected = self.selectTopFeatures(selected)
        
        #selected = selected[corrTopSelected]
        
        return selected
        
        
    def chooseXTestCols(self, data):
        cols = data.columns
        for col in cols:
            #print(self.trainFeatures.__contains__(col))
            if(self.trainFeatures.__contains__(col) == False):
                data.drop(axis=1, columns=col, inplace=True)
        
        for feat in self.trainFeatures:
            if(cols.__contains__(feat) == False):
                data[feat] = 0
        
    
    
    def cleanTestingData(self, data):
        print("-----------------------------------")
        print("PreProcessing Test")
        print("-----------------------------------")
        dummy_frames = self.make_dummy_Frames(data)
        self.concatDummyVars(data, dummy_frames)
        self.fillNulls(data, "In-app Purchases", '0')
        self.dateEncoding(data)
        self.encodeDeveloper(data)
        self.update_app_purchase(data)
        self.setLanguageNumbers(data)
        
        self.handleMissingValues(data)
        
        self.calculateGameAge(data)
        self.processAgeRating(data)
        self.encodeGeneres(data)
        self.makeGenereDummy(data)
        self.setGenereNumbers(data)
        self.chooseXTestCols(data)
        
        xtestfeat = data[self.trainFeatures]
        data = xtestfeat
        scaledXtest = self.scalingTransform(data)
        selectedXtest = self.mapSelectedFeatures(scaledXtest)
        return selectedXtest

preprocessing = PreProcessing(X_train, X_test)

topFeatures = None
testTopFeat = None



import pickle



from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score


# Declare models

lr = linear_model.LinearRegression()
ridgeReg = Ridge(alpha=0.1)
lasso = Lasso(alpha=0.1)

def saveModel(model, name):
    pickle.dump(model, open(name + '.pkl', 'wb'))




def linearRegressionModel():
    # Fit the model

    lr.fit(topFeatures, y_train)
    yprediction = lr.predict(testTopFeat)

    print("\nSimple Linear Regression Model----------------------------------\n")
    print('Co-efficient of linear regression', lr.coef_)
    print('Intercept of linear regression model', lr.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, yprediction))
    print('R2 Score ', r2_score(y_test, yprediction))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, yprediction)))
    saveModel(lr, "linearreg")
    return yprediction


def ridgeRegressionModel():
    ridgeReg.fit(topFeatures, y_train)
    ypredridge = ridgeReg.predict(testTopFeat)

    print("\nRidge Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression', ridgeReg.coef_)
    print('Intercept of linear regression model', ridgeReg.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredridge))
    print('R2 Score ', r2_score(y_test, ypredridge))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, ypredridge)))
    saveModel(ridgeReg, "ridgereg")
    return ypredridge


# Lasso regression model

def lassoRegressionModel():
    lasso.fit(topFeatures, y_train)
    ypredlasso = lasso.predict(testTopFeat)

    print("\nLasso Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression', lasso.coef_)
    print('Intercept of linear regression model', lasso.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredlasso))
    print('R2 Score ', r2_score(y_test, ypredlasso))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, ypredlasso)))
    saveModel(lasso, "lassoreg")
    return ypredlasso





def loadModel(name, train , test, y):
    print("-----------------------------------")
    print("Loading Model " + name)
    print("-----------------------------------")
    pickled_model = pickle.load(open(name + '.pkl', 'rb'))
    ypredpickle = pickled_model.predict(test)
    
    print('Co-efficient ', pickled_model.coef_)
    print('Intercept', pickled_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y, ypredpickle))
    print('R2 Score ', r2_score(y, ypredpickle))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y, ypredpickle)))



ypred = None
ypred2 = None
ypred3 = None

trained = False

def plotModel(X, Y, Pre):
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    for i in range(X.shape[1]):
        plt.scatter(X[:, i], Y)

    for i in range(X.shape[1]):
        plt.plot(X[:, i], Pre, color='red', linewidth=3)

    plt.show()

while (True):
    print("-----------------------------------")
    print("Enter a choice")
    print("1) Train Models")
    print("2) Load Previously Trained Models")
    print("3) Test New Data")
    print("0) Exit")
    print("-----------------------------------")
    choice = input();
    if(choice == "1"):
        
        if(trained == False):
            topFeatures = preprocessing.cleanTrainData()
            testTopFeat = preprocessing.cleanTestingData(X_test)
            trained = True
        y_test = np.array(y_test)
        ypred = linearRegressionModel()
        ypred2 = ridgeRegressionModel()
        ypred3 = lassoRegressionModel()
        
        plotModel(testTopFeat, y_test, ypred)
        plotModel(testTopFeat, y_test, ypred2)
        plotModel(testTopFeat, y_test, ypred3)
        
    elif (choice == "2"):
        if(trained == False):
            topFeatures = preprocessing.cleanTrainData()
            testTopFeat = preprocessing.cleanTestingData(X_test)
            trained = True

        loadModel("linearreg", topFeatures, testTopFeat, y_test)
        loadModel("ridgereg", topFeatures, testTopFeat, y_test)
        loadModel("lassoreg", topFeatures, testTopFeat, y_test)

        

        
        
        
        
    elif(choice == "3"):
        print("Enter filename")
        filename = input();
        testdata = pd.read_csv(filename);
        testdata.drop_duplicates(inplace=True)
        test = testdata.iloc[:, :-1]
        ytest = testdata.iloc[:, -1]
        cleanedTest = preprocessing.cleanTestingData(test)
        loadModel("linearreg", topFeatures, cleanedTest, ytest)
        loadModel("ridgereg", topFeatures, cleanedTest, ytest)
        loadModel("lassoreg", topFeatures, cleanedTest, ytest)
        
    else:
        break






















