import numpy as np
import pandas as pd
import statistics
import sklearn.preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

import time, datetime

import warnings
warnings.filterwarnings("ignore")

gamedata = pd.read_csv("games-classification-dataset.csv");

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
        featureSelector = SelectKBest(score_func=f_classif, k=38)
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

        return self.selectFeatures(scaledTrain)
        
        
    def chooseXTestCols(self, data):
        cols = data.columns
        for col in cols:
            #print(self.trainFeatures.__contains__(col))
            if(self.trainFeatures.__contains__(col) == False):
                data.drop(axis=1, columns=col, inplace=True)
        
        for feat in self.trainFeatures:
            if(cols.__contains__(feat) == False):
                data[feat] = 0
        
    # def cleanTestData(self):
    #     print("-----------------------------------")
    #     print("PreProcessing Test")
    #     print("-----------------------------------")
    #     dummy_frames = self.make_dummy_Frames(self.Xtest)
    #     self.concatDummyVars(self.Xtest, dummy_frames)
    #     #self.Xtest.drop(columns=["Lang_nan"], inplace=True, axis=1)
    #     self.fillNulls(self.Xtest, "In-app Purchases", '0')
    #     self.dateEncoding(self.Xtest)
    #     self.encodeDeveloper(self.Xtest)
    #     self.update_app_purchase(self.Xtest)
    #     self.setLanguageNumbers(self.Xtest)
    #     self.calculateGameAge(self.Xtest)
    #     self.processAgeRating(self.Xtest)
    #     self.encodeGeneres(self.Xtest)
    #     self.makeGenereDummy(self.Xtest)
    #     self.setGenereNumbers(self.Xtest)
    #     self.chooseXTestCols(self.Xtest)
    #     self.fillNulls(self.Xtest, "Languages", self.langMod[0])
    #     #self.Xtest["Languages"].fillna(axis=0, inplace=True, value=self.langMod[0])
    #     xtestfeat = self.Xtest[self.trainFeatures]
    #     self.Xtest = xtestfeat
    #     scaledXtest = self.scalingTransform(self.Xtest)
    #     selectedXtest = self.mapSelectedFeatures(scaledXtest)
    #     return selectedXtest
    
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







        

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pickle
import time


logisticTime = 0
SVMTime = 0
KNNTime = 0
DTTime = 0
RFTime = 0

logisticTimeHP = 0
KNNTimeHP = 0
DTTimeHP = 0
RFTimeHP = 0

logisticAccuracy = 0
SVMAccuracy = 0
KNNAccuracy = 0
DTAccuracy = 0
RFAccuracy = 0

logisticHPAccuracy = 0
KNNHPAccuracy = 0
DTHPAccuracy = 0
RFHPAccuracy = 0


def saveModel(model, name):
    pickle.dump(model, open(name + '.pkl', 'wb'))

def logistcReg():
    classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
    startTime = time.time()
    classifier.fit(topFeatures, y_train)
    endTime = time.time()
    global logisticTime
    logisticTime = round(endTime - startTime, 2)
    ypred = classifier.predict(testTopFeat)
    
    logisticAccuracy = round(classifier.score(testTopFeat, y_test) * 100, 2)
    
    print("-----------------------------------")
    print("Logistic Regression")
    print("-----------------------------------")
    print("Test Accuracy: " + str(logisticAccuracy))
    print("Train Accuracy: " + str(round(classifier.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypred)
    print(cm)
    print("-----------------------------------")
    print("Logistic Regression After Hyperparameter tuning")
    print("-----------------------------------")
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    startTimeHP = time.time()
    grid_search.fit(topFeatures, y_train)
    endTimeHP = time.time()
    global logisticTimeHP
    logisticTimeHP = round(endTimeHP - startTimeHP, 2)
    
    logisticHPAccuracy = round(grid_search.score(testTopFeat, y_test) * 100, 2)
    
    print("Best value of C:", grid_search.best_params_['C'])
    print("Validation score:", grid_search.best_score_)
    print("Test Accuracy: " + str(logisticHPAccuracy))
    print("Train Accuracy: " + str(round(grid_search.score(topFeatures, y_train) * 100, 2)))
    
    saveModel(grid_search, "logistic")
    
    return logisticAccuracy, logisticHPAccuracy



def KNNClassifier():
    knn = KNeighborsClassifier(n_neighbors=3)
    startTime = time.time()
    knn.fit(topFeatures, y_train)
    endTime = time.time()
    global KNNTime
    KNNTime = round(endTime - startTime, 2)
    ypredknn = knn.predict(testTopFeat)
    
    KNNAccuracy = round(knn.score(testTopFeat, y_test) * 100, 2)
    
    print("-----------------------------------")
    print("KNN Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(KNNAccuracy))
    print("Train Accuracy: " + str(round(knn.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredknn)
    print(cm)
    print("-----------------------------------")
    print("KNN Classification After Hyperparameter tuning")
    print("-----------------------------------")
    param_grid = {
        'n_neighbors': [1, 2, 3, 4, 5],
        'weights': ['uniform'],
        'algorithm': ['ball_tree']
    }
    knn_grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    startTimeHP = time.time()
    knn_grid.fit(topFeatures, y_train)
    endTimeHP = time.time()
    global KNNTimeHP
    KNNTimeHP = round(endTimeHP - startTimeHP, 2)
    print("Best Params" + str(knn_grid.best_params_))    
    print("Best Score" + str(round(knn_grid.best_score_ * 100, 2)))  
    ypredknnhyp = knn_grid.predict(testTopFeat)
    
    KNNHPAccuracy = round(knn_grid.score(testTopFeat, y_test) * 100, 2)
    
    print("Test Accuracy: " + str(KNNHPAccuracy))
    print("Train Accuracy: " + str(round(knn_grid.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredknnhyp)
    print(cm)
    
    saveModel(knn_grid, "knn")
    
    return KNNAccuracy, KNNHPAccuracy
    
def SVMClassifier():
    C=10
    SVM = svm.SVC(kernel='poly', degree=10, C=C)
    startTime = time.time()
    SVM.fit(topFeatures, y_train)
    endTime = time.time()
    global SVMTime
    SVMTime = round(endTime - startTime, 2)
    ypredsvm = SVM.predict(testTopFeat)
    
    SVMAccuracy = round(SVM.score(testTopFeat, y_test) * 100, 2)
    
    print("-----------------------------------")
    print("SVM Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(SVMAccuracy))
    print("Train Accuracy: " + str(round(SVM.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredsvm)
    print(cm)
    
    saveModel(SVM, "svm")
    
    return SVMAccuracy



def DTreeClassifier():
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=(10))
    startTime = time.time()
    dt.fit(topFeatures, y_train)
    endTime = time.time()
    global DTTime
    DTTime = round(endTime - startTime, 2)
    ypreddt = dt.predict(testTopFeat)
    
    DTAccuracy = round(dt.score(testTopFeat, y_test) * 100, 2)
    
    print("-----------------------------------")
    print("Decision Tree Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(DTAccuracy))
    print("Train Accuracy: " + str(round(dt.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypreddt)
    print(cm)
    print("-----------------------------------")
    print("Decision Tree Classification After Hyperparameter tuning")
    print("-----------------------------------")
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'min_samples_split': [30],
        'criterion': ['gini']
        }
    dt_grid = GridSearchCV(dt, param_grid, cv=5)
    startTimeHP = time.time()
    dt_grid.fit(topFeatures, y_train)
    endTimeHP = time.time()
    global DTTimeHP
    DTTimeHP = round(endTimeHP - startTimeHP, 2)
    ypreddthyp = dt_grid.predict(testTopFeat)
    
    DTHPAccuracy = round(dt_grid.score(testTopFeat, y_test) * 100, 2)
    
    print("Best Params" + str(dt_grid.best_params_))    
    print("Best Score" + str(round(dt_grid.best_score_ * 100, 2)))  
    print("Test Accuracy: " + str(DTHPAccuracy))
    print("Train Accuracy: " + str(round(dt_grid.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypreddthyp)
    print(cm)
    
    saveModel(dt_grid, "dt")
    
    return DTAccuracy, DTHPAccuracy


def RFClassifier():
    rf = RandomForestClassifier(max_depth=(15), max_features="log2", min_samples_leaf=2, min_samples_split= 13, n_estimators = 39)
    startTime = time.time()
    rf.fit(topFeatures, y_train)
    endTime = time.time()
    global RFTime
    RFTime = round(endTime - startTime, 2)
    ypredrf = rf.predict(testTopFeat)
    
    RFAccuracy = round(rf.score(testTopFeat, y_test) * 100, 2)
    
    print("-----------------------------------")
    print("Random Forest Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(RFAccuracy))
    print("Train Accuracy: " + str(round(rf.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredrf)
    print(cm)
    print("-----------------------------------")
    print("Random Forest Classification after hyperparamter tuning")
    print("-----------------------------------")
    #param_grid = {'n_estimators': [50, 100, 150, 200], 'max_depth': [2, 3, 4, 5, 6, 7, 8], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8]}
    
    param_grid = {
        'n_estimators': randint(10, 100),
        'max_features': ['auto'],
        'max_depth': [None] + list(range(5, 50, 5)),
        'min_samples_split': [15], # Required to split
        'min_samples_leaf': [1],
        'criterion': ['gini']
    }
    
    rf_grid = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    startTimeHP = time.time()
    rf_grid.fit(topFeatures, y_train)
    endTimeHP = time.time()
    global RFTimeHP
    RFTimeHP = round(endTimeHP - startTimeHP, 2)
    ypredrfhyp = rf_grid.predict(testTopFeat)
    
    RFHPAccuracy = round(rf_grid.score(testTopFeat, y_test) * 100, 2)
    
    print("Best Params" + str(rf_grid.best_params_))    
    print("Best Score" + str(round(rf_grid.best_score_ * 100, 2)))  
    print("Test Accuracy: " + str(RFHPAccuracy))
    print("Train Accuracy: " + str(round(rf_grid.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredrfhyp)
    print(cm)
    
    saveModel(rf_grid, "rf")
    
    return RFAccuracy, RFHPAccuracy



def loadModel(name, train , test, y):
    print("-----------------------------------")
    print("Loading Model " + name)
    print("-----------------------------------")
    pickled_model = pickle.load(open(name + '.pkl', 'rb'))
    startTime = time.time()
    ypredpickle = pickled_model.predict(test)
    endTime = time.time()
    testTime = round(endTime - startTime, 2);
    print("Best Params" + str(pickled_model.best_params_))    
    print("Best Score" + str(round(pickled_model.best_score_ * 100, 2)))  
    print("Test Accuracy: " + str(round(pickled_model.score(test, y) * 100, 2)))
    #print("Train Accuracy: " + str(round(pickled_model.score(train, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y, ypredpickle)
    print(cm)
    
    return testTime
    
def loadNativeModel(name, train , test, y):
    print("-----------------------------------")
    print("Loading Model " + name)
    print("-----------------------------------")
    pickled_model = pickle.load(open(name + '.pkl', 'rb'))
    startTime = time.time()
    ypredpickle = pickled_model.predict(test)
    endTime = time.time()
    testTime = round(endTime - startTime, 2);
    print("Test Accuracy: " + str(round(pickled_model.score(test, y) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y, ypredpickle)
    print(cm)
    return testTime
    
import matplotlib.pyplot as plt

totalTime = 0
totalTimeHP = 0


def plotTime(data, dataWithHP):
    
    print(data.values())
    totalTime = sum(data.values());
    totalTimeHP = sum(dataWithHP.values());
    
    totalData = {
        "Total Time": totalTime,
        "With Hyperparameter tuning": totalTimeHP
        }
    
    plt.bar(data.keys(), data.values(), width = 0.4)
    
    plt.xlabel("Models")
    plt.ylabel("Time")
    plt.title("Model without hyperparameter tuning")
    plt.show()
    
    plt.bar(dataWithHP.keys(), dataWithHP.values(), width = 0.4)
    
    plt.xlabel("Models")
    plt.ylabel("Time")
    plt.title("Model with hyperparameter tuning")
    plt.show()
    
    plt.bar(totalData.keys(), totalData.values(), width = 0.4)
    
    plt.xlabel("Models")
    plt.ylabel("Time")
    plt.title("Total Training Time")
    plt.show()
    
    
def plotTestTime(data):
    
    totalTime = sum(data.values());
    
    totalData = {
        "Total Time": totalTime,
        }
    
    plt.bar(data.keys(), data.values(), width = 0.4)
    
    plt.xlabel("Models")
    plt.ylabel("Time")
    plt.title("Models test time")
    plt.show()
    
    
    
    plt.bar(totalData.keys(), totalData.values(), width = 0.4)
    
    plt.xlabel("Models")
    plt.ylabel("Time")
    plt.title("Total Test Time")
    plt.show()

def plotAccuracy(data, dataWithHP):
    
    
    
    plt.bar(data.keys(), data.values(), width = 0.4)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accurcy without hyperparameter tuning")
    plt.ylim(50, 70)
    
    plt.show()
    
    print(dataWithHP.values())
    
    plt.bar(dataWithHP.keys(), dataWithHP.values(),
        width = 0.4)
    
    
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accurcy with hyperparameter tuning")
    plt.ylim(50, 70)
    plt.show()
    
trained = False

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
        logAcc, logAccHp = logistcReg()
        knnAcc, knnAccHp = KNNClassifier()
        svmAcc = SVMClassifier()
        dtAcc, dtAccHp = DTreeClassifier()
        rfAcc, rfAccHp = RFClassifier()
        
        print("-----------------------------------")
        print("Fitting Time in Seconds")
        print("1) Logistic " + str(logisticTime))
        print("2) KNN " + str(KNNTime))
        print("3) SVM " + str(SVMTime))
        print("0) DT " + str(DTTime))
        print("0) RF " + str(RFTime))
        print("-----------------------------------")
        
        print("-----------------------------------")
        print("Fitting time after hyperparamener in seconds")
        print("1) Logistic " + str(logisticTimeHP))
        print("2) KNN " + str(KNNTimeHP))
        print("0) DT " + str(DTTimeHP))
        print("0) RF " + str(RFTimeHP))
        print("-----------------------------------")
        
        data = {
            "Logistic Regression": logisticTime,
            "KNN": KNNTime,
            "SVM": SVMTime,
            "DT": DTTime,
            "RF": RFTime
            }
        dataWithHP = {
            "Logistic Regression": logisticTimeHP,
            "KNN": KNNTimeHP,
            "Decision Tree": DTTimeHP,
            "Random Forest": RFTimeHP
            }
        
        plotTime(data, dataWithHP)
        
        data = {
            "Logistic Regression": logAcc,
            "KNN": knnAcc,
            "SVM": svmAcc,
            "DT": dtAcc,
            "RF": rfAcc
            }
        dataWithHP = {
            "Logistic Regression": logAccHp,
            "KNN": knnAccHp,
            "DT": dtAccHp,
            "RF": rfAccHp
            }
        
        plotAccuracy(data, dataWithHP)
        
        
    elif (choice == "2"):
        if(trained == False):
            topFeatures = preprocessing.cleanTrainData()
            testTopFeat = preprocessing.cleanTestingData(X_test)
            trained = True

        logTime = loadModel("logistic", topFeatures, testTopFeat, y_test)
        dtTime = loadModel("dt", topFeatures, testTopFeat, y_test)
        knnTime = loadModel("knn", topFeatures, testTopFeat, y_test)
        rfTime = loadModel("rf", topFeatures, testTopFeat, y_test)
        svmTime = loadNativeModel("svm", topFeatures, testTopFeat, y_test)
        
        data = {
            "Logistic Regression": logTime,
            "KNN": knnTime,
            "SVM": svmTime,
            "DT": dtTime,
            "RF": rfTime
            }
        
        plotTestTime(data)
        
        
        
    elif(choice == "3"):
        print("Enter filename")
        filename = input();
        testdata = pd.read_csv(filename);
        testdata.drop_duplicates(inplace=True)
        test = testdata.iloc[:, :-1]
        ytest = testdata.iloc[:, -1]
        cleanedTest = preprocessing.cleanTestingData(test)
        loadModel("logistic", topFeatures, cleanedTest, ytest)
        loadModel("dt", topFeatures, cleanedTest, ytest)
        loadModel("knn", topFeatures, cleanedTest, ytest)
        loadModel("rf", topFeatures, cleanedTest, ytest)
        loadNativeModel("svm", topFeatures, cleanedTest, ytest)
        
    else:
        break






















