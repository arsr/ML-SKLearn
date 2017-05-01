from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import utilities.utilities as utl
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import sklearn.model_selection




def DT():
    clf = DecisionTreeClassifier(criterion = "entropy", min_samples_split= 20, max_depth=10)
    clf, score, st_time = Kfoldwork(clf)
    predict(clf, score, "DT", st_time)

##Naive Bayes

def NaiveBAyes():
    clf = GaussianNB()
    clf, score, st_time = Kfoldwork(clf)
    predict(clf, score, "Naive Bayes", st_time)

#KNN
def KNN():
    clf = KNeighborsClassifier(n_neighbors=3)
    clf, score, st_time = Kfoldwork(clf)
    predict(clf, score, "KNN",st_time)

#Logit regression
def LogRegression():
    clf = LogisticRegression(penalty='l1')
    clf, score, st_time = Kfoldwork(clf)
    predict(clf, score, "Logit",st_time)

def SVM():

    clf = svm.SVC(C = 1.0, kernel='linear', max_iter=60000)
    clf, score, st_time  = Kfoldwork(clf)
    predict(clf, score, "SVM", st_time)

def RFC():
    clf = RandomForestClassifier(n_estimators= 10, min_samples_split= 10, criterion='gini')
    clf, score, st_time  = Kfoldwork(clf)
    predict(clf, score, "Random forest" ,st_time)

def Adaboost():
    clf = AdaBoostClassifier( n_estimators=10, learning_rate=1)
    clf, score, st_time = Kfoldwork(clf)
    predict(clf, score, "Ada Boost", st_time)


def predict(clf, training_Score,  model_name, st_time):
    star_time = timer()
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = timer()
    print('\n'+ str(model_name) +'\n'+"testing accuracy   : "+ str(accuracy))
    print("training accuracy :" + str(training_Score) +'\n')
    print("testing time :" + str( end_time - star_time))
    print("training time :" +str(st_time))
    accuracy_list.append(accuracy)

def Kfoldwork(clf):
    start_time = timer()
    kf = KFold(n_splits=10)
    for train, test in kf.split(x_train):
        train_x, test_x = x_train.iloc[train], x_train.iloc[test]
        train_y, test_y = y_train.iloc[train], y_train.iloc[test]

        clf = clf.fit(x_train, y_train)
        score = clf.score(test_x, test_y)

    end_time = timer()
    _time = end_time - start_time
    return  clf, score, _time



if __name__ == '__main__':

#change the flag to 1, to run it on Breast Cancere Data set
    flag = 0
    accuracy_list = []
    def GetTrainingData():

        Data = utl.GetData('train')

        x_train = Data[['Age', 'workclass', 'fnlwgt', 'marital-status', 'relationship', 'race', 'education-num', 'sex']]
        y_train = Data[['Salary']]
        y_train = y_train.squeeze()

        Data_test = utl.GetData('test')
        x_test = Data[['Age', 'workclass', 'fnlwgt', 'marital-status', 'relationship', 'race', 'education-num', 'sex']]
        y_test = Data[['Salary']]

        return  x_train, y_train, x_test, y_test


    def GetTrainingData_BCancer():
        data = utl.GetBreastCancerData()
        X = data[['CT', 'UC Size', 'UC Shape', 'MA', 'SECS', 'BN', 'BC', 'NN', 'Mito']]
        y = data[['Y']]
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.7)

        return   X_train, X_test, y_train, y_test

    if (flag == 0):
        x_train, y_train, x_test, y_test = GetTrainingData()
    else:
        x_train, x_test, y_train, y_test = GetTrainingData_BCancer()
        y_train = y_train.squeeze()
    RFC()
    Adaboost()
    SVM()
    DT()
    NaiveBAyes()
    KNN()
    LogRegression()
    

