#CLASSIFICATION 
#Classification using various models in sklearn

#INPUT: training data (stored in input3.csv)
#OUTPUT: Model name, model parameters, best score - score on train data, test score - score on test data,
#        Produces visualisations

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import svm, grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# # INPUT
data=np.loadtxt(sys.argv[1], delimiter=',',skiprows=1)

# for visualisations
pos_egs = data[np.where(data[:,2]==1)]
neg_egs=data[np.where(data[:,2]==0)]

# # EXTRACTING FEATURES
rows,cols=data.shape
feature_ubound = cols-1

# extracting training data (i.e. features)
X=data[:,0:feature_ubound]

# extracting labels
y = data[:,feature_ubound]

# #SPLIT INTO TRAINING AND TEST SETS
X_train_cv, X_test, y_train_cv, y_test = train_test_split(X, y, test_size=0.4, stratify=y)


#TRAINING VARIOUS MODELS

#initialise parameters
params_linear = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],'kernel':['linear']}
params_poly = {'C' : [0.1, 1, 3], 'degree' : [4, 5, 6], 'gamma' : [0.1, 0.5],'kernel':['poly']}
params_rbf = {'C' : [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma' : [0.1, 0.5, 1, 3, 6, 10],'kernel':['rbf']}
params_logistic = {'C':  [0.1, 0.5, 1, 5, 10, 50, 100]}
params_knn = {'n_neighbors' : range(1,51), 'leaf_size' : range(5,61,5)}
params_decision_tree = {'max_depth' : range(1,51), 'min_samples_split' : range(2,11)}
params_forest = {'max_depth' : range(1,51), 'min_samples_split' : range(2,11)}

all_params = [('svm_linear',params_linear,svm.SVC()),
              ('svm_polynomial',params_poly,svm.SVC()),
              ('svm_rbf',params_rbf,svm.SVC()),
              ('logistic',params_logistic,LogisticRegression()),
              ('knn',params_knn,KNeighborsClassifier()),
              ('decision_tree',params_decision_tree,tree.DecisionTreeClassifier()),
              ('random_forest',params_forest,RandomForestClassifier())]
              

#output file for assignment
ofh=open('outpuut3.csv','w')

#fit models
for model_name,p_g,est in all_params:
    clf = grid_search.GridSearchCV(estimator=est, param_grid=p_g, cv=5) #5 k folds cross validation
    clf.fit(X_train_cv,y_train_cv)
    best_score=clf.best_score_
    test_score=clf.score(X_test,y_test)
    
    #write to file for assignment
    ofh.write(model_name+','+str(best_score)+','+str(test_score)+'\n')
    
    print model_name
    print clf.best_params_
    print 'best score ',best_score
    print 'test score ', test_score
    print ''
    
    #VISUALISE CLASSIFIER BOUNDARIES
    h=0.02 #step size
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))
    Z=clf.predict(np.c_[x1.ravel(), x2.ravel()])
    Z = Z.reshape(x1.shape)
    plt.plot(pos_egs[:,0],pos_egs[:,1],'ro',neg_egs[:,0],neg_egs[:,1], 'bo')
    plt.contourf(x1, x2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.show()
    
ofh.close() 