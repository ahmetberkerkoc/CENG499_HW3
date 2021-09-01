import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import delete
from numpy.random import gamma
from draw import draw_svm
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn   import datasets

def experiment1(train_data, train_labels):
    C_values = [0.01,0.1,1,10,100]
    for index in range(len(C_values)):
        path1 = 'results_for_svm/C_values_{}'.format(index)
        clf = SVC(C=C_values[index],kernel='linear')
        clf.fit(train_data, train_labels)
        draw_svm(clf,train_data,train_labels,-3,3,-3,3,path1)

def experiment2(train_data,train_labels):
    kernel_values = ['linear','rbf','poly','sigmoid']
    for index in range(len(kernel_values)):
        path1 = 'results_for_svm/kernel_values_{}'.format(kernel_values[index])
        clf = SVC(C=1,kernel=kernel_values[index])
        clf.fit(train_data, train_labels)
        draw_svm(clf,train_data,train_labels,-3,3,-3,3,path1)

def experiment3(train_data,train_labels,test_data,test_labels):
    #print(np.shape(train_data))

    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    
    parameters = {'kernel':('linear','rbf','poly','sigmoid'),'C':[0.01,0.1,1,10,100],'gamma':[0.00001,0.0001,0.001,0.01,0.1,1]}
    svc = SVC()
    clf=GridSearchCV(svc,parameters)
    clf.fit(d2_train_data,train_labels)
    print('Best score is {}'.format(clf.best_score_))
    print('Best parameters are C: {}, gamma: {}, kernel: {}'.format(clf.best_params_['C'],clf.best_params_['gamma'],clf.best_params_['kernel']))
    print('Test Score with best hyperparameters is {}'.format(clf.score(d2_test_data,test_labels))) #Test result for the best
    #print('finish')

def experiment3_manuel_grid_search(train_data,train_labels,test_data,test_labels):
    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    parameters = {'kernel':('linear','rbf','poly','sigmoid'),'C':[0.01,0.1,1,10,100],'gamma':[0.00001,0.0001,0.001,0.01,0.1,1]}
    clf = GridSearchCV(SVC(), parameters)
    clf.fit(d2_train_data,train_labels)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    
    #y_true, y_pred = test_labels, clf.predict(d2_test_data)
    #print(classification_report(y_true, y_pred))
    

def experiment4(train_data,train_labels,test_data,test_labels):
    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    clf = SVC(C=1,kernel='rbf')
    clf.fit(d2_train_data, train_labels)
    print('Test Accuracy for experimet 4 is {}'.format(clf.score(d2_test_data,test_labels)))
    #print('finish')

def my_confusion_matrix(train_data,train_labels,test_data,test_labels):
    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    clf = SVC(C=1,kernel='rbf')
    clf.fit(d2_train_data,train_labels)
    pred_labels=clf.predict(d2_test_data)
    tn,fp,fn,tp=confusion_matrix(test_labels,pred_labels).ravel()
    print('True Positive:{} || False Positive:{}  '.format(tp,fp))
    print('False Negative:{} || True Negative:{}  '.format(fn,tn))
    print("Recall: {} ,Precision: {}".format((tp/(tp+fn)),(tp/(tp+fp))))
    print("Accuracy: {}".format((tp+tn)/(tp+tn+fp+fn)))

    
def oversample_minority_class(train_data,train_labels,test_data,test_labels):
    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    occurance1_test = np.count_nonzero(train_labels)
    occurance0_test = np.count_nonzero(train_labels==0)
    #I know that 0 is minority class
    resample = occurance1_test/occurance0_test
    resample=int(resample)
    length = len(train_labels)
    for re in range(resample-1):
        for i in range(length):
            if(train_labels[i]==0):
                #print(d2_test_data[i])
                #print(d2_test_data[i].shape)
                #print(d2_test_data[i].shape[0])
                element=np.reshape(d2_train_data[i],(-1,d2_train_data[i].shape[0]))
                #print(element.shape)
                d2_train_data=np.append(d2_train_data,element,axis=0)
                train_labels=np.append(train_labels,train_labels[i])
                
    new_occurance1_test = np.count_nonzero(train_labels)
    new_occurance0_test = np.count_nonzero(train_labels==0)
    print("After Oversampling")
    
    clf = SVC(C=1,kernel='rbf')
    clf.fit(d2_train_data,train_labels)
    pred_labels=clf.predict(d2_test_data)
    tn,fp,fn,tp=confusion_matrix(test_labels,pred_labels).ravel()
    print('True Positive:{} || False Positive:{}  '.format(tp,fp))
    print('False Negative:{} || True Negative:{}  '.format(fn,tn))
    print("Recall: {} ,Precision: {}".format((tp/(tp+fn)),(tp/(tp+fp))))
    print("Accuracy: {}".format((tp+tn)/(tp+tn+fp+fn)))
    #train data is oversampled 
    
    
def undersample_majority_class(train_data,train_labels,test_data,test_labels):
    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    occurance1_train = np.count_nonzero(train_labels)
    occurance0_train = np.count_nonzero(train_labels==0)
    difference = occurance1_train-occurance0_train*2 #it provide ratio is 2:1 
    
    length = len(train_labels)
    k=0
    i=length-1
    while(i>=0):
        if(train_labels[i]==1):
            d2_train_data = np.delete(d2_train_data,i,axis=0)
            train_labels = np.delete(train_labels,i)
            k=k+1
            if k==difference:
                break
        i=i-1
    new_occurance1_test = np.count_nonzero(train_labels) #for debug
    new_occurance0_test = np.count_nonzero(train_labels==0) #for debug
    print("After undersampling")
    
    clf = SVC(C=1,kernel='rbf')
    clf.fit(d2_train_data,train_labels)
    pred_labels=clf.predict(d2_test_data)
    tn,fp,fn,tp=confusion_matrix(test_labels,pred_labels).ravel()
    print('True Positive:{} || False Positive:{}  '.format(tp,fp))
    print('False Negative:{} || True Negative:{}  '.format(fn,tn))
    print("Recall: {} ,Precision: {}".format((tp/(tp+fn)),(tp/(tp+fp))))
    print("Accuracy: {}".format((tp+tn)/(tp+tn+fp+fn)))



def make_balance(train_data,train_labels,test_data,test_labels):
    nsamples, nx, ny = np.shape(train_data)
    d2_train_data = np.reshape(train_data,(nsamples,nx*ny))
    nsamples, nx, ny = np.shape(test_data)
    d2_test_data = np.reshape(test_data,(nsamples,nx*ny))
    clf = SVC(C=1,kernel='rbf',class_weight='balanced')
    clf.fit(d2_train_data,train_labels)
    pred_labels=clf.predict(d2_test_data)
    tn,fp,fn,tp=confusion_matrix(test_labels,pred_labels).ravel()
    print('True Positive:{} || False Positive:{}  '.format(tp,fp))
    print('False Negative:{} || True Negative:{}  '.format(fn,tn))
    print("Recall: {} ,Precision: {}".format((tp/(tp+fn)),(tp/(tp+fp))))
    print("Accuracy: {}".format((tp+tn)/(tp+tn+fp+fn)))
    #print('Test Accuracy for experimet 4 is {}'.format(clf.score(d2_test_data,test_labels)))
    


if __name__ == '__main__':
    #Experiment1
    '''
    train_data = np.load( 'hw3_data/linsep/train_data.npy' )
    train_labels = np.load('hw3_data/linsep/train_labels.npy')
    experiment1(train_data,train_labels)
    '''
    #Experiment2
    '''
    train_data2 = np.load('hw3_data/nonlinsep/train_data.npy')
    train_labels2 = np.load('hw3_data/nonlinsep/train_labels.npy')
    experiment2(train_data2,train_labels2)
    '''
    '''
    #Experiment3
    
    train_data3 = np.load ( 'hw3_data/fashion_mnist/train_data.npy')
    train_labels3 = np.load ( 'hw3_data/fashion_mnist/train_labels.npy')
    test_data3 = np.load ( 'hw3_data/fashion_mnist/test_data.npy')
    test_labels3 = np.load ('hw3_data/fashion_mnist/test_labels.npy')
    train_data3_normalize = train_data3/256
    test_data3_normalize = test_data3/256
    experiment3(train_data3_normalize,train_labels3,test_data3_normalize,test_labels3) #use it for find best
    #experiment3_manuel_grid_search(train_data3_normalize,train_labels3,test_data3_normalize,test_labels3) #use it optain all
    '''
    
    #Experiment4
    train_data4 = np.load ( 'hw3_data/fashion_mnist_imba/train_data.npy')
    train_labels4 = np.load ( 'hw3_data/fashion_mnist_imba/train_labels.npy' )
    test_data4 = np.load ( 'hw3_data/fashion_mnist_imba/test_data.npy' )
    test_labels4 = np.load ( 'hw3_data/fashion_mnist_imba/test_labels.npy')
    train_data4_normalize = train_data4/256
    test_data4_normalize = test_data4/256
    #experiment4(train_data4_normalize,train_labels4,test_data4_normalize,test_labels4)
    #my_confusion_matrix(train_data4_normalize,train_labels4,test_data4_normalize,test_labels4)
    #oversample_minority_class(train_data4_normalize,train_labels4,test_data4_normalize,test_labels4)
    #undersample_majority_class(train_data4_normalize,train_labels4,test_data4_normalize,test_labels4) 
    make_balance(train_data4_normalize,train_labels4,test_data4_normalize,test_labels4)
    