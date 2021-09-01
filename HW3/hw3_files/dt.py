from numpy import log2, maximum
import numpy as np
import numpy
from numpy.core.fromnumeric import argmax, size
import copy

from numpy.lib.polynomial import roots


class Node:

    def __init__(self, data,bucket,attribute):

        self.left = None
        self.right = None
        self.data = data
        self.bucket=bucket
        self.attribute = attribute
        #self.split_value = split_value

    def insert(self, data,bucket,attribute):
    # Compare the new value with the parent node
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data,bucket,attribute)
                else:
                    self.left.insert(data,bucket,attribute)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data,bucket,attribute)
                else:
                    self.right.insert(data,bucket,attribute)
        else:
            self.data = data
            self.bucket=bucket
            self.attribute=attribute 
            
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        check_bucket=self.bucket.count(0)
        if check_bucket < 2:
            print('split value: {}, bucket: [{}, {}, {}], attribute: {} '.format(self.data,self.bucket[0],self.bucket[1],self.bucket[2],self.attribute))
        else:
            print('LEAVE! bucket: [{}, {}, {}], attribute: {} '.format(self.bucket[0],self.bucket[1],self.bucket[2],self.attribute))
        if self.right:
            self.right.PrintTree()








def entropy(bucket):
    total = sum(bucket)
    if(total==0):
        return 0
    E=0
    for i in range(len(bucket)):
        p=bucket[i]/total
        if(p!=0):
            E+=(-p*log2(p))
    return E
    
    
    
    
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """


def info_gain(parent_bucket, left_bucket, right_bucket):
    
    parent_ent = entropy(parent_bucket)
    left_ent = entropy(left_bucket)
    right_ent = entropy(right_bucket)
    example_size = np.sum(parent_bucket)
    gain = parent_ent - (np.sum(left_bucket) / example_size) * left_ent - (
                np.sum(right_bucket) / example_size) * right_ent
    return gain
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """


def gini(bucket):
    
    total = sum(map(lambda i : i * i, bucket))/(sum(bucket)**2) 
    gini_index = 1-total
    
    return gini_index
    
    
    
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """


def avg_gini_index(left_bucket, right_bucket):
    
    left_gini=gini(left_bucket)
    right_gini=gini(right_bucket)
    size_left = sum(left_bucket)
    size_right=sum(right_bucket)
    total_size=size_left+size_right
    
    avg = left_gini*(size_left/total_size)+right_gini*(size_right/total_size)
    return avg
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
def label_calculator(labels,num_class):
    bucket = np.zeros(num_class)
    for i in range(len(labels)):
        bucket[labels[i]]+=1 
    return bucket
def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    out =[]
    attribute_list=copy.deepcopy(data[:,attr_index])
    attribute_list = np.sort(attribute_list)
    left_bucket=[]
    label_left=[]
    right_bucket=[]
    label_right=[]
    for i in range(len(attribute_list)-1):
        left_bucket=[]
        right_bucket=[]
        label_left=[]
        label_right=[]
        x=(attribute_list[i]+attribute_list[i+1])/2
        for j in range(len(data)):
            if(data[j][attr_index]>=x):
                right_bucket.append(data[j])
                label_right.append(labels[j])
            else:
                left_bucket.append(data[j])
                label_left.append(labels[j])
        parent=label_calculator(labels,num_classes)
        right=label_calculator(label_right,num_classes)
        left=label_calculator(label_left,num_classes)
        #pritn(left_bucket)
        #pritn(right_bucket)
        #pritn(label_left)
        #pritn(label_right)
        if heuristic_name =='info_gain':
            out.append([x,info_gain(parent,left,right)])
        else:
            out.append([x,avg_gini_index(left,right)])
       
                
    
    return out
        
    
    #print(np.shape(data))
    #print(np.shape(labels))
    #print(num_classes)
    #print(attr_index)
    #print(heuristic_name)
    print('dikkat')
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """


def chi_squared_test(left_bucket, right_bucket):
    n_l = sum(left_bucket)
    n_r = sum(right_bucket)
    total = n_l + n_r
    n_j = np.zeros(len(left_bucket))
    for i in range(len(left_bucket)):
        n_j[i]=left_bucket[i]+right_bucket[i]
     
    table = np.zeros((2,len(left_bucket)))
    obj = np.array([left_bucket,right_bucket])
    x1=n_l/total
    x2=n_r/total
    table[0] = np.multiply(n_j,x1)
    table[1] = np.multiply(n_j,x2)
    value=0
    for i in range(2):
        for j in range(len(left_bucket)):
            if(table[i][j]!=0):
                value+=((obj[i][j]-table[i][j])**2)/table[i][j]
    zero_colum = np.count_nonzero(n_j==0)
    
    degree_of_freedom=(2-1)*(len(left_bucket)-1-zero_colum)
    return value,degree_of_freedom
    #print("check")
    
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
def find_max(out0,out1,out2,out3):
    maximum=-1
    attribute=0
    split_point=0
    split_index=0
    #print(len(out0))
    for j in range(len(out0)):
        if(out0[j][1]>maximum):
            #print(out0[j][1])
            maximum = out0[j][1]
            attribute=0
            split_point=out0[j][0]
            split_index=j
        if(out1[j][1]>maximum):
            maximum = out1[j][1]
            attribute=1
            split_point=out1[j][0]
            split_index=j
        if(out2[j][1]>maximum):
            maximum = out2[j][1]
            attribute=2
            split_point=out2[j][0]
            split_index=j
        if(out3[j][1]>maximum):
            maximum = out3[j][1]
            attribute=3
            split_point=out3[j][0]
            split_index=j
    return maximum,attribute,split_point,split_index


def find_min(out0,out1,out2,out3):
    maximum=float('inf')
    attribute=0
    split_point=0
    split_index=0
    #print(len(out0))
    for j in range(len(out0)):
        if(out0[j][1]<maximum):
            #print(out0[j][1])
            maximum = out0[j][1]
            attribute=0
            split_point=out0[j][0]
            split_index=j
        if(out1[j][1]<maximum):
            maximum = out1[j][1]
            attribute=1
            split_point=out1[j][0]
            split_index=j
        if(out2[j][1]<maximum):
            maximum = out2[j][1]
            attribute=2
            split_point=out2[j][0]
            split_index=j
        if(out3[j][1]<maximum):
            maximum = out3[j][1]
            attribute=3
            split_point=out3[j][0]
            split_index=j
    return maximum,attribute,split_point,split_index

def data_split(train_data,train_label,attribute,split_point):
    k=0
    l=0
    right_data=[]
    left_data = []
    right_label=[]
    left_label=[]
    print(split_point)
    print(attribute)
    for i in range(len(train_label)):
        if train_data[i][attribute]>=split_point:
            right_data.append(train_data[i])
            right_label.append(train_label[i])
        else:
            left_data.append(train_data[i])
            left_label.append(train_label[i])
    
    x1=numpy.array(right_data)
    x2=numpy.array(left_data)
    x3=numpy.array(right_label)
    x4=numpy.array(left_label)
    return x1,x2,x3,x4

def ID3(train_data,train_labels,heuirtic_name,num_classes):
    ninty_percent=[0,2.71,4.61,6.25,7.78,9.24,10.6,12,13.4,14.7,16,17.3,18.5,19.8,21.1,22.3]
    
    
    bucket=[0,0,0]
    bucket[0]=np.count_nonzero(train_labels==0)
    bucket[1]=np.count_nonzero(train_labels==1)
    bucket[2]=np.count_nonzero(train_labels==2)
    print(bucket)
    out0=calculate_split_values(train_data,train_labels,num_classes,0,heuirtic_name)
    out1=calculate_split_values(train_data,train_labels,num_classes,1,heuirtic_name)
    out2=calculate_split_values(train_data,train_labels,num_classes,2,heuirtic_name)
    out3=calculate_split_values(train_data,train_labels,num_classes,3,heuirtic_name)
    if heuirtic_name=='info_gain':
        maximum, attribute,  split_point, split_index = find_max(out0,out1,out2,out3)
    else:
        maximum, attribute,  split_point, split_index = find_min(out0,out1,out2,out3)
    
    root = Node(split_point,bucket,attribute)
    check_bucket=bucket.count(0)
    if check_bucket >= 2:
        return root
    else:
   
        data_right, data_left,right_label,left_label = data_split(train_data,train_labels,attribute,split_point) 
        left_bucket=label_calculator(left_label,num_classes)
        right_bucket=label_calculator(right_label,num_classes)
        '''
        #without prununing
        root.left = ID3(data_left,left_label,heuirtic_name,num_classes)
        root.right = ID3(data_right,right_label,heuirtic_name,num_classes)
        return root
        '''        
        
        #with pruning
        xx, df=chi_squared_test(left_bucket,right_bucket)
        if(xx>ninty_percent[df]):
            root.left = ID3(data_left,left_label,heuirtic_name,num_classes)
            root.right = ID3(data_right,right_label,heuirtic_name,num_classes)
            return root
        else:
            return root
        
    #print('split_point: {}'.format(split_point))
    
    
        
      
    #print('check')
    
def prediction(data,root):

    
    if(root.left==None and root.right==None):
        
        #res= [i for i, element in enumerate(root.bucket) if element!=0]
        #res=res[0] without pruning
        return root.bucket
    else:
        if(data[root.attribute]<root.data):
            return prediction(data,root.left)
        else:
            return prediction(data,root.right)

            
        
def accuracy_calculator(test_data,test_labels,root):
    i=0
    accuracy=0
    for test in test_data:
        pred=prediction(test,root)
        pred=np.argmax(pred)
        if(pred==test_labels[i]):
            accuracy=accuracy+1
        i=i+1
    accuracy=accuracy/len(test_labels)
    return accuracy

    
    
if __name__ == '__main__':
    
    train_data = np.load( 'hw3_data/iris/train_data.npy')
    train_labels = np.load( 'hw3_data/iris/train_labels.npy')
    test_data = np.load( 'hw3_data/iris/test_data.npy')
    test_labels = np.load( 'hw3_data/iris/test_labels.npy' )
    
    #root=ID3(train_data,train_labels,'info_gain',3)
    root=ID3(train_data,train_labels,'avg_gini_index',3)
    print('printing tree')
    root.PrintTree()
    
    #test 
    
    acc=accuracy_calculator(test_data,test_labels,root)
    print('Test_accuracy for avg_gini_index with pruning: {}'.format(acc))