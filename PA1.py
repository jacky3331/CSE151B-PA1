##########################################################################################
# Authors: Jacky Dam and Ali Zaidi
# Making Better Decisions with Logistic Regression and the Softmax Classifier in Python
# CSE 151B: PA1
# Date: Oct 18, 2020
##########################################################################################

import random as rand
import numpy as np
import matplotlib.pyplot as plt


from fashion_mnist_dataset.utils import mnist_reader
X_train, y_train = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind='t10k')

################################################################
# Tools for LRGD and SMGD:
################################################################
def shuffle_data(dataSetX,dataSetY):
    my_dataX=np.copy(dataSetX)  
    my_dataY=np.copy(dataSetY)      
    shuffleIndex=np.random.permutation(len(my_dataX)) # get random permutation of sequence of numbers

    shuffled_dataX=my_dataX[shuffleIndex] # shuffle data based on shuffleIndex
    shuffled_dataY=my_dataY[shuffleIndex]     

    return list(zip(shuffled_dataX,shuffled_dataY))  

def selective_dataset(dataSet, firstClass, secondClass):
    return [d for d in dataSet if (np.array_equal(d[1],firstClass) or np.array_equal(d[1],secondClass))] # contain images belonging only to firstClass and secondClass

def LRkFoldSplit(dataSetX,dataSetY,folds,firstClass,secondClass): # Logistic regression kFold method
    dataSetSplit=[]     
    dataSet=shuffle_data(dataSetX,dataSetY) 
    data=selective_dataset(dataSet,firstClass,secondClass)
    length_of_data=len(data)  
    foldSize=len(data)/folds   # get foldSize for folding the dataSet

    for i in range(folds):    
        temp=[]
        while len(temp) < foldSize:  # split data based on foldSize
            temp.append(data.pop())   
        dataSetSplit.append(temp)  

    return dataSetSplit,length_of_data

def normalize(dataSet): # min-max normalize (x-min)/(max-min)
    my_data=np.copy(dataSet)
    min=np.min(my_data, axis=0) 
    max=np.max(my_data, axis=0)     

    for index in range(len(min)):
        if min[index]==max[index]: # to prevent divison by zero, remove this column since all the data is the same (min == max) 
            removeIndex=index
            max=np.delete(max,removeIndex,0) # also remove this max from max vector if same
            min=np.delete(min,removeIndex,0) # also remove this min from min vector if same
            my_data=np.delete(my_data,removeIndex,1)

    for vector in range(len(my_data)):
        my_data[vector] = (my_data[vector]-min)/(max-min) # normalize
    
    return my_data

def PCA(dataSet,p):
    A=normalize(dataSet) # normalize data
    features = np.transpose(dataSet) 
    M=np.mean(features, axis=1) # get mean

    C=A-M # mean center the data

    cov_matrix = np.cov(C, rowvar=False) # get covariance matrix
    values, vectors = np.linalg.eigh(cov_matrix)  # get the eigenvalues and eigenvectors

    index=np.argsort(values)[::-1] # sort from greatest to least
    values=values[index]
    vectors=vectors[:,index]

    pTopPC=np.transpose(np.take(vectors, np.arange(p),axis=0)) # sort the eigenvectors

    return pTopPC

def hotEncode(dataSet):
    temp=[0,0,0,0,0,0,0,0,0,0]
    new_data=[]

    for item in dataSet:
        temp[item] = 1 # put a one for the corresponding class
        new_data.append(temp)
        temp=[0,0,0,0,0,0,0,0,0,0] # reset back to zeros vector
    return new_data

def augmentMatrix(dataSet):
    my_data=np.copy(dataSet)

    return np.insert(my_data,0,1.0, axis=1) # insert a one for the bias

def projectImage(PCA,dataSet):
    my_data=[]
    for image in dataSet:
        projection=np.matmul(image,PCA) # calculate the projection
        my_data.append(projection)
    
    return my_data


################################################################
#Logistic Regression
################################################################
def LRlossFunction(target, output):
    if output==0: 
        output+=10**(-8) # to prevent log(0)
    elif output==1:
        output -= 10**(-8)  # to prevent log(0)

    return -(target*np.log(output)+(1-target)*np.log(1-output))

def target(input,firstClass):
    if np.array_equal(input, firstClass): # if same target then return 1 otherwise 0
        return 1
    else:
        return 0

def LRprediction(weights, image):
    yHat=1/(1+np.exp(-np.matmul(np.transpose(weights),image))) # using the LR prediction formula
    if yHat>=0.5:
        return 1
    else:
        return 0

def LRalgorithmAccuracy(val_set, weights, firstClass, secondClass):
    actual=[] # y
    guess=[] # y hat
    count=0 # correct count
    for example in val_set:
        if np.array_equal(example[1],firstClass) or np.array_equal(example[1],secondClass): # only select images from firstClass and secondClass
            actual.append(target(example[1],firstClass))
            yHat=LRprediction(weights, example[0])
            guess.append(yHat) # append the model prediction

    
    for index in range(len(guess)):
        if guess[index]==actual[index]:
            count+=1
    return count/len(guess)*100.0 # accuracy percentage

def LRGradientDescent(val_set, dataSet, learning_rate, M_epochs, batch_size, trueClass, secondClass):
    weights = [0.0 for i in range(len(dataSet[0][0]))] # reset the weights to zero
    netLoss=[] # loss between epochs
    counter=0 
    loss_per_epoch =[]
    loss_per_epoch_val = []

    for epoch in range(M_epochs):
        loss=[] # loss per image of training
        val_loss = []  # loss per image of validation
        i = 0
        for j in range(1,len(dataSet),batch_size):

            yHat = LRprediction(weights,dataSet[j][0]) # get prediction for training image
            error = target(dataSet[j][1], trueClass)-yHat # get error for training image
            loss.append(LRlossFunction(target(dataSet[j][1], trueClass),yHat)) # append the loss
            weights = weights+learning_rate*dataSet[j][0]*error # calculate the weights

            if (len(val_set[i]) != 0 and i < j): # to prevent overflowing in the valset
                i += 1
                yValHat = LRprediction(weights, val_set[i][0]) # get prediction for val image
                errorVal = target(val_set[i][1], trueClass)-yValHat  # get error for val image
                val_loss.append(LRlossFunction(target(val_set[i][1], trueClass), yValHat)) # append the val loss

        netLoss.append(sum(loss))  # used for Early Stop (if loss keeps on increasing, stop after counter times)
        loss_per_epoch.append(sum(loss)) # add up training loss for batch_size of data
        loss_per_epoch_val.append(sum(val_loss)) # add up val loss for batch_size of data

        # if loss is not improving then increase the counter, otherwise reset it  
        if len(netLoss) >= 2 and abs(netLoss[-1]-netLoss[-2]) < 1:
            counter+=1
        else:
            counter=0

        # if loss doesn't improve after 4 epoch, early stop
        if counter==4:
            break
    
    return weights, loss_per_epoch, loss_per_epoch_val 

def LRcrossValidation(dataSetX ,dataSetY ,testSetX ,testSetY ,p ,k ,learning_rate, M_epochs, batch_size, firstClass, secondClass):
    
    PCA_Calculation=PCA(dataSetX,p) # perform PCA on dataSet
    projected_Xset=projectImage(PCA_Calculation, dataSetX) # get the project of the images

    augX_Set=augmentMatrix(projected_Xset) # augment the matrix (add the bias)
    hotEncodeY_Set=hotEncode(dataSetY) # do one hot encoding on the y labels
    folds,length_of_data=LRkFoldSplit(augX_Set,hotEncodeY_Set,k,firstClass,secondClass)  # do the k folds also combine the dataset and y labels [(image,label), (image,label)...]

    # do everything above for the test set
    projected_Testset=projectImage(PCA_Calculation, testSetX) 
    augTest_Set=augmentMatrix(projected_Testset)
    hotEncodeTest_Set=hotEncode(testSetY)
    TestSet=shuffle_data(augTest_Set,hotEncodeTest_Set)

    foldModelPerformance=[] # store the performance per fold
 
    loss_per_fold=[] # train loss per fold
    loss_per_fold_val = [] # val loss per fold
    
    for fold in range(k): 
        val_set=folds[fold] # val set
        train_set=[]

        temp=folds[:fold] + folds[fold+1:] # the rest are train
        for set in temp:
            train_set+=set

        modelPerformance = []  # model performance

        weights, loss_per_epoch_trimmed, loss_per_epoch_val_trimmed = LRGradientDescent(val_set, train_set, learning_rate, M_epochs, batch_size, firstClass, secondClass) # get the weights and loss of val and train per fold
        modelPerformance.append([LRalgorithmAccuracy(val_set, weights, firstClass, secondClass),weights]) # calculate the accuracy based on prediction
        
        loss_per_fold.append(loss_per_epoch_trimmed) # store the train loss per fold
        loss_per_fold_val.append(loss_per_epoch_val_trimmed) # store the val loss per fold
        foldModelPerformance += modelPerformance

    # get the best performance with corresponding weights 
    temp_max=[]
    weights=[]
    for item in foldModelPerformance:
        temp_max.append(item[0])
        weights.append(item[1])
    max_index=np.argsort(temp_max)[0]


    # Error
    avg_loss = [] 
    epoch_count = []
    min_fold_len = len(min(loss_per_fold, key=len)) # find the min epoch over all the folds

    loss_per_fold_trimmed=[] 
    for arr in loss_per_fold:
        loss_per_fold_trimmed.append(arr[:min_fold_len]) # all the folds epoch should have the same length (k_min == min_fold_len)

    counter = 1 # number of epochs
    for arr in np.transpose(loss_per_fold_trimmed):
        sum_of_column = sum(arr) # sum up over the epochs
        avg_of_column = sum_of_column/len(arr) # get the avg over epochs
        avg_loss.append(avg_of_column/(length_of_data/(k)))
        epoch_count.append(counter)
        counter+=1

    #Val Error
    val_avg_loss = []
    loss_per_fold_val_trimmed = []

    for arr in loss_per_fold_val:
        loss_per_fold_val_trimmed.append(arr[:min_fold_len]) # use same min epochs for val set, so graphing is consistent between val and train set

    for arr in np.transpose(loss_per_fold_val_trimmed):
        sum_of_column = sum(arr)  # sum up over the epochs
        avg_of_column = sum_of_column/len(arr)  # get the avg over epochs
        val_avg_loss.append(avg_of_column/(length_of_data/(k)))


    plt.errorbar(epoch_count, avg_loss, label='Train line')
    plt.errorbar(epoch_count, val_avg_loss, label='Val line')
    plt.title('Epoch vs Loss Function (PC = 250)')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss Function')
    plt.legend()
    plt.show()
    
    print(LRalgorithmAccuracy(TestSet, foldModelPerformance[max_index][1], firstClass, secondClass)) # print Accuracy over TestSet
    return foldModelPerformance[max_index][1]

    
################################################################
#SoftMax Regression:
################################################################
def SMkFoldSplit(dataSetX,dataSetY,folds): # fold for Softmax
    dataSetSplit=[]     
    data=shuffle_data(dataSetX,dataSetY) # suffle data before fold      
    foldSize=len(data)/folds # fold division
    
    for i in range(folds):    
        temp=[]
        while len(temp) < foldSize:  # split data based on foldSize
            temp.append(data.pop())   
        dataSetSplit.append(temp)    # add the fold data back to dataSet Split
    return dataSetSplit

def SMprediction(k_weights, image): # Stable Softmax
    netInput = []
    num_of_classes = 10
    probabilityList = np.zeros(num_of_classes) # make the probabilityList for prediction
    for index in range(len(k_weights)):
        netInput.append(np.matmul(np.transpose(k_weights[index]), image)) # get the probability with k_weight and image

    totalInput = np.sum(np.exp(netInput-np.max(netInput)))  # stable Softmax denimnator
    # totalInput = np.sum(np.exp(netInput))

    for index in range(len(probabilityList)):
        probabilityList[index]=np.exp(netInput[index]-np.max(netInput))/totalInput
        # probabilityList[index] = np.exp(netInput[index])/totalInput

    return probabilityList

def SMlossFunction(target, output, batch_size):
    for index in range(len(output)):
        if output[index]==0:
            output[index]+=10**(-8) # prevent log(0)
        elif output[index]==1:
            output[index] -= 10**(-8)  # prevent log(0)
    
    return -np.sum(target*np.log(output))

def SMdecision(target, yHat):
    # targetIndex = np.argmax(target)
    # error = (target-yHat)[targetIndex]
    error = (target-yHat)  # the error difference of target - prediction
    return error
    # [1,0,0,0,0]
    # [0,1,0,0,0]
    # [1,-1,0,0,0]
    
def SMGradientDescent(val_set, dataSet, k_weights, learning_rate, M_epochs, batch_size, firstClass):
    counter = 0 # epoch counter
    netLoss=[] # newLoss over batchsize

    loss_per_epoch =[]
    loss_per_epoch_val = []
    for epoch in range(M_epochs):
        loss=[]
        val_loss = []
        i = 0
        
        for j in range(1,len(dataSet), batch_size):
        
            yHat = SMprediction(k_weights, dataSet[j][0]) # model prediction (Train)
            error = SMdecision(dataSet[j][1], yHat) # error (Train)
            loss.append(SMlossFunction(dataSet[j][1],yHat,batch_size)) # add the loss

            
            if(len(val_set[i])!=0) and i<j:
                i+=1
                yValHat = SMprediction(k_weights, val_set[i][0]) # model prediction (Val)
                errorVal = SMdecision(val_set[i][1], yValHat) # error (Val)
                val_loss.append(SMlossFunction(val_set[i][1], yValHat, batch_size))


            for index in range(len(error)):
                k_weights[index] = k_weights[index] + learning_rate*dataSet[j][0]*error[index] # update the k_weight 


        netLoss.append(sum(loss))
        loss_per_epoch.append(sum(loss)) # add the loss epoch (Train)
        loss_per_epoch_val.append(sum(val_loss))  # add the loss epoch (Train)

        # if loss is not improving then increase the counter, otherwise reset it  
        if len(netLoss)>=2 and abs(netLoss[-1]-netLoss[-2])<1:
            counter+=1
        else:
            counter=0

        # if loss doesn't improve after 6 epoch, early stop
        if counter==6:
            break

    # return weights and loss for train and val
    return k_weights, loss_per_epoch, loss_per_epoch_val

def SMalgorithmAccuracy(val_set, weights):
    actual = []  # y
    guess = []  # y hat
    count = 0
    for example in val_set:
        actual.append(np.argmax(example[1]))
        yHat=np.argmax(SMprediction(weights, example[0]))
        guess.append(yHat) # append the model prediction
    
    for index in range(len(guess)):
        if guess[index]==actual[index]:
            count+=1
    return count/len(guess)*100.0  # accuracy percentage

def SMcrossValidation(dataSetX, dataSetY, testSetX, testSetY, p, k, learning_rate, M_epochs, batch_size):
    
    PCA_Calculation = PCA(dataSetX, p)  # run PCA over dataSet (also does normalize)
    projected_Xset=projectImage(PCA_Calculation, dataSetX) # project images for PCA_Calculation
    augX_Set=augmentMatrix(projected_Xset) # add bias 1
    hotEncodeY_Set=hotEncode(dataSetY)
    folds=SMkFoldSplit(augX_Set, hotEncodeY_Set, k) # create the k folds
    
    # do the same for
    projected_Testset=projectImage(PCA_Calculation, testSetX)
    augTest_Set=augmentMatrix(projected_Testset)
    hotEncodeTest_Set=hotEncode(testSetY)
    TestSet=shuffle_data(augTest_Set,hotEncodeTest_Set)

    foldModelPerformance=[] # performance per Fold
    num_of_classes=10
    loss_per_fold=[]
    loss_per_fold_val =[]
    
    for fold in range(k):
        val_set=folds[fold]
        train_set=[]
        k_vector=[]

        temp=folds[:fold] + folds[fold+1:]
        for set in temp:
            train_set+=set
        
        weights = [0.0 for i in range(len(augX_Set[0]))] # initialize the weights to zero 
        k_weights = [] # weight for each class
        for classification in range(num_of_classes):
            k_weights.append(weights)

        modelPerformance=[]
        weights, loss_per_epoch_trimmed, loss_per_epoch_val_trimmed = SMGradientDescent(val_set,
        train_set, k_weights, learning_rate, M_epochs, batch_size, k_vector)  # run SMGradientDescent over the val_set and train_set 
        
        modelPerformance.append([SMalgorithmAccuracy(val_set, weights), weights]) # append the Performance from each fold
        
        loss_per_fold.append(loss_per_epoch_trimmed)  # the trimmed version will contain the min num of epochs per fold (train)
        loss_per_fold_val.append(loss_per_epoch_val_trimmed) # the trimmed version will contain the min num of epochs per fold (val)
        foldModelPerformance+=modelPerformance

    temp_max=[]
    weights=[]
    for item in foldModelPerformance:
        temp_max.append(item[0])
        weights.append(item[1])

    max_index=np.argmax(temp_max) # get the best Performance index


    # Error
    avg_loss = []
    epoch_count = []
    min_fold_len = len(min(loss_per_fold, key=len)) # find the min epoch over all the folds

    loss_per_fold_trimmed = []
    for arr in loss_per_fold:
        loss_per_fold_trimmed.append(arr[:min_fold_len])  # all the folds epoch should have the same length (k_min == min_fold_len)

    counter = 0 # number of epochs
    for arr in np.transpose(loss_per_fold_trimmed):
        sum_of_column = sum(arr)  # sum up over the epochs
        avg_of_column = sum_of_column/len(arr) # get the avg over epochs
        avg_loss.append(avg_of_column/(len(dataSetX)/(k)))
        epoch_count.append(counter)
        counter += 1

    #Val Error
    val_avg_loss = []
    loss_per_fold_val_trimmed = []

    for arr in loss_per_fold_val:
        loss_per_fold_val_trimmed.append(arr[:min_fold_len]) # use same min epochs for val set, so graphing is consistent between val and train set

    for arr in np.transpose(loss_per_fold_val_trimmed):
        sum_of_column = sum(arr)
        avg_of_column = sum_of_column/len(arr)  # get the avg over epochs
        val_avg_loss.append(avg_of_column/(len(dataSetX)/(k)))

    plt.errorbar(epoch_count, avg_loss, label='Train line')
    plt.errorbar(epoch_count, val_avg_loss, label='Val line')
    plt.title('Epoch vs Loss Function (PC = 250)')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss Function')
    plt.legend()
    plt.show()

    print(SMalgorithmAccuracy(TestSet, foldModelPerformance[max_index][1]))
    return foldModelPerformance[max_index][1]

###########################################################
#SoftMax Testing: (~73% accuracy)
###########################################################
# test_weights=SMcrossValidation(X_train, y_train, X_test, y_test, 784, 10, 0.01, 100, 100)
x=PCA(X_train,784)
w=LRcrossValidation(X_train, y_train, X_test, y_test, 784, 10, 0.01, 100, 16, [1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1])

testimage=np.matmul(np.matmul(w[1:],x),np.transpose(x)).reshape(28,28)
plt.figure()
plt.imshow(testimage)
plt.colorbar()
plt.grid(False)
plt.title('Weights Visualization for Softmax Regression')
plt.show()

# testimage=np.matmul(np.matmul(test_weights[1][1:],x),np.transpose(x)).reshape(28,28)
# plt.figure()
# plt.imshow(testimage)
# plt.colorbar()
# plt.grid(False)
# plt.title('Weights Visualization for Softmax Regression')
# plt.show()

# testimage=np.matmul(np.matmul(test_weights[2][1:],x),np.transpose(x)).reshape(28,28)
# plt.figure()
# plt.imshow(testimage)
# plt.colorbar()
# plt.grid(False)
# plt.title('Weights Visualization for Softmax Regression')
# plt.show()

# testimage=np.matmul(np.matmul(test_weights[3][1:],x),np.transpose(x)).reshape(28,28)
# plt.figure()
# plt.imshow(testimage)
# plt.colorbar()
# plt.grid(False)
# plt.title('Weights Visualization for Softmax Regression')
# plt.show()

# testimage=np.matmul(np.matmul(test_weights[4][1:],x),np.transpose(x)).reshape(28,28)
# plt.figure()
# plt.imshow(testimage)
# plt.colorbar()
# plt.grid(False)
# plt.title('Weights Visualization for Softmax Regression')
# plt.show()

# testimage=np.matmul(np.matmul(test_weights[5][1:],x),np.transpose(x)).reshape(28,28)
# plt.figure()
# plt.imshow(testimage)
# plt.colorbar()
# plt.grid(False)
# plt.title('Weights Visualization for Softmax Regression')
# plt.show()
###########################################################
#Logistic Regression Testing:
###########################################################
#T-Shirts vs. Ankle Boots (~95% accuracy)
# LRcrossValidation(X_train, y_train, X_test, y_test, 784, 10, 0.01, 100, 16, [1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1])

#Pullovers vs. Coats (~70% accuracy)
# LRcrossValidation(X_train, y_train, X_test, y_test, 784, 10, 0.01, 100, 16, [0,0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0])
