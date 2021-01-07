import pandas as pd
import numpy as np
train=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/train.csv")
campdata=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/campaign_data.csv")
coupon=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/coupon_item_mapping.csv")
custdemo=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/customer_demographics.csv")
custtran=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/customer_transaction_data.csv")
itemdata=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/item_data.csv")
test=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/test.csv")


import os
os.chdir("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a")
train_new=pd.read_csv('train4_1.csv')
test_new=pd.read_csv("test4_1.csv")
ans_train=pd.read_csv('train4_y.csv')

#campdata.info()
#campdata["start_date"]
#campdata["end_date"]

train1=pd.merge(train,campdata,how="left",on='campaign_id')#78,369
train2=pd.merge(train1,coupon,how="left",on='coupon_id')#6,42,0694
train3=pd.merge(train2,itemdata,how="left",on='item_id')#6,42,0694

train_im1=pd.merge(custtran,itemdata,how="left",on='item_id')#6,42,0694


disc=[]
odisc=[]
totalsales=[]

for i in np.sort((np.unique(train_im1.customer_id))):
    disc.append(np.sum(train_im1.coupon_discount[train_im1.customer_id==i]))
    odisc.append(np.sum(train_im1.other_discount[train_im1.customer_id==i]))
    totalsales.append(np.sum(train_im1.quantity[train_im1.customer_id==i]*train_im1.selling_price[train_im1.customer_id==i]))
    if(i%100==0):
        print(i)

train_im2=pd.concat([pd.DataFrame(np.unique(train_im1.customer_id)),pd.DataFrame(disc),pd.DataFrame(odisc),pd.DataFrame(totalsales)],axis=1)
train_im2.columns=['customer_id','disc','odisc','totalsales']
train4=pd.merge(train3,train_im2,how='left',on='customer_id')
#train4=pd.merge(train3,train_im1,how="left",on=['item_id','customer_id'])

catdisc=[]
catodisc=[]
cattotalsales=[]

for i in np.sort((np.unique(train_im1.category))):
    catdisc.append(np.sum(train_im1.coupon_discount[train_im1.category==i]))
    catodisc.append(np.sum(train_im1.other_discount[train_im1.category==i]))
    cattotalsales.append(np.sum(train_im1.quantity[train_im1.category==i]*train_im1.selling_price[train_im1.category==i]))

train_im3=pd.concat([pd.DataFrame(np.unique(train_im1.category)),pd.DataFrame(catdisc),pd.DataFrame(catodisc),pd.DataFrame(cattotalsales)],axis=1)
train_im3.columns=['category','catdisc','catodisc','cattotalsales']

train5=pd.merge(train4,train_im3,how='left',on='category')

bdisc=[]
bodisc=[]
btotalsales=[]


for i in np.sort((np.unique(train_im1.brand))):
    bdisc.append(np.sum(train_im1.coupon_discount[train_im1.brand==i]))
    bodisc.append(np.sum(train_im1.other_discount[train_im1.brand==i]))
    btotalsales.append(np.sum(train_im1.quantity[train_im1.brand==i]*train_im1.selling_price[train_im1.brand==i]))
    if(i%100==0):
        print(i)

train_im4=pd.concat([pd.DataFrame(np.unique(train_im1.brand)),pd.DataFrame(bdisc),pd.DataFrame(bodisc),pd.DataFrame(btotalsales)],axis=1)
train_im4.columns=['brand','bcatdisc','bodisc','btotalsales']

train6=pd.merge(train5,train_im4,how='left',on='brand')

train6_obj=pd.concat([train6.iloc[:,5],train6.iloc[:,10:12]],axis=1)
train6_obj_final=pd.get_dummies(train6_obj)
train6_num=pd.concat([train6.iloc[:,12:]],axis=1)
train6_final1=pd.concat([train6_num,train6_obj_final],axis=1)
train6_y=train6.iloc[:,4]

del train6_final1['category_Restauarant']


#train6_final=(train6_final1-train6_final1.mean())/train6_final1.std()
train6_final=train6_final1
'''
train6_final.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train6_final.csv',index=False)
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train6_final,train6_y, test_size=0.2)
parameters = model(len(train6_final.columns),np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))


test1=pd.merge(test,campdata,how="left",on='campaign_id')#78,369
test2=pd.merge(test1,coupon,how="left",on='coupon_id')#6,42,0694
test3=pd.merge(test2,itemdata,how="left",on='item_id')#6,42,0694

test_im1=pd.merge(custtran,itemdata,how="left",on='item_id')#6,42,0694


disc=[]
odisc=[]
totalsales=[]

for i in np.sort((np.unique(test_im1.customer_id))):
    disc.append(np.sum(test_im1.coupon_discount[test_im1.customer_id==i]))
    odisc.append(np.sum(test_im1.other_discount[test_im1.customer_id==i]))
    totalsales.append(np.sum(test_im1.quantity[test_im1.customer_id==i]*test_im1.selling_price[test_im1.customer_id==i]))
    if(i%100==0):
        print(i)

test_im2=pd.concat([pd.DataFrame(np.unique(test_im1.customer_id)),pd.DataFrame(disc),pd.DataFrame(odisc),pd.DataFrame(totalsales)],axis=1)
test_im2.columns=['customer_id','disc','odisc','totalsales']
test4=pd.merge(test3,test_im2,how='left',on='customer_id')
#test4=pd.merge(test3,test_im1,how="left",on=['item_id','customer_id'])

catdisc=[]
catodisc=[]
cattotalsales=[]

for i in np.sort((np.unique(test_im1.category))):
    catdisc.append(np.sum(test_im1.coupon_discount[test_im1.category==i]))
    catodisc.append(np.sum(test_im1.other_discount[test_im1.category==i]))
    cattotalsales.append(np.sum(test_im1.quantity[test_im1.category==i]*test_im1.selling_price[test_im1.category==i]))

test_im3=pd.concat([pd.DataFrame(np.unique(test_im1.category)),pd.DataFrame(catdisc),pd.DataFrame(catodisc),pd.DataFrame(cattotalsales)],axis=1)
test_im3.columns=['category','catdisc','catodisc','cattotalsales']

test5=pd.merge(test4,test_im3,how='left',on='category')


bdisc=[]
bodisc=[]
btotalsales=[]


for i in np.sort((np.unique(test_im1.brand))):
    bdisc.append(np.sum(test_im1.coupon_discount[test_im1.brand==i]))
    bodisc.append(np.sum(test_im1.other_discount[test_im1.brand==i]))
    btotalsales.append(np.sum(test_im1.quantity[test_im1.brand==i]*test_im1.selling_price[test_im1.brand==i]))
    if(i%100==0):
        print(i)

test_im4=pd.concat([pd.DataFrame(np.unique(test_im1.brand)),pd.DataFrame(bdisc),pd.DataFrame(bodisc),pd.DataFrame(btotalsales)],axis=1)
test_im4.columns=['brand','bcatdisc','bodisc','btotalsales']

test6=pd.merge(test5,test_im4,how='left',on='brand')




test6_obj=pd.concat([test6.iloc[:,4],test6.iloc[:,9:11]],axis=1)
test6_obj_final=pd.get_dummies(test6_obj)
test6_num=pd.concat([test6.iloc[:,11:]],axis=1)
test6_final1=pd.concat([test6_num,test6_obj_final],axis=1)

test6_final=test6_final1
#test6_final=(test6_final1-train6_final1.mean())/train6_final1.std()
'''
test6_final.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/test6_final.csv',index=False)
'''

##NN below
import tensorflow as tf
from tensorflow.python.framework import ops

'''
Here's what's happening: When you specify the operations needed for a computation, you are telling TensorFlow how 
to construct a computation graph. The computation graph can have some placeholders whose values you will specify 
only later. Finally, when you run the session, you are telling TensorFlow to execute the computation graph.
'''

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32,shape=(n_x,None),name="Placeholder_1")
    Y = tf.placeholder(tf.float32,shape=(n_y,None),name="Placeholder_2")
    
    return X, Y

def initialize_parameters(szip):
        
#        tf.set_random_seed(1)                   # so that your "random" numbers match ours
            
        W1 = tf.get_variable("W1", [90,szip], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b1", [90,1], initializer = tf.zeros_initializer())
        W2= tf.get_variable("W2", [45,90], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [45,1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [1,45], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
        ### END CODE HERE ###
    
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        
        return parameters
    


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                                # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3

   
    

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)

    return cost    

def random_mini_batches(X, Y, mini_batch_size = 46960, seed = 0):
    import math
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.T.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X.T[:, permutation]
    w=Y.reshape(1,m)
    shuffled_Y = w[:, permutation].reshape(1,m)
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
    
    
    
#Build entire model now
def model(sizeip,X_train, Y_train, X_test, Y_test, learning_rate = 0.1,
          num_epochs = 500, minibatch_size = 46960, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.T.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(sizeip, 1)

    # Initialize parameters
    parameters = initialize_parameters(sizeip)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost =  compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += np.sum(minibatch_cost / num_minibatches)

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, np.sum(epoch_cost)))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        #predicted = tf.nn.sigmoid(Z3)
        #correct_pred = tf.equal(tf.round(predicted), Y)
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #print('Test Accuracy:', sess.run([accuracy, tf.round(predicted)], feed_dict={X: X_test, Y: Y_test}))

        # Calculate the correct predictions
        #correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        #accuracy = tf.equal(tf.cast(correct_prediction, "float"))

        #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters    


#RF
from sklearn.ensemble import RandomForestClassifier

# build a classifier
rf = RandomForestClassifier(n_estimators=50,class_weight={0:.15, 1:0.95}) #,class_weight={0:.15, 1:0.99}

#Train the model using the training sets
rf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred_full = rf.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_pred_full, y_test))


finalpred_full=rf.predict(test6_final)

testans=[]
for i in np.sort(np.unique(test6.id)):
    testans.append(np.average(finalpred_full[test6.id==i]))
    if(i%100==0):
        print(i)

testans1=testans
final_pred=pd.concat([pd.DataFrame(np.unique(test6.id)),pd.DataFrame(testans1)],axis=1)
final_pred.columns=['id','redemption_status']
final_pred.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)

#finalpred_half=rf1.predict(test4_half)
#finalpred_half=pd.concat([test4_1_half.id,pd.DataFrame(finalpred_half,columns=['redemption_status'],index=test4_1_half.index)],axis=1)
#finalpred_full=pd.concat([test4_1_full.id,pd.DataFrame(finalpred_full,columns=['redemption_status'],index=test4_1_full.index)],axis=1)
#
#final_pred=pd.concat([finalpred_half,finalpred_full],axis=0)
#final_pred_sorted=final_pred.sort_index()
#final_pred_sorted.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)
#collab upload files can be done as below
#from google.colab import files
#upload=files.upload()

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
model1 = abc.fit(X_train, y_train)
y_pred_full = model1.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_pred_full, y_test))

finalpred_full=model1.predict(test6_final)

testans=[]
for i in np.sort(np.unique(test6.id)):
    testans.append(np.average(finalpred_full[test6.id==i]))
    if(i%100==0):
        print(i)

testans1=testans
final_pred=pd.concat([pd.DataFrame(np.unique(test6.id)),pd.DataFrame(testans1)],axis=1)
final_pred.columns=['id','redemption_status']
final_pred.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)

#XGBOOST
import xgboost as xgb
from sklearn.metrics import mean_squared_error
xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 3, alpha = 0.10, n_estimators = 50)
#xg_reg.fit(X_train,y_train)
xg_reg.fit(train6_final,train6_y)
preds = xg_reg.predict(X_test)
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(preds, y_test))
preds_1 = xg_reg.predict(test6_final)

testans=[]
for i in np.sort(np.unique(test6.id)):
    testans.append(np.average(preds_1[test6.id==i]))
    if(i%100==0):
        print(i)

testans1=testans
final_pred=pd.concat([pd.DataFrame(np.unique(test6.id)),pd.DataFrame(testans1)],axis=1)
final_pred.columns=['id','redemption_status']
final_pred.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)
