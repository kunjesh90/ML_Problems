import pandas as pd
import numpy as np
train=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/train.csv")
train.info()
train.head()
train.isna().sum()
campdata=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/campaign_data.csv")
campdata.info()
campdata.isna().sum()
campdata.head()

coupon=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/coupon_item_mapping.csv")
coupon.info()
coupon.isna().sum()

custdemo=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/customer_demographics.csv")
custdemo.info()
custdemo.isna().sum()
custdemo.no_of_children.unique()
custdemo.no_of_children[custdemo.no_of_children.isna()]=0
custdemo.isna().sum()
custdemo.marital_status[custdemo.marital_status.isna()]="NotGiven"
custdemo.isna().sum()
custdemo.marital_status.unique()

custtran=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/customer_transaction_data.csv")
custtran.info()
custtran.head()
custtran.isna().sum()

itemdata=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/item_data.csv")
itemdata.info()
itemdata.isna().sum()

test=pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/test.csv")
test.info()
test.isna().sum()

ct=np.zeros((len(train))).astype(str)
for i in range(len(train)):
    ct[i]=campdata.campaign_type[np.where(train.campaign_id[i]==campdata.campaign_id)[0][0]]
        
ct1=pd.DataFrame(ct)    
ct1.columns=["campaign_type"]
train1=pd.concat([train,ct1],axis=1)
train1.head() #attached camp type 
train1.info()
#coupid=np.zeros((len(custtran)))
#for i in range(len(custtran)):
#    try:
#        coupid[i]=coupon.coupon_id[np.where(custtran.item_id[i]==coupon.item_id)[0][0]]
#    except:
#        coupid[i]=0
    
#train.info()    
#x1=train.coupon_id[0]
#y1=train.customer_id[0]
#xx=custtran[custtran.customer_id==y1]
#yy=coupon[coupon.coupon_id==x1]
#
#for i in range(len(yy)):
#    for j in range(len(xx)):
#        if yy.item_id.iloc[i]==xx.item_id.iloc[j]:
#            print(i,"and",j)

train2=pd.merge(train1,custdemo , how='left', on='customer_id') 
train2.isna().sum()    #there are NAs for which customers data not available

ct=np.zeros((len(test))).astype(str)
for i in range(len(test)):
    ct[i]=campdata.campaign_type[np.where(test.campaign_id[i]==campdata.campaign_id)[0][0]]
        
ct1=pd.DataFrame(ct)    
ct1.columns=["campaign_type"]
test1=pd.concat([test,ct1],axis=1)
test1.head() #attached camp type 
test1.info()

test2=pd.merge(test1,custdemo , how='left', on='customer_id') 
test2.isna().sum()    #there are NAs for which customers data not available




train3=pd.merge(custtran,itemdata, how='left', on='item_id') 
train3.info()
train3.isna().sum()    
#train4=train3[train3.coupon_discount]
disc=[]
odisc=[]
totalsales=[]

for i in np.sort((np.unique(train3.customer_id))):
    disc.append(np.sum(train3.coupon_discount[train3.customer_id==i]))
    odisc.append(np.sum(train3.other_discount[train3.customer_id==i]))
    totalsales.append(np.sum(train3.quantity[train3.customer_id==i]*train3.selling_price[train3.customer_id==i]))

import matplotlib.pyplot as plt
plt.plot(abs(np.array(disc)))
np.mean(disc)
np.median(disc)
discdf=pd.concat([pd.DataFrame(np.unique(train3.customer_id),columns=['customer_id']),pd.DataFrame(odisc,columns=['ocoupondisc']),pd.DataFrame(disc,columns=['coupondisc']),pd.DataFrame(totalsales,columns=['totalsales'])],axis=1)

train4=pd.merge(train2,discdf,how='left',on='customer_id') 
test4=pd.merge(test2,discdf,how='left',on='customer_id') 
test4.isna().sum()    #there are NAs for which customers data not available
train4.isna().sum()    #there are NAs for which customers data not available

#l=np.zeros((len(disc))).astype('str')
#for i in range(len(disc)):
#    if abs(disc[i])>abs(np.mean(disc)):
#        l[i]="Likely"
#    else:
#        l[i]="Unklikely"

#l=pd.DataFrame(l,columns=["couponlikely"])
#w=np.sort((np.unique(train3.customer_id)))      
#w=pd.DataFrame(w,columns=["customer_id"])
#lw=pd.concat([l,w],axis=1)
#train4=pd.merge(train2,lw, how='left', on='customer_id') 
#train4.isna().sum()
#train4.info()
#test4=pd.merge(test2,lw, how='left', on='customer_id') 
#test4.isna().sum()
#test4.info()

disc1=[]
for i in np.sort((np.unique(train3.item_id))):
    disc1.append(np.sum(train3.coupon_discount[train3.item_id==i]+train3.other_discount[train3.item_id==i]))
    if(i%100==0):
        print(i)
        
import matplotlib.pyplot as plt
plt.plot(abs(np.array(disc1)))
np.mean(disc1)
np.median(disc1)
l1=np.zeros((len(disc1))).astype('str')

for i in range(len(disc1)):
    if abs(disc1[i])>0:#from graph chekced and found 0 tims avg disc as a limit
        l1[i]="likely"
    else:
        l1[i]="unklikely"
#    if(i%100==0):
#        print(i)
    
l1=pd.DataFrame(l1,columns=["itemcouponlikely"])
w1=np.sort((np.unique(train3.item_id)))
w1=pd.DataFrame(w1,columns=["item_id"])
lw1=pd.concat([l1,w1],axis=1)

train5=pd.merge(coupon,lw1, how='left', on='item_id') 
likelycouponid=train5.coupon_id[train5.itemcouponlikely=="likely"]
#train5_1=pd.concat([train5.iloc[:,0],train5.iloc[:,2]],axis=1)
#train4_1=pd.merge(train4,train5_1,how='left',on='coupon_id')
train4["likelycouponid"]=np.zeros(len(train4))
for i in range(len(train4)):
    if(train4.coupon_id[i] in likelycouponid):
        train4["likelycouponid"][i]=1
    if(i%100==0):
        print(i)
#train4_1.info()        

test4["likelycouponid"]=np.zeros(len(test4))
for i in range(len(test4)):
    if(test4.coupon_id[i] in likelycouponid):
        test4["likelycouponid"][i]=1
    if(i%100==0):
        print(i)
        
test4.info()       


x=pd.merge(train,coupon,how='inner',on='coupon_id')
x.info()
x.head()
length1=np.zeros((len(x.coupon_id.unique())))
a=0
for i in x.coupon_id.unique():
    length1[a]=len(x.item_id[x.coupon_id==i].unique())
    a+=1

im=pd.concat([pd.DataFrame(x.coupon_id.unique()),pd.DataFrame(length1)],axis=1)
im.columns=['coupon_id','numitemcoup']
train_im=pd.merge(train,im,how='left',on='coupon_id')

#totalitemincoup=np.zeros((len(train)))
#for j in range(len(train)):
#    totalitemincoup[j]= length1[np.where(x.coupon_id.unique()==train.coupon_id[j])[0][0]]

train.info()
    
y=pd.merge(test,coupon,how='inner',on='coupon_id')
y.info()
y.head()
length1y=np.zeros((len(y.coupon_id.unique())))
a=0
for i in y.coupon_id.unique():
    length1y[a]=len(y.item_id[y.coupon_id==i].unique())
    a+=1

im_test=pd.concat([pd.DataFrame(y.coupon_id.unique()),pd.DataFrame(length1y)],axis=1)
im_test.columns=['coupon_id','numitemcoup']
test_im=pd.merge(test,im_test,how='left',on='coupon_id')

#xx=pd.merge(x,custtran,how='inner',on=['customer_id','item_id'])
    
#totalitemincoup_y=np.zeros((len(test)))
#for j in range(len(test)):
#    totalitemincoup_y[j]= length1[np.where(y.coupon_id.unique()==test.coupon_id[j])[0][0]]

 
''' intermediate save the cleaned data        
test4.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/test4.csv',index=False)
train4.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/train4.csv',index=False)
if session ends then take above files
test4=pd.read_csv('C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/test4.csv')
train4=pd.read_csv('C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/train4.csv')

'''
train4_1=pd.concat([train4,train_im['numitemcoup']],axis=1)
test4_1=pd.concat([test4,test_im['numitemcoup']],axis=1)

train_y=train4_1.redemption_status
del train4_1['redemption_status']

train4_1.info()        
train4_1.isna().sum()
test4_1.info()        
test4_1.isna().sum()

'''
test4_1.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/test4_1.csv',index=False)
train4_1.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train4_1.csv',index=False)
train_y.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train4_y.csv',index=False)
train4_1=pd.read_csv('C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train4_1.csv')
test4_1=pd.read_csv('C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/test4_1.csv')
train_y=pd.read_csv('C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train4_y.csv')
'''

train4_1_full=train4_1[train4_1.age_range.isna()==False]
test4_1_full=test4_1[test4_1.age_range.isna()==False]

train_y_full=train_y[train4_1.age_range.isna()==False]
len(train4_1_full)==len(train_y_full)

train4_1_half=train4_1[train4_1.age_range.isna()==True]
test4_1_half=test4_1[test4_1.age_range.isna()==True]

train_y_half=train_y[train4_1.age_range.isna()==True]
len(train4_1_half)==len(train_y_half)

#test4_1_full=test4_1[test4_1.age_range.isna()==False]
#len(test4_full)

#test4_1_half=test4_1[test4_1.age_range.isna()==True]
len(test4_1_half)



train4_1_half.isna().sum()
test4_1_half.isna().sum()


test4_full_obj=pd.concat([test4_1_full.iloc[:,4:11],test4_1_full.iloc[:,14]],axis=1)
test4_full_obj_1=test4_full_obj.astype('object')
test4_full_obj_1.info()
test4_full_obj_dummies=pd.get_dummies(test4_full_obj_1)
test4_full=pd.concat([test4_full_obj_dummies,test4_1_full.iloc[:,11:14],test4_1_full.iloc[:,-1]],axis=1)
test4_full.info()


test4_half_obj=pd.concat([test4_1_half.iloc[:,4],test4_1_half.iloc[:,14]],axis=1)
test4_half_obj_1=test4_half_obj.astype('object')
test4_half_obj_1.info()
test4_half_obj_dummies=pd.get_dummies(test4_half_obj_1)
test4_half=pd.concat([test4_half_obj_dummies,test4_1_half.iloc[:,11:14],test4_1_half.iloc[:,-1]],axis=1)
test4_half.info()


train4_full_obj=pd.concat([train4_1_full.iloc[:,4:11],train4_1_full.iloc[:,14]],axis=1)
train4_full_obj_1=train4_full_obj.astype('object')
train4_full_obj_1.info()
train4_full_obj_dummies=pd.get_dummies(train4_full_obj_1)
train4_full=pd.concat([train4_full_obj_dummies,train4_1_full.iloc[:,11:14],train4_1_full.iloc[:,-1]],axis=1)
train4_full.info()#40col


train4_half_obj=pd.concat([train4_1_half.iloc[:,4],train4_1_half.iloc[:,14]],axis=1)
train4_half_obj_1=train4_half_obj.astype('object')
train4_half_obj_1.info()
train4_half_obj_dummies=pd.get_dummies(train4_half_obj_1)
train4_half=pd.concat([train4_half_obj_dummies,train4_1_half.iloc[:,11:14],train4_1_half.iloc[:,-1]],axis=1)
train4_half.info() #8col

'''
train4_half.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train4_half.csv',index=False)
train4_full.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train4_full.csv',index=False)
test4_half.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/test4_half.csv',index=False)
test4_full.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/test4_full.csv',index=False)
train_y_full.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train_y_full.csv',index=False)
train_y_half.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/a/train_y_half.csv',index=False)

'''


'''
train4_half_1=pd.concat([train4_half.iloc[:,4],train4_half.iloc[:,11:]],axis=1)
train4_half_1.info()
test4_half_1=pd.concat([test4_half.iloc[:,4],test4_half.iloc[:,11:]],axis=1)
test4_half_1.info()

train4_half_1_obj=train4_half_1.iloc[:,:-1]
train4_half_1_obj=train4_half_1_obj.astype('object')
train4_half_1_obj=pd.get_dummies(train4_half_1_obj)
test4_half_1_obj=test4_half_1.iloc[:,:-1].astype('object')
test4_half_1_obj=pd.get_dummies(test4_half_1_obj)

train4_half_1_obj.info()#6 columns
test4_half_1_obj.info()#6 columns
train4_half_1_obj_new=pd.concat([train4_half_1_obj,train4_half_1.iloc[:,-1]],axis=1)
test4_half_1_obj_new=pd.concat([test4_half_1_obj,test4_half_1.iloc[:,-1]],axis=1)


train4_full_1=train4_full.iloc[:,4:]
train4_full_1.info()
test4_full_1=test4_full.iloc[:,4:]
test4_full_1.info()


train4_full_1_obj=train4_full_1.iloc[:,:-1].astype('object')
train4_full_1_obj.info()
test4_full_1_obj=test4_full_1.iloc[:,:-1].astype('object')
test4_full_1.info()

train4_full_1_obj=pd.get_dummies(train4_full_1)
train4_full_1_obj.info()#38 columns
test4_full_1_obj=pd.get_dummies(test4_full_1)
train4_full_1_obj.info()#38 columns

train4_full_1_obj_new=pd.concat([train4_full_1_obj,train4_full_1.iloc[:,-1]],axis=1)
test4_full_1_obj_new=pd.concat([test4_full_1_obj,test4_full_1.iloc[:,-1]],axis=1)

'''
#pd.DataFrame(length1,columns=["eligibleitem"])

#NN Starts
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

def random_mini_batches(X, Y, mini_batch_size = 256, seed = 0):
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
def model(sizeip,X_train, Y_train, X_test, Y_test, learning_rate = 0.01,
          num_epochs = 1500, minibatch_size = 256, print_cost = True):
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

train4_full=(train4_full-train4_full.mean())/train4_full.std()
test4_full=(test4_full-test4_full.mean())/test4_full.std()
train4_half=(train4_half-train4_half.mean())/train4_half.std()
test4_half=(test4_half-test4_half.mean())/test4_half.std()

train4_full.info()
train4_half.info()
#below 7 lines for nongeo data
train4_full_nongeo=pd.concat([train4_full.iloc[:,0:2],train4_full.iloc[:,-6:]],axis=1)
train4_full_nongeo.info()
train4_full =pd.concat([train4_full_nongeo,train4_half],axis=0).sort_index()

test4_full_nongeo=pd.concat([test4_full.iloc[:,0:2],test4_full.iloc[:,-6:]],axis=1)
test4_full_nongeo.info()
test4_full =pd.concat([test4_full_nongeo,test4_half],axis=0)
test4_full=test4_full.sort_index()

train_y=pd.concat([train_y_full,train_y_half],axis=0).sort_index()
#for non geo data end

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train4_full,train_y, test_size=0.0001)

parameters = model(len(train4_full.columns),np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))

def sigmoid(x):
    return(1/(1+np.exp(-x)))

z1=np.matmul(parameters["W1"],np.array(X_train.T))+parameters["b1"]
a1=np.maximum(z1,0)
z2=np.matmul(parameters["W2"],a1)+parameters["b2"]
a2=np.maximum(z2,0)
z3=np.matmul(parameters["W3"],a2)+parameters["b3"]
ans=sigmoid(z3)    

len1=ans.shape[1]

resultant=np.zeros(len1)
for i in range(len1):
    if ans[:,i]>0.05:
        resultant[i]=1
    else:
        resultant[i]=0

np.unique(resultant, return_counts=True)

np.unique(np.array(y_test), return_counts=True)

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_train, resultant))


#from sklearn.metrics import auc
#y = np.array([1, 1, 2, 2])
#pred = np.array([0.1, 0.4, 0.35, 0.8])
#fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
#metrics.auc(fpr, tpr)
#predict for test full
z1=np.matmul(parameters["W1"],np.array(test4_full.T))+parameters["b1"]
a1=np.maximum(z1,0)
z2=np.matmul(parameters["W2"],a1)+parameters["b2"]
a2=np.maximum(z2,0)
z3=np.matmul(parameters["W3"],a2)+parameters["b3"]
ans_full=sigmoid(z3)    

len11=ans_full.shape[1]
resultant_test_full=np.zeros(len11)
for i in range(len11):
    if ans_full[:,i]>0.112:   #Play
        resultant_test_full[i]=1
    else:
        resultant_test_full[i]=0

np.unique(resultant_test_full,return_counts=True)
#only for nongeodata full
pd.concat([test.id,pd.DataFrame(resultant_test_full,columns=['redemption_status'],index=test.index)],axis=1).to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)
#end

df2=pd.concat([test4_1_full.id,pd.DataFrame(resultant_test_full,columns=['redemption_status'],index=test4_1_full.index)],axis=1)


#NN for half

X_train, X_test, y_train, y_test = train_test_split(train4_half,train_y_half, test_size=0.0001)

parameters1 = model(len(train4_half.columns),np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))
def sigmoid(x):
    return(1/(1+np.exp(-x)))

z1=np.matmul(parameters1["W1"],np.array(X_train.T))+parameters1["b1"]
a1=np.maximum(z1,0)
z2=np.matmul(parameters1["W2"],a1)+parameters1["b2"]
a2=np.maximum(z2,0)
z3=np.matmul(parameters1["W3"],a2)+parameters1["b3"]
ans=sigmoid(z3)    

len1=ans.shape[1]

resultant1=np.zeros(len1)
for i in range(len1):
    if ans[:,i]>0.5:
        resultant1[i]=1
    else:
        resultant1[i]=0

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_train, resultant1))

#predict for half
z1=np.matmul(parameters1["W1"],np.array(test4_half.T))+parameters1["b1"]
a1=np.maximum(z1,0)
z2=np.matmul(parameters1["W2"],a1)+parameters1["b2"]
a2=np.maximum(z2,0)
z3=np.matmul(parameters1["W3"],a2)+parameters1["b3"]
ans_test_half=sigmoid(z3)    

len1=ans_test_half.shape[1]
resultant_test_half=np.zeros(len1)
for i in range(len1):
    if ans[:,i]>0.5: #Play
        resultant_test_half[i]=1
    else:
        resultant_test_half[i]=0
        
#pd.DataFrame(resultant_test_half,columns=['redemption_status'],index=False)
df1=pd.concat([test4_1_half.id,pd.DataFrame(resultant_test_half,columns=['redemption_status'],index=test4_1_half.index)],axis=1)

final_pred=pd.concat([df1,df2],axis=0)
final_pred_sorted=final_pred.sort_index()
final_pred_sorted.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)


#Random Forest : CART
train4_full.info()
train4_half.info()
train_y_full
train_y_half

from sklearn.ensemble import RandomForestClassifier

# build a classifier
rf = RandomForestClassifier(n_estimators=1500,class_weight={0:.15, 1:0.99})

#Train the model using the training sets
rf.fit(train4_full, train_y_full)

#Predict the response for test dataset
y_pred_full = rf.predict(train4_full)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_pred_full, train_y_full))

finalpred_full=rf.predict(test4_full)

rf1 = RandomForestClassifier(n_estimators=1500,class_weight={0:.15, 1:0.99})
#Train the model using the training sets
rf1.fit(train4_half, train_y_half)

#Predict the response for test dataset
y_pred_half = rf1.predict(train4_half)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_pred_half, train_y_half))


finalpred_half=rf1.predict(test4_half)
finalpred_half=pd.concat([test4_1_half.id,pd.DataFrame(finalpred_half,columns=['redemption_status'],index=test4_1_half.index)],axis=1)
finalpred_full=pd.concat([test4_1_full.id,pd.DataFrame(finalpred_full,columns=['redemption_status'],index=test4_1_full.index)],axis=1)

final_pred=pd.concat([finalpred_half,finalpred_full],axis=0)
final_pred_sorted=final_pred.sort_index()
final_pred_sorted.to_csv(r'C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/project/Amex/final_pred.csv',index=False)
#
#x=pd.merge(train,coupon,how='inner',on='coupon_id')
#x.info()
#x.head()
#length1=np.zeros((len(x.coupon_id.unique())))
#a=0
#for i in x.coupon_id.unique():
#    length1[a]=len(x.item_id[x.coupon_id==i].unique())
#    a+=1
#    
#    
#y=pd.merge(test,coupon,how='inner',on='coupon_id')
#y.info()
#y.head()
#length1y=np.zeros((len(y.coupon_id.unique())))
#a=0
#for i in y.coupon_id.unique():
#    length1y[a]=len(y.item_id[y.coupon_id==i].unique())
#    a+=1