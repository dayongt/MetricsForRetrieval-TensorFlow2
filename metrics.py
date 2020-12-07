# implement map at k for hashing retrieval purpose in Tensorflow 2.3.0 
import tensorflow as tf 
import numpy as np 

 
def dist(X,Y): 
    # calculate the mutual Euclidean distance between two data matrix
    # X: n*d
    # Y: m*d
    # output: n*m; the (i,j) element of output is the Euclidean distance between i-th row of X and j-th row of Y
    Y1 = tf.transpose(tf.math.reduce_sum(tf.math.square(Y),axis=1)) 
    X1 = tf.math.reduce_sum(tf.math.square(X),axis=1) 
    ab = tf.matmul(X,tf.transpose(Y)) 
    return tf.tile(X1[...,tf.newaxis],[1,Y.shape[0]])+tf.tile(Y1[tf.newaxis,...],[X.shape[0],1]) - tf.multiply(2.0,ab) 
def hamming_dist(X,Y): 
    # do the similar thing as 'dist' but for Hamming distance
    # X and Y are binary label matrix
    X = tf.cast(X>0,tf.float32) 
    Y = tf.cast(Y>0,tf.float32) 
    return tf.matmul(1-X,tf.transpose(Y))+tf.matmul(X,tf.transpose(1-Y)) 

 
def and_(X,Y): 
    # X: n*l binary label matrix
    # Y: m*l binary label matrix
    # output: n*m; the (i,j) element of output indicates whether the i-th row of X and the j-th row of Y are true neighbors.
    # For multi-labeled data sets, the true neighbors are defined as those sharing at least one common label.
    # For example, [0,1,1,0] and [0,1,0,0] are neighbors, but [0,1,1,0] and [1,0,0,1] are not.
    return tf.matmul(X,tf.transpose(Y))!=0 

 
def mAP(Bn,Bs,Ln,Ls,radius=50): 
    #Bn: n*d generally the retrieval data set
    #Bs: m*d generally the query data set
    #Ln: the groundtruth labels for Bn
    #Ls: the groundtruth labels for Bs
    #output: the mAP at radius.

    Bn = tf.cast(Bn,tf.float32) 
    Bs = tf.cast(Bs,tf.float32) 
    Ln = tf.cast(Ln,tf.float32) 
    Ls = tf.cast(Ls,tf.float32) 
    D = dist(Bn,Bs) 
    L = tf.argsort(D,0)[0:radius,:] 
    L = tf.range(0,Bn.shape[0]*Bs.shape[0],Bn.shape[0]) + L 
    L = tf.reshape(tf.transpose(L),[-1]) 
    T = tf.cast(and_(Ln,Ls),tf.float32) 
    T = tf.reshape(tf.transpose(T),[-1]) 
    T = tf.gather(T,L) 
    T = tf.reshape(T,(Bs.shape[0],-1)) 

    T2 = tf.cumsum(T,1)*tf.tile(tf.reshape(tf.divide(1,tf.range(1,radius+1,dtype=tf.float32)),[1,-1]),[T.shape[0],1]) 
    return tf.reduce_mean(tf.divide(tf.reduce_sum(T2*T,1),tf.reduce_sum(T,1))) 

 
def fmeasure(Bn,Bs,Ln,Ls,radius=2,beta=2): 
    #Bn: n*d generally the retrieval data set
    #Bs: m*d generally the query data set
    #Ln: the groundtruth labels for Bn
    #Ls: the groundtruth labels for Bs
    #output: the Fmeasure at radius.
    
    Bn = tf.cast(Bn,tf.float32) 
    Bs = tf.cast(Bs,tf.float32) 
    Ln = tf.cast(Ln,tf.float32) 
    Ls = tf.cast(Ls,tf.float32) 
    D = hamming_dist(Bn,Bs) 
    L = D<=radius 
    T = and_(Ln,Ls) 
    TP = tf.reduce_sum(tf.cast(tf.logical_and(L,T),tf.float32),axis=0) 
    RP = tf.reduce_sum(tf.cast(L,tf.float32),axis=0) 
    AP = tf.reduce_sum(tf.cast(T,tf.float32),axis=0) 
    eps = tf.constant(1e-10)
    precision = TP/(RP +eps)
    recall = TP/(AP +eps)
    return beta*tf.reduce_mean(precision*recall/(precision+recall))
