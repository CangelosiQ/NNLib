# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:43:37 2017

@author: Quentin Cangelosi

To Do:
  - change print_info from bool to integer so that you can choose different levels of output information
  - pruning
  - drop out
  - improve/update back_propagation with tanh and linear functions

"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import random
import copy
import math 
__all__ = ['NN',
'logistic','linear', 'linear_pos', 'tanh', 'softmax','forward_propagation', 'back_propagation', 'build_model','predict',
'loss','classification_loss','least_square_loss',
'test_gradient', 'log_sum_exp_over_rows',  'theta_to_model', 'model_to_theta', 'initial_W', 'initial_b', 'classification_performance','split_dataset','split_KFolds','KFolds','normalize_data','add_noise']

class NN():
    def __init__(self,dim_model,types_layers):
        n_params=0
        biases=[]
        for n in range(1,len(dim_model)):
            n_params = n_params+ dim_model[n-1]*dim_model[n]
            biases.append(np.zeros(dim_model[n]))
        self.W=theta_to_model(np.zeros(n_params) ,dim_model)
        self.b=biases
        self.type_layer=types_layers
        self.dim_model=dim_model
        # self.connections_depths=

    def __str__(self):
        print('Neural Network with dimensions:',self.dim_model)
        print('  - Layers:',self.type_layer)
        print('  - W: length',len(self.W))
        for w in self.W:
            print('      ',w.shape)
        print('  - b: length',len(self.b))
        for bs in self.b:
            print('      ',bs.shape)
        return '\n'


"""
# =============================================================================
# ======================== FUNCTIONS FOR NODES =============================
# =============================================================================
"""
## Function Logistic ===========================================================
def logistic(W,b,X):
    return 1 / (1 + np.exp(-(np.dot(W,X)+np.tile(b,(X.shape[1],1)).T )))

## Function Logistic prime ===========================================================
def logistic_prime(W,b,X):
    return 0

## Function Linear =============================================================
def linear(W,b,X):
    return np.dot(W,X)+np.tile(b,(X.shape[1],1)).T

## Function Linear =============================================================
def tanh(W,b,X):
    return 2*logistic(W,2*b,2*X)-1
    
## Function Linear Positive ====================================================
def linear_pos(W,b,X):
    out=(np.dot(W,X)+np.tile(b,(X.shape[1],1)).T)*np.sign(np.dot(W,X)+np.tile(b,(X.shape[1],1)).T)
    return out


## Function Softmax ============================================================
def softmax(W,b,X):
    class_input=np.dot(W , X)+np.tile(b,(X.shape[1],1)).T
    class_normalizer = log_sum_exp_over_rows(class_input) # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1)) # log of probability of each class. size: <number of classes, i.e. dim_model[-1]> by <number of data cases>
    #   class_prob = np.exp(log_class_prob) # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. dim_model[-1]> by <number of data cases>
    return log_class_prob

"""
# =============================================================================
# ======================== FUNCTIONS FOR NET =============================
# =============================================================================
"""
## Function Forward Propagation ================================================
def forward_propagation(model, X):
    options={'logistic':logistic,
             'softmax':softmax,
             'linear':linear,
             'linear_pos':linear_pos,
             'tanh':tanh}

    i=0
#    print(model)
    for t in model.type_layer:
        y = options[t](model.W[i],model.b[i],X)
#        print('Layer',i+1,':',t,'mean(y)',y.mean())
        X=y
        i=i+1
    return y


## Function back_propagation ===================================================
def back_propagation(model, X, y, wd_coefficient):
    # input data size
    N = X.shape[1]
    options={'logistic':logistic,
             'softmax':softmax,
             'tanh':tanh}
    i=0
    Z=[]
    Z.append(X)
    for t in model.type_layer:
        intermediate=Z[-1]
        Z.append(options[t](model.W[i],model.b[i],intermediate))
        i=i+1

    class_prob = np.exp(Z[-1])


    # hidden to output gradient
    output_delta = (class_prob - y)
    dW=[]
    dB=[]
    cumul=1
    for i in range(-len(model.W)+1,1):
        i=-i
        if model.type_layer[i]=='logistic':
            sigma_p=(Z[i+1]*(1-Z[i+1]))
            if i==len(model.W)-1:
                cumul=output_delta*sigma_p
                dW.append( cumul.dot( Z[i].T) / N )
                dB.append( cumul.dot(np.ones((Z[i].shape[1],))) / N )
            else:
                pass
            cumul=(model.W[i+1].T.dot(cumul)*sigma_p)
            dW.append(cumul.dot( Z[i].T) / N )
            # print('la',model.W[i+1].shape,core[-1].shape,output_delta.shape,Z[i].shape)
            # print(step1.shape,step2.shape,dW[-1].shape)
            dB.append( cumul.dot(np.ones((Z[i].shape[1],)))/N )
        if model.type_layer[i]=='softmax':
            dW.append( output_delta.dot(Z[i].T)/ N )
            dB.append( output_delta.dot( np.ones((Z[i].shape[1],))) / N )
            cumul=output_delta
        # print(dW[-1].shape,wd_coefficient,model.W[i].shape)
        dW[-1] += wd_coefficient * model.W[i]
        # print(dB[-1].shape,model.b[i].shape)
        dB[-1] += wd_coefficient * model.b[i]
    return [np.flipud(dW),np.flipud(dB)]
    
    '''for i in range(-len(model.W)+1,1):
        i=-i
        if model.type_layer[i]=='logistic':
            core.append(Z[i+1]*(1-Z[i+1]))
            dW.append(np.dot(np.dot(np.transpose(model.W[i+1]),output_delta)*core[-1], np.transpose(Z[i])) / N)
            dB.append(np.dot(np.dot(np.transpose(model.W[i+1]),output_delta)*core[-1], np.ones((Z[i].shape[1],1)))/N)
        if model.type_layer[i]=='softmax':
            dW.append(np.dot(output_delta , np.transpose(Z[i]))/ N)
            dB.append(np.dot(output_delta , np.ones((Z[i].shape[1],1)))/ N)
        dW[-1] += wd_coefficient * model.W[i]
        dB[-1] += wd_coefficient * np.reshape(model.b[i],(model.b[i].shape[0],1))
    return [np.flipud(dW),np.flipud(dB)]
'''

## Function BUILD_MODEL ========================================================
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layers
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, wd_coefficient, dim_model, type_layer, loss_type,
                   n_iters, learning_rate, momentum_multiplier,
                   do_early_stopping, mini_batch_coef, threshold,
                   print_info=False,train_test_split=True,X_test=None,y_test=None):

    N=y.shape[1]
    mini_batch_size=mini_batch_coef*N
                   #    X=preprocessing.normalize(X,axis=0)

    ##--- Creating and initialising neural net
    model= NN(dim_model,type_layer)
    model = initial_W(model)

    ##--- Splitting train/test data
    if train_test_split:
        X_train,y_train,X_test,y_test=split_dataset(X,y)
    else:
        X_train=X
        y_train=y
        if X_test==None:
            X_test=X
            y_test=y

    # datas_validation=[X,y]

    n_training_cases = X_train.shape[1]

    ##--- Test Gradient
    if n_iters != 0:
        test_gradient(model,loss,(X, y,loss_type, wd_coefficient),back_propagation,(X, y, wd_coefficient),full=False)

    ##--- Preparations before training
    theta = model_to_theta(model.W)
    theta_b=model_to_theta(model.b)
    momentum_speed = theta * 0
    momentum_speed_b = theta_b * 0
    training_data_losses = []
    validation_data_losses = []
    if do_early_stopping:
        best_so_far_theta = -1 # this will be overwritten soon
        best_so_far_validation_loss = np.Inf
        best_so_far_after_n_iters = -1
    print('Initial Cost Function: ',loss(model, X, y,loss_type, wd_coefficient))

    ##--- Training
    for i in range(1, n_iters+1):
#        print('\n Starting iteration ',i)

        model.W = theta_to_model(theta,dim_model)
        training_batch_start = ((i-1) * mini_batch_size % n_training_cases)
        training_batch_inputs = X_train[:, training_batch_start : training_batch_start + mini_batch_size+1]
        training_batch_targets = y_train[:,training_batch_start : training_batch_start + mini_batch_size+1]

        ## Computing gradient
        dW,dB=back_propagation(model, training_batch_inputs, training_batch_targets, wd_coefficient)
        gradient = model_to_theta(dW)
        grad_b = model_to_theta(dB)

        ## Updating Neural Net
        momentum_speed = momentum_speed * momentum_multiplier - gradient
        momentum_speed_b = momentum_speed_b * momentum_multiplier - grad_b
        theta = theta + momentum_speed * learning_rate
        # projection of inadmissible W2s on the domain constraint (which is 0 as we only authorise positive values)
        # this can be even more precise, separating A B C D parameters
        ind1=dim_model[0]*dim_model[1]
        ind2=ind1+dim_model[1]*dim_model[2]
        theta[ind1:ind2] = theta[ind1:ind2]*(theta[ind1:ind2]>0)+1e-9*(theta[ind1:ind2]<=0)
        theta_b = theta_b + momentum_speed_b * learning_rate
        model.b = theta_to_b(theta_b,dim_model)
        model.W = theta_to_model(theta,dim_model)

        if np.sum(model.W[1]<0)>0:
            sys.exit('There is a negative weight in W2!')

        ## Evaluating
        training_data_losses.append(loss(model, X_train, y_train,loss_type, wd_coefficient))
        validation_data_losses.append(loss(model, X_test, y_test,loss_type, wd_coefficient))

        if i>1 and np.abs(training_data_losses[-1]-training_data_losses[-2])/np.abs(training_data_losses[-2])< threshold:
            print('Evolution under threshold, stopping after',i,'iterations.')
            break
        if do_early_stopping  and  (validation_data_losses[-1] < best_so_far_validation_loss) :
            best_so_far_theta = theta # this will be overwritten soon
            best_so_far_validation_loss = validation_data_losses[-1]
            best_so_far_after_n_iters = i

        ## Printing information
        if (i % round(n_iters/dim_model[-1])) == 0 and print_info==True :
            print('After %d optimization iterations, training data loss is %f, and validation data loss is %f\n' %(i, training_data_losses[-1], validation_data_losses[-1]))

        ## Testing gradient once more
    if n_iters != 0:
        test_gradient(model,loss,(X, y,loss_type, wd_coefficient),back_propagation,(X, y, wd_coefficient),full=False)

    ##--- Post-Processing
    if do_early_stopping:
        print('Early stopping: validation loss was lowest after %d iterations. We chose the model that we had then.\n' % best_so_far_after_n_iters)
        theta = best_so_far_theta

    model.W = theta_to_model(theta,dim_model)
    if n_iters != 0:
        plt.figure()
        plt.semilogy(training_data_losses, color='b')
        plt.semilogy(validation_data_losses, color='r')
        plt.title(['Cost function with learning rate',learning_rate,', momentum',momentum_multiplier,', batch size', mini_batch_size])
        plt.legend(['training', 'validation'])
        plt.ylabel('loss')
        plt.xlabel('iteration number')

    datas2 = [[X_train,y_train], [X_test,y_test], [X_test,y_test]]
    data_names = ['training', 'validation', 'test']
    losses=[]

    for data_i in range(3):
        data = datas2[data_i]
        data_name = data_names[data_i]
        losses.append(loss(model, data[0],data[1],loss_type, wd_coefficient))
        print('\nThe loss on the %s data is %f\n' %(data_name, losses[-1]))
        if wd_coefficient!=0:
            print('The classification loss (i.e. without weight decay) on the %s data is %f\n' %(data_name, loss(model, data[0],data[1],loss_type, 0)))

    return [model,losses]


## Function K-Folds ===========================================================
def KFolds(X,y,k,func_build, args_func_build,print_info=True):
    X_train_KFolds,y_train_KFolds,X_test_KFolds,y_test_KFolds=split_KFolds(X,y,k)
    L_train=[]
    L_test=[]
    # L_all=[]
    # model=copy.deepcopy(args_func_build[1])
    args=copy.deepcopy(args_func_build)
    for i in range(k):
        [m,losses,diverged]=func_build(X_train_KFolds[i],y_train_KFolds[i], *args_func_build,train_test_split=False,X_test=X_test_KFolds[i],y_test=y_test_KFolds[i])
        if print_info:
            print("K-Fold %d over %d, train_loss=%0.4f, test_loss=%0.4f" %(i+1,k,losses[0],losses[1]))
        L_train.append(losses[0])
        L_test.append(losses[1])
        args_func_build=copy.deepcopy(args)
        # L_all.append(losses[2])
    if print_info:
        print("Average Results K-Folds: train_loss=%0.4f, test_loss=%0.4f" %(np.mean(L_train),np.mean(L_test)))
    return L_train, L_test

    
## Function predict ===========================================================
def predict(model,X):
    return forward_propagation(model,X)
    
## Function dropout predictions ===============================================
# Dropout method: we "switch off" a node and predict with this model the output
# then do it again with a different node switched off until we decide to stop
# At the end, the prediction is the average of all the predictions.
# This method generally input the generalization accuracy by preventing the 
# model to overfit
def dropout_predictions(model,X):
    dims=model.dim_model
    print(dims,model.W)
    nb_nodes=np.sum(dims[1:-1]) # the number of hidden nodes
    nb_weights_per_layer=[]
    for i in range(len(dims)-1):
        nb_weights_per_layer.append(dims[i]*dims[i+1])
    print('Nb nodes: %d, Nb weights_per_layer: %s' % (nb_nodes,nb_weights_per_layer))
    current_layer = 1
    
    for i in range(nb_nodes):
        print('Drop out with Node: %d - current_layer: %d'%(i,current_layer))
        if i >= np.sum(dims[1:current_layer+1]):
            print('i over ',np.sum(dims[1:current_layer]))
            current_layer+=1
        delta=np.sum(nb_weights_per_layer[:current_layer-1])
        deltap=np.sum(nb_weights_per_layer[:current_layer])
        ind_in_layer=i-np.sum(dims[:current_layer-1])
        print('delta: %d, deltap: %d, ind_in_layer: %d'%(delta,deltap,ind_in_layer))
        # switching off node i
        theta=copy.copy(model_to_theta(model.W))
        
        # print(theta)
        index=range((delta+dims[current_layer-1]*ind_in_layer).astype(int),(delta+dims[current_layer-1]*(ind_in_layer+1)).astype(int))
        for i in index:
            print(i,end=' ')
        theta[index]=0
        # print('\n',type(theta))
        ind=deltap+range(dims[current_layer+1])*dims[current_layer]+ind_in_layer
        print(ind.astype(int))
        theta[ind.astype(int)]=0
        # print(theta)
        
        # biases
        theta_b=model_to_theta(model.b)
        print(theta_b)
        theta_b[i]=0
        
        model_to_use=copy.copy(model)
        model_to_use.W=theta_to_model(theta,dims)
        model_to_use.b=theta_to_(theta,dims)
        stop=input('Press any key and enter')
    return 0

    

"""============================================================================
    =======================  FUNCTIONS COST  ==============================
============================================================================"""
## Classification Loss =========================================================
def classification_loss(y,t):
    return -np.mean(np.sum(y*t, axis=0))

## Least square Loss ===========================================================
def least_square_loss(y,t):
    return np.mean(np.power(y-t,2))

## Function Loss ===============================================================
def loss(model, X, targets, method, wd_coefficient=0):
    options={'classification':classification_loss,
             'least_square':least_square_loss}

    y=forward_propagation(model,X)
    loss = options[method](y,targets)
    wd_loss = np.sum(np.power(model_to_theta(model.W),2))/2*wd_coefficient
    return loss + wd_loss


"""============================================================================
    =====================  FUNCTIONS for WEIGHTS   ========================
============================================================================"""

## Function Model to Theta =====================================================
def model_to_theta(model_W):
    # This function takes a list of matrices of weights, and turns it into 
    # one long vector. Taking line by line.
    T=[]
    for i in range(len(model_W)):
        if len(model_W[i].shape)==2:
            [n,m]=model_W[i].shape
        else:
            n=model_W[i].shape[0]
            m=1
        T.append(np.reshape(model_W[i],(n*m,1)))
    return np.concatenate(T)


## Function Theta to Model =====================================================
def theta_to_model(theta,dim_model):
    # This function takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    W=[]
    offset=0
    # print(dim_model,theta.shape)
    for n in range(1,len(dim_model)):
        W.append(np.reshape(theta[offset: offset+dim_model[n-1]*dim_model[n]], (dim_model[n],dim_model[n-1])))
        offset=offset+dim_model[n-1]*dim_model[n]
    return W

## Function Theta to Model =====================================================
def theta_to_b(theta,dim_model):
    # This function takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    b=[]
    offset=0
    for n in range(1,len(dim_model)):
        b.append(np.reshape(theta[offset: offset+dim_model[n]], (dim_model[n],)))
        offset=offset+dim_model[n]
    return b


## Function Initialisation Weigths =============================================
def initial_W(model,positif=False):
    # should be random init!
      # Initialize the parameters to random values. We need to learn these.
    np.random.seed(1)
    for i in range(len(model.W)):
        model.W[i] = np.random.randn(model.W[i].shape[0],model.W[i].shape[1]) / np.sqrt(model.W[i].shape[0])
        if positif==True: model.W[i]=np.abs(model.W[i])
    return model

## Function Initialisation Weigths =============================================
def initial_b(model,positif=False):
    # should be random init!
      # Initialize the parameters to random values. We need to learn these.
    np.random.seed(1)
    for i in range(len(model.b)):
        model.b[i] = np.random.randn(len(model.b[i]),) / np.sqrt(len(model.b[i]))
        if positif==True: model.b[i]=np.abs(model.b[i])
    return model

"""============================================================================
    =====================  FUNCTIONS for DATA   ========================
============================================================================"""
## Function data Normalisation =================================================
def normalize_data(X,axis=0,method='L2',positif=False,params=None):
    if params==None:
        if method=='L2':
            s=np.linalg.norm(X,axis=axis)
            # print(s)
            m=np.mean(X,axis)*0
        if method=='Features':
            m=np.mean(X,axis)
            if axis==1:
                m=np.reshape(m,(X.shape[0],1))
            
            s=np.max(np.abs(X-m),axis)
        if axis==1:
            m=np.reshape(m,(X.shape[0],1))
            # s=np.reshape(np.var(X,axis),(X.shape[0],1))
            s=np.reshape(s,(X.shape[0],1))
        else:
            m=np.mean(X,axis)
            # s=np.var(X,axis)
            s=np.max(np.abs(X-m),axis)
    else:
        m,s=params
    # print(m,s,np.mean(X-m,axis),np.var((X-m)/s,axis))
    X=(X-m)/s
    if positif:
        X=(X+1)/2
    
    if params==None:
        return X,m,s
    else:
        return X

## Function split data set =====================================================
def split_dataset(X,y,prop=0.75):
    n=y.shape[1]
    m=X.shape[0]
    A=copy.deepcopy(X.T)
    A=np.append(A,copy.deepcopy(y.T),axis=1)
    random.shuffle(A)
    train,test=np.split(A,[round(n*prop)])
    X_train=train.T[:m,:]
    y_train=train.T[m:,:]
    X_test=test.T[:m,:]
    y_test=test.T[m:,:]
    # print('X y X_train, y_train X_test y_test',X.shape,y.shape,X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    return X_train,y_train,X_test,y_test
    
## Function Split K-Folds ======================================================
def split_KFolds(X,y,k):
    n=y.shape[1]
    m=X.shape[0]
    
    A=copy.deepcopy(X.T)
    A=np.append(A,copy.deepcopy(y.T),axis=1)
    random.shuffle(A)
    # r=np.arange(0,len(A),round(len(A)/k))
    KFolds=np.array_split(A,k)
    X_train_KFolds=[]
    y_train_KFolds=[]
    X_test_KFolds=[]
    y_test_KFolds=[]
    for i in range(k):
        X_train_KFolds.append(np.concatenate(KFolds[:-1],axis=0).T[:m,:])
        y_train_KFolds.append(np.concatenate(KFolds[:-1],axis=0).T[m:,:])
        X_test_KFolds.append(KFolds[-1].T[:m,:])
        y_test_KFolds.append(KFolds[-1].T[m:,:])
        # print(len(KFolds),KFolds[0].shape,X_train_KFolds[-1].shape,y_train_KFolds[-1].shape,X_test_KFolds[-1].shape,y_test_KFolds[-1].shape)
        
        KFolds=np.roll(KFolds,1)
    return X_train_KFolds, y_train_KFolds, X_test_KFolds, y_test_KFolds
    
def add_noise(X,new_size,noise_level=0.01,axis=1):
    n=X.shape[axis]
    nb=int(np.ceil(new_size/n))
    # print(nb)
    if n==new_size:
        X_noise=X+np.random.randn(X.shape[0],X.shape[1])*X*noise_level
    else:
        X_noise=copy.copy(X)
    for i in range(nb):
        X_noise=np.concatenate([X_noise,X+np.random.randn(X.shape[0],X.shape[1])*X*noise_level],axis)
    if axis==1:
        return X_noise[:,:new_size]
    else:
        return X_noise[:new_size,:]
        
    


"""============================================================================
    =====================  ADDITIONAL FUNCTIONS   ========================
============================================================================"""
## Function Log Sum Exp over rows ==============================================
def log_sum_exp_over_rows(a):
    # This computes log(sum(exp(a), 1)) in a numerically stable way
    maxs_small = np.max(a, axis=0)
    maxs_big = np.tile(maxs_small, (a.shape[0], 1))
    return np.log(np.sum(np.exp(a - maxs_big),axis=0)) + maxs_small

## Function Test Gradient =====================================================
def test_gradient(model, func_loss,args_loss,func_grad,args_grad,full=False):
#    print('===== Welcome to test gradient =====')
    problem=False
    nb_pb=0
    test_model=model
    base_theta = model_to_theta(model.W)
    plt.plot(base_theta)
    h = 1e-3
    correctness_threshold = 2e-1
    analytic_gradient = model_to_theta(func_grad(model,*args_grad)[0])
    biggest_error=0
    # Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
    if full:
        N=len(analytic_gradient)
        ###################### TO REMOVE WHEN FINISHED |
        N=N-20 ############### <-----------------------
        F=1
        P=0
    else:
        N=100
        F=0
        P=1
    for i in range(N):
        problem=False
        test_index = ((i * 1299721 % base_theta.shape[0]))*P+i*F # 1299721 is prime and thus ensures a somewhat random-like selection of indices
#        print('testing element',test_index)
        analytic_here = analytic_gradient[test_index]
        theta_step = base_theta * 0
        theta_step[test_index] = h
        contribution_distances = np.concatenate([range(-4,0), range(1,5)])
        contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280]
        temp = 0

        for contribution_index in range(0,8):
            test_model.W=theta_to_model(base_theta + theta_step * contribution_distances[contribution_index],model.dim_model)
            temp = temp + func_loss(test_model,*args_loss) * contribution_weights[contribution_index]

        fd_here = temp / h
        diff = np.abs(analytic_here - fd_here)
            # print('#d #e #e #e #e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here)
        if (analytic_here==0 and fd_here < correctness_threshold) or (diff / (np.abs(analytic_here)) < correctness_threshold):
            if analytic_here==0:
                continue
            if diff / (np.abs(analytic_here))>biggest_error:
                biggest_error=diff / (np.abs(analytic_here))
            # print('\nTheta element %d, with value %e, has finite difference gradient %e but analytic gradient %e. Difference:' %(test_index+1, base_theta[test_index], fd_here, analytic_here), np.abs(fd_here-analytic_here)[0],' (', (np.abs(fd_here-analytic_here)/fd_here*100)[0],'%) Ratio:',(analytic_here/fd_here)[0])
            continue
        problem=True
        nb_pb+=1
        print('\n ERROR! Theta element %d, with value %e, has finite difference gradient %e but analytic gradient %e. Difference:' %(test_index+1, base_theta[test_index], fd_here, analytic_here), np.abs(fd_here-analytic_here)[0],' (', (diff / (np.abs(analytic_here))*100)[0],'%) Ratio:',(analytic_here/fd_here)[0])
#        sys.exit(1)
        if problem and not full:
            sys.exit(1)

    if nb_pb==0:
        print('Gradient test passed. That means that the gradient that your code computed is within',biggest_error*100,'% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).\n')
    else:
        print('There were',nb_pb, 'errors over', N, ' (', nb_pb/N*100,'%)')
        sys.exit(1)


# NOT UPDATED
## Function Classification Performance =========================================
def classification_performance(model, data):
    # This returns the fraction of data cases that is incorrectly classified by the model.
    hid_output = logistic(model.W[0],model.b[0], data[0])
    class_input = np.dot(model.W[1], hid_output)

    choices = np.argmax(class_input,axis=0)+1 # choices is integer: the chosen class, plus 1.
    targets = np.argmax(data[1],axis=0)+1 # targets is integer: the target class, plus 1.
    return np.mean(np.float64(choices != targets))


