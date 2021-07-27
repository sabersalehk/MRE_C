# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:49:54 2019

@author: Saber Salehkaleybar
"""
import numpy as np
import scipy as sp
from multiprocessing import Pool
import argparse



def comptheta(X,y,d):
    theta = cp.Variable(d)
    objective = cp.Minimize(cp.sum_squares(X*theta - y)+0.1*cp.sum_squares(theta))
    constraints = [0 <= theta, theta <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return theta.value

def listtokey(L):
     return " ".join(str(x) for x in L)

def keytolist(K):
    return list(map(int,K.split(' ')))

def MRE_C(X, y, d, n, t, l, edgesize, loss_type):
    """ Running MRE_C algorithm at the machines
    """
    s_part = .5*np.ones(d)
    s_parti = np.ones(d)
    prob = 2**((d-2)*np.arange(t+1))
    prob = prob/np.sum(prob)
    l_selec = np.random.choice(t+1,1,p=prob)
    grid_l = range(1,int(2**l_selec)+1)
    p_parti = np.random.choice(grid_l,d,p=np.ones(int(2**l_selec))/(2**l_selec))
    p_part = s_part-(edgesize-edgesize/2**(l_selec))*np.ones(d)+(p_parti-1)*(2*edgesize/2**(l_selec))
    X2 = X#X[int(np.floor(n/2))+1:n,:]
    y2 = y#y[int(np.floor(n/2))+1:n]
    if loss_type == 'ridge':
        if l_selec == 0:
            Delta_part = (2/n)*np.matmul(np.transpose(X2),(np.matmul(X2,s_part) - y2))+l*2*s_part
        else:
            if l_selec == 1:
                Delta_part = (2/n)*np.matmul(np.transpose(X2),(np.matmul(X2,p_part) - y2))-(2/n)*np.matmul(np.transpose(X2),(np.matmul(X2,s_part) - y2)) + 2*l*(p_part-s_part)
            else:
                p_parent_parti = np.ceil(p_parti/2)
                p_parent_part = s_part-(edgesize-edgesize/2**(l_selec-1))*np.ones(d)+(p_parent_parti-1)*(2*edgesize/2**(l_selec-1))
                Delta_part = (2/n)*np.matmul(np.transpose(X2),(np.matmul(X2,p_part) - y2))-(2/n)*np.matmul(np.transpose(X2),(np.matmul(X2,p_parent_part) - y2))+2*l*(p_part-p_parent_part)
    else:
        if l_selec==0:
            Delta_part=(1/n)*np.matmul(np.transpose(X2),np.divide(np.dot(-1*y,np.exp(-1*np.dot(y2,np.matmul(X2,s_part)))),(1+np.exp(-1*np.dot(y2,np.matmul(X2,s_part))))))
        else:
            if l_selec==1:
                Delta_part=(1/n)*np.matmul(np.transpose(X2),np.divide(np.dot(-1*y,np.exp(-1*np.dot(y2,np.matmul(X2,p_part)))),(1+np.exp(-1*np.dot(y2,np.matmul(X2,p_part))))))-\
                (1/n)*np.matmul(np.transpose(X2),np.divide(np.dot(-1*y,np.exp(-1*np.dot(y2,np.matmul(X2,s_part)))),(1+np.exp(-1*np.dot(y2,np.matmul(X2,s_part))))))
            else:
                p_parent_parti=np.ceil(p_parti/2)
                p_parent_part=s_part-(edgesize-edgesize/2**(l_selec-1))*np.ones(d)+(p_parent_parti-1)*(2*edgesize/2**(l_selec-1))
                Delta_part=(1/n)*np.matmul(np.transpose(X2),np.divide(np.dot(-1*y,np.exp(-1*np.dot(y2,np.matmul(X2,p_part)))),(1+np.exp(-1*np.dot(y2,np.matmul(X2,p_part))))))-\
                (1/n)*np.matmul(np.transpose(X2),np.divide(np.dot(-1*y,np.exp(-1*np.dot(y2,np.matmul(X2,p_parent_part)))),(1+np.exp(-1*np.dot(y2,np.matmul(X2,p_parent_part))))))

    return s_part, p_part ,Delta_part, l_selec, p_parti, s_parti
            
def runinstance(ID):
    """
    Running an instance of the problem containing three parts:
    1. Generating data
    2. Running MRE_C (machines side) and collecting the signals at the server
    3. Obtaining theta_hat at the server
    """
    sp.random.seed(ID)
    d = params['d']
    t = params['t']
    mu = params['mu']
    n = params['n']
    m = params['m']
    N = n*m
    sigma = params['sigma']
    sigmaE = params['sigmaE']
    l = params['l']
    edgesize = params['edgesize']
    loss_type = params['loss_type']
    

    #####1.Generating data##############
    if loss_type == 'ridge':
        theta_opt = np.random.uniform(0,1,d)
        X = np.random.normal(mu,sigma,N*d)
        X = np.reshape(X,(N,d))
        E = np.random.normal(mu,sigmaE,N)
        y = np.matmul(X,theta_opt)+E
    else:
        theta_opt=np.random.uniform(0,1,d)
        X=np.random.normal(mu,sigma,N*d)
        X=np.reshape(X,(N,d))
        ytemp=1/(1+np.exp(-np.matmul(X,theta_opt)))
        y=2*np.random.binomial(1,ytemp)-1

    if loss_type == 'ridge':
        theta_all = (sigma**2/(sigma**2+l))*theta_opt#np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)+l*np.eye(d)),np.matmul(np.transpose(X),y))#comptheta(X,y,d)
    else:
        theta_all = theta_opt
    #2. Running MRE_C (machines side) and collecting the signals at the server in thedict
    thedict = {}
    for j in range(m):
        print("Machine:", j) 
        ###########Allocating data to machine j############
        Xtemp = X[(j*n+1):((j+1)*n)+1,:]
        ytemp = y[(j*n+1):((j+1)*n)+1]
        ###########Executing MRE_C on the allocated data##
        s_part,p_part,Delta_part,l_selec,p_parti,s_parti = MRE_C(Xtemp, ytemp, d, n, t, l, edgesize, loss_type)
        ###########Adding results to dictionary###########
        key = np.empty(d+np.size(p_parti)+1).astype(int) #generating key
        key[:d] = s_parti.astype(int) #the first d charachters show the s_part
        key[d] = int(l_selec) #d+1-th charachter shows the selected level
        if l_selec>0:
            key[d+1:] = p_parti.astype(int) #the next d charachter show the p_part
        else:
            key[d+1:] = np.zeros(d)
        if listtokey(key) in thedict:
            thedict[listtokey(key)].append(Delta_part) 
        else:
            thedict[listtokey(key)] = [Delta_part]
            
    maxsize = 0
    for key, value in thedict.items():
        keyarray = keytolist(key)
        if keyarray[d] == 0:
            if np.asarray(value).size>maxsize:
                maxsize = np.asarray(value).size
                keysstar = key
    
    keysstar_array = keytolist(keysstar)
    #3. Obtaining theta_hat at the server by estimating the grad of loss of function over multi-resolution grid
    dictgrad = {}
    temp = np.asarray(thedict.get(keysstar))
    if temp.size > 1:
        dictgrad[keysstar] = sum(temp)/(temp.size/d)
    else:
        dictgrad[keysstar] = temp[0]
    
    mingrad=1000    
    for level in range(1,t+1):
        if level==1:
            for key, value in thedict.items():
                keyarray = keytolist(key)
                if keyarray[:d] == keysstar_array[:d]:
                    if keyarray[d] == 1:
                        temp = np.asarray(value)
                        if temp.size > 1:
                            dictgrad[key] = sum(temp)/(temp.size/d) + dictgrad[keysstar]
                        else:
                            dictgrad[key] = temp[0] + dictgrad[keysstar] 
                        if np.linalg.norm(dictgrad[key],2) < mingrad:
                            mingrad = np.linalg.norm(dictgrad[key],2)
                            keyopt = key
        else:
             for key, value in thedict.items():
                 keyarray = keytolist(key)
                 if keyarray[:d] == keysstar_array[:d]:
                     if keyarray[d] == level:
                         temp = np.asarray(value)
                         key_parent_array = keyarray
                         key_parent_array[d] = level-1
                         key_parent_array[d+1:] = np.ceil(np.asarray(key_parent_array[d+1:])/2).astype(int)
                         key_parent = listtokey(key_parent_array)
                         if key_parent in dictgrad:
                             if temp.size > 1:
                                 dictgrad[key] = sum(temp)/(temp.size/d) + dictgrad[key_parent]
                             else:
                                dictgrad[key] = temp[0] + dictgrad[key_parent]
                             if np.linalg.norm(dictgrad[key],2) < mingrad:
                                 mingrad = np.linalg.norm(dictgrad[key],2)
                                 keyopt = key
    ###########Obtaining theta_hat###############
    keyarray = keytolist(keyopt)
    s_part = 0.5*np.ones(d)#for n=1
    p_parti = np.asarray(keyarray[d+1:])
    level = keyarray[d]
    if level==0:
        output=s_part
    else:
        output = s_part-(edgesize-edgesize/2**(level))*np.ones(d)+(p_parti-1)*(2*edgesize/2**(level))
    
    #return np.linalg.norm(output-theta_opt,2)
    return np.linalg.norm(output-theta_all,2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=10**5)
    parser.add_argument('--n', type=int, default=1) #The current implementation is only for n = 1
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--t', type=int, default=9)
    parser.add_argument('--edgesize', type=float, default=0.5)
    parser.add_argument('--mu', type=int, default=0)
    parser.add_argument('--sigma', type=int, default=1)
    parser.add_argument('--sigmaE', type=float, default=0.1)
    parser.add_argument('--l', type=float, default=0.1)
    parser.add_argument('--numthread', type=int, default=31)#set it to the number of maximum thread minus one
    parser.add_argument('--loss_type', type=str, default='ridge')


    args = parser.parse_args()
    # convert to dictionary
    params = vars(args)
    
    numthread = params['numthread']
    with Pool(numthread) as p:
        accuracy=p.map(runinstance, np.random.choice(10**6, numthread))
    print('errors over instances:', accuracy)
    print('average error:', np.mean(accuracy)) 