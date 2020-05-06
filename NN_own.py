# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:25:40 2019

@author: Sumanth
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
'''nl = input("number of layers: ")
nuh = input("number of units in hidden layer: ")'''

nl = 2
nuh = 100
epoch = 100


mnist = pd.read_csv('/Users/Sumanth/Desktop/Semester_5/ELL_409/Assignment3/2017EE10451.csv', header=None)


mnist = np.array(mnist)

x,xt,y,yt = train_test_split(mnist[0:3000,0:784],mnist[0:3000,784:],train_size=0.9,random_state=42)
print(x.shape, y.shape)
print(xt.shape, yt.shape)
#print(len(y))
 
NormC = 255*0.99+0.01
x = np.asfarray(x)/NormC
#x = np.array(x).T
nl = int(nl)
nuh = int(nuh)
exvec = [ ]
for i in range(len(y)):
    labelvec = np.arange(10)*1.
    for j in range(0, 10):
        if(labelvec[j] == y[i]):
            labelvec[j] = 0.99
        else :
            labelvec[j] = 0.01  
    exvec.append(labelvec)
#print(exvec)            

xt=np.asfarray(xt)/NormC

#activation function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def relu(x):
    return np.maximum(0,x)

def sigmoid_det(z):
    return z*(1-z)

def relu_det(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

#Neuralnetwork Class......

class neuralNetwork:

#Initialiasing variables
     def initial(self,x,y,l,lu):
         self.x = x
         self.y = y
         self.l = l
         
         self.l = int(l)
         lu = int(lu)
         
         self.initweights(l,lu)
         
#Initialising weights
     def initweights(self,l,lu):
         self.w = []
         k =0
         while(k<(l+1)):
             if (k==0):
                 r = lu
                 c = 784
                 wm = np.random.randn(r, c)
                 self.w.append(wm)
             elif (k==l):
                 r = 10
                 c = lu
                 wm = np.random.randn(r, c)
                 self.w.append(wm)
             else:
                r = lu
                c = lu
                wm = np.random.randn(r, c)
                self.w.append(wm)
                
             k = k+1
             
#Forward and backward propogation
             
     def feedandback (self,x,y):
         
         x = np.array(x).T
         temp = [x]
#         print("temp: ", temp)
         k = 0
         while(k<(self.l+1)):
            inv = temp[-1] 
           # print('k: ', k)
            #print(self.w[k].shape)
            p = np.dot(self.w[k],inv)
            self.otv = relu(p)
            temp.append(self.otv)
            k = k+1
              
            
            
# backPropaogation starts
            
         m = self.l+1
         y = np.expand_dims(np.array(y).T,axis=-1)
         oerr = y - np.expand_dims(self.otv,axis=-1)
         while(m>0):
            # print('m: ', m)
             self.otv = temp[m]
             inv = temp[m-1]
            # print(inv)
             tmp_delta = oerr*relu_det(np.expand_dims(self.otv,axis=-1))
             inv = np.expand_dims(inv,axis=-1)
             self.w[m-1] = self.w[m-1]+(0.001)*(np.dot(tmp_delta,inv.T))
             oerr = np.dot(self.w[m-1].T,tmp_delta)
             m = m-1


     def test(self, invec):
         
          invect = np.array(invec).T
          kk = 1
       
          while (kk<(self.l+2)):
             # print('kk:' ,kk)
              uu = np.dot(self.w[kk-1],invect)

              outvec = relu(uu)
#              print('kk: ',kk,'outvec: ',outvec)
              invect = outvec
    
           
              kk=kk + 1

          return outvec
      

if __name__ == "__main__":
   

    Nnobj=neuralNetwork()
    Nnobj.initial(x,y,nl,nuh)
   
    for k in range (epoch):
        for i in range(len(y)):
            Nnobj.feedandback(x[i],exvec[i])
    jj=0
    out = []
    for jj in range(len(xt)):

            res=(Nnobj.test(xt[jj]))
            out.append(np.argmax(res))
    correct_classifications = 0
    for i in range(len(yt)):
        if (out[i] == yt[i]):
            correct_classifications+=1
    accuracy = correct_classifications*100/len(yt)
    print(accuracy)       
       
def main():
    main()  
     
        
             
             

             
            
            
            
            
            
             
          
         
             
            
                 
                 
             
         

    

                
        
    
    




