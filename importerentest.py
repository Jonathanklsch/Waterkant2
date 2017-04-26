import numpy as np
import scipy 
#from scipy import ndimage # voor het inladen van plaatjes
import random
import os


def nextbatch(batch_size,plaatjes_breedte,plaatjes_hoogte):
    
    plaatjes_b = plaatjes_breedte
    plaatjes_h = plaatjes_hoogte
    aantal_kanalen=3
    plaatjes = np.zeros([batch_size,  plaatjes_h, plaatjes_b, aantal_kanalen])
    labels = np.zeros([batch_size])
    plaatjesplat = np.zeros([batch_size, plaatjes_h* plaatjes_b* aantal_kanalen])
    #print(plaatjesplat[1][900000])
    
    projectdir = os.getcwd()
    path1= projectdir + "/1test/"
    path0= projectdir + "/0test/"
    dirs1=os.listdir(path1)
    dirs0=os.listdir(path0)
    
    
    #Nu kun je de batch gaan vullen 
    
    for i in range(batch_size):
        x = random.randint(0, 1)
        if  x == 0:
            # gooi een muntje op en bepaal welke dataset je gebruikt...
            #laad een plaatje:
            z1=random.randint(0,len(dirs0)-1)
            #print('z1',z1)        
            plaatjes[i] = scipy.ndimage.imread(path0 + dirs0[z1])
            plaatjesplat[i] = plaatjes[i].flatten()
            #plaatjesplat[i] = np.reshape(plaatjes[i], (plaatjes_b*plaatjes_h*aantal_kanalen))
            
            labels[i] = 0 # of 0, afhankelijk van dat muntje... #we moeten de labels onehot maken
        if x == 1:
            # gooi een muntje op en bepaal welke dataset je gebruikt...
            #laad een plaatje:
            z2=random.randint(0,len(dirs1)-1)
            #print('z2',z2) 
            plaatjes[i] = scipy.ndimage.imread(path1 + dirs1[z2])
            plaatjesplat[i] = plaatjes[i].flatten()
            #plaatjesplat[i] = np.reshape(plaatjes[i], (1,plaatjes_b*plaatjes_h*aantal_kanalen))
            labels[i] = 1 # of 0, afhankelijk van dat muntje...
    
    return (plaatjesplat, labels)

#y= nextbatch(10,28,21)
#w= y.shape
#print('y',y)
#print('w',w)
      

# we denken dat de vorm van de output nu goed is.