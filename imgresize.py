import numpy as np
import scipy 
import scipy.ndimage
#from scipy import ndimage # voor het inladen van plaatjes
import random
import os
    
plaatjes_b = 80
plaatjes_h = 60

projectdir = os.getcwd()
path1= projectdir + '/1raw/'
path0= projectdir + '/0raw/'
dirs1=os.listdir(path1)
dirs0=os.listdir(path0)

savedirtrain0 = projectdir + "/0train/"
savedirtrain1 = projectdir + "/1train/"
savedirtest0 = projectdir + "/0test/"
savedirtest1 = projectdir + "/1test/"

#Zonderblok    
for i in range(len(dirs0)):
    x=random.uniform(0,1)
    if x < 0.9:        
        img = scipy.ndimage.imread(path0 + dirs0[i])
        img = scipy.misc.imresize(img, [plaatjes_h,plaatjes_b])
        scipy.misc.imsave(savedirtrain0 + dirs0[i] , img)
    else:
        img = scipy.ndimage.imread(path0 + dirs0[i])
        img = scipy.misc.imresize(img, [plaatjes_h,plaatjes_b])
        scipy.misc.imsave(savedirtest0 + dirs0[i] , img)
        
 
#MetBlok
for i in range(len(dirs1)):
    x=random.uniform(0,1)
    if x < 0.9:        
        img = scipy.ndimage.imread(path1 + dirs1[i])
        img = scipy.misc.imresize(img, [plaatjes_h,plaatjes_b])
        scipy.misc.imsave(savedirtrain1 + dirs1[i] , img)
    else:
        img = scipy.ndimage.imread(path1 + dirs1[i])
        img = scipy.misc.imresize(img, [plaatjes_h,plaatjes_b])
        scipy.misc.imsave(savedirtest1 + dirs1[i] , img)
    
            
            
        

