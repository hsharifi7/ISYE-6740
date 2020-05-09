######################################################################
#The codes are based on Python3. 
#Please install numpy, scipy, matplotlib packages before using.
#Thank you for your suggestions!
#
#@version 1.0
#@author CSE6740/CS7641 TA
######################################################################
import numpy as np
import matplotlib.pyplot as plt
import math

def show_image_function(centroids, H, W):
    
    N = int((centroids.shape[1]) / (H * W))
    assert (N == 3 or N == 1)
    
    # Organize the images into rows x cols.
    K = centroids.shape[0]
    COLS = round( math.sqrt(K) )
    ROWS = math.ceil(K / COLS)
    
    COUNT = COLS * ROWS
        
    plt.clf()
    #plt.hold(True)
    # Set up background at value 100 [pixel values 0-255].
    image = np.ones((ROWS * (H + 1), COLS * (W + 1), N)) * 100
    for i in range(0, centroids.shape[0]):
        r = math.floor(i / COLS)
        c = np.mod(i , COLS)
        
        image[(r * (H + 1) + 1):((r + 1) * (H + 1)), \
            (c * (W + 1) + 1):((c + 1) * (W + 1)), :] = \
            centroids[i, :].reshape((H, W, N))
    
    plt.imshow(image.squeeze(), plt.cm.gray)
    
    
    
    
        
        
        
        
        
        
        
    
    
    
