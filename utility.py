#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Bio
from Bio.PDB import PDBParser
from scipy.spatial.distance import euclidean
import itertools


# In[ ]:


# Calculates Rotation Matrix given euler angles.
def RotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


# In[ ]:


# protein radius based on distance matrix 
def approx_radius(coord):
    dist = np.zeros((len(coord), len(coord)))
    for i in range(len(coord)):
        for j in range(0,len(coord)): 
            dist[i][j]=np.double(euclidean(coord[i],coord[j]))  
    approx_radius = np.argmax(dist)/2
    return approx_radius


# In[ ]:


def new_coord_moving(coordi, trans, theta, mean_move):
    
    coordi_o = coordi - mean_move # move the moving protein to the origin 
    
    R = RotationMatrix(theta)
    coordi_r = []
    for ii in range(len(coordi)):
        coordi_r.append(np.dot(R, coordi_o[ii])) # rotate
    
    coordi_t = coordi_r + mean_move + trans
#     coordi_t = coordi_r + trans # starting point: 2*(r1+r2) + trans
    return coordi_t


# In[ ]:


def new_dist(coord1, coord2):
    new_dist = np.zeros((len(coord1), len(coord2)))
    for ii in range(len(coord1)):
        for jj in range(len(coord2)):
            new_dist[ii][jj] = np.double(euclidean(coord1[ii], coord2[jj]))
    return new_dist


# In[ ]:




