#!/usr/bin/env python
# coding: utf-8

# # Updated Performance Measure ($\hat{R}$-HV) for R-EMO Algorithms

# ## Prerequisite

# In[1]:


import numpy as np
import math as ma
import scipy
import random
from pymoo.indicators.hv import HV
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_problem, get_reference_directions
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from scipy.stats import qmc
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from sklearn import preprocessing
from matplotlib.patches import Polygon
import pandas as pd
import xlsxwriter
from pandas import Series, ExcelWriter
from sklearn.metrics.pairwise import euclidean_distances


# ### Function for Computing Points in circle

# In[2]:


def RoI_circle(center, clust, r):
    return np.array([point for point in clust if (center[0] - point[0])**2 + (center[1] - point[1])**2 <= r**2])


# ### Function for Centroid Computation

# In[3]:


def centroid(clust):
    dim = len(clust[0,:])
#     print(dim)
    centr_p = np.zeros([1,dim])
    for i in range(0,dim):
        centr_p[0,i] = np.median(clust[:,i])
#         centr_p[0,i] = np.mean(clust[:,i])
    zp_dist = euclidean_distances(clust, centr_p)
    zp_idx = np.array(zp_dist).argmin()
    centr = clust[zp_idx,:]
    return centr


# ### Function for Radius Computation

# In[4]:


def Radius(clust):
    rad = np.zeros([len(clust),1])
    centr = centroid(clust)
    for i in range(0,len(clust)):      
      rad[i] = ma.dist(centr, clust[i,:])
    r = np.max(rad)
    return r


# ### Function for Normalizing Data

# In[5]:


def Normal(data,UB,LB):
    dim = len(UB)
    normal_data = np.zeros([len(data[:,0]),dim])
    for i in range(0,dim):
        normal_data[:,i] = (data[:,i]-LB[i])/(UB[i]-LB[i])
    return normal_data


# ### Function for UB and LB Computation

# In[6]:


def bounds(data):
    size = len(data[0])
    UB = np.zeros([1,size]); LB = np.zeros([1,size]); 

    for i in range(0,size):
        UB[0,i] = np.max(res.F[:,i])
        LB[0,i] = np.min(res.F[:,i])
        
    return UB, LB


# In[7]:


# UB, LB = bounds(ref_directions)


# ### Existing R-HV Computation

# In[8]:


from sklearn.metrics.pairwise import euclidean_distances
def Exist_RHV_points(centr, clust, ref_points):
    ideal_point = [0, 0]
    zp_dist = euclidean_distances(clust, [centr])
    zp_idx = np.array(zp_dist).argmin()
    zp = clust[zp_idx,:]
    ASF = (zp - ideal_point)/(ref_points - ideal_point)
    k = np.array(ASF).argmax()
    zl = ideal_point + ((zp[k] - ideal_point[k])/(ref_points[k] - ideal_point[k]))*(ref_points - ideal_point)
    clust_RHV = clust + (zl-zp)
    return clust_RHV


# ### Modified R-HV Computation

# In[9]:


from sklearn.metrics.pairwise import euclidean_distances
def Mod_RHV_points(ASF_Sol, clust, ref_points):
    
    dim = len(ref_points)
    ## Compute centroid from cluster
    centr = centroid(clust)
    
    ## Compute distance between ASF Solutions and MCDM Centroids
    d_R = ma.dist(ASF_Sol,centr)

    ## New Centroid on Reference line
    vec = ref_points - ASF_Sol
    centra_new = np.zeros([1,dim])
    centra_new = ASF_Sol + d_R*(vec/np.linalg.norm(vec))

    ## Initializing the Shifted MCDM
    clust_RHV = np.zeros([len(clust),dim])

    ## Computing the Shifted MCDM
    clust_RHV = clust + (centra_new - centr)
    return clust_RHV


# ### Function for Plotting

# In[10]:


def plot(PF,MCDM_solns,ref_points,ASF_Sol,a,b):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 12
    MCDM_solns = MCDM_solns[MCDM_solns[:,1]<a]
    MCDM_solns = MCDM_solns[MCDM_solns[:,0]<b]

    plt.scatter(res.F[:, 0],  res.F[:, 1],  s=15, facecolors='k', edgecolors='k')
    plt.scatter(MCDM_solns[:, 0],  MCDM_solns[:, 1],  s=25, facecolors='r', edgecolors='r')

    plt.scatter(ref_points[:, 0], ref_points[:, 1], s=150,  marker='*', facecolors='b', edgecolors='b')

    plt.xlabel('$f_1$',fontsize=14)
    plt.ylabel('$f_2$',fontsize=14)
    plt.legend(["Pareto Front", "MCDM Solns", "Ref Points"], loc ="upper right",fontsize=12)
    plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)

    plt.scatter(ASF_Sol[0, :],  ASF_Sol[1, :],  s=75, facecolors='violet', edgecolors='k')
    
    ideal_point = [min(res.F[:, 0]), min(res.F[:, 1])] 

    plt.plot([ideal_point[0], ref_points[0,0]], [ideal_point[1], ref_points[0,1]], 'bo', linestyle="--")
    plt.plot([ideal_point[0], ref_points[1,0]], [ideal_point[1], ref_points[1,1]], 'bo', linestyle="--")
#     plt.plot([ideal_point[0], ref_points[2,0]], [ideal_point[1], ref_points[2,1]], 'bo', linestyle="--")
    
    plt.plot([ideal_point[0]], [ideal_point[1]], 'mo', linestyle="--")
    plt.text(1.0*ref_points[0,0], 1.05*ref_points[0,1], "$R_1$", fontsize=14)
    plt.text(1.0*ref_points[1,0], 1.1*ref_points[1,1], "$R_2$", fontsize=14)
#     plt.text(1.02*ref_points[2,0], 0.95*ref_points[2,1], "$R_2$", fontsize=14)
    plt.text(1.02*ideal_point[0],  0.5*ideal_point[1], "$O$", fontsize=14)

    plt.xlabel('$f_1$',fontsize=16)
    plt.ylabel('$f_2$',fontsize=16)
    plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)
    plt.show()


# # PROBLEM DEFINITION STARTS HERE

# In[11]:


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([1, 2**0.5, 2**0.5, 1]),
                         xu=np.array([3, 3, 3, 3]))

    def _evaluate(self, x, out, *args, **kwargs):
        F = 10; E = 200000; L = 200; S = 10; a = F/S;
        f1 = (L*(2*x[0]+ (2**0.5)*x[1] + x[2]**0.5 + x[3])-1.24019913e+03)/1.57847952e+03
        f2 = ((2*F*L/E)*(1/x[0] + (2**0.5)/x[1] - (2**0.5)/x[2] + 1/x[3]) - 3.26475150e-03)/3.65618970e-02
        out["F"] = [f1, f2]
problem = MyProblem()


# In[12]:


ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=100)

# create the algorithm object
algorithm = NSGA3(pop_size=101, ref_dirs=ref_dirs)
#algorithm = NSGA2(pop_size=201)
res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1)


# ### Computing ASF Solutions

# In[13]:


def ASF_solution(ref_point):
    class MyProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=4,
                             n_obj=1,
                             n_constr=0,
                             xl=np.array([1, 2**0.5, 2**0.5, 1]),
                             xu=np.array([3, 3, 3, 3]))

        def _evaluate(self, x, out, *args, **kwargs):
            F = 10; E = 200000; L = 200; S = 10; a = F/S;
            f1 = (L*(2*x[0]+ (2**0.5)*x[1] + x[2]**0.5 + x[3])-1.24019913e+03)/1.57847952e+03
            f2 = ((2*F*L/E)*(1/x[0] + (2**0.5)/x[1] - (2**0.5)/x[2] + 1/x[3]) - 3.26475150e-03)/3.65618970e-02
            
            ideal_point = np.array([0,0]);
            
            f_ASF = (np.max((np.array([f1,f2]) - ideal_point)/(ref_point-ideal_point))
                     + 0.0001*np.sum(np.array([f1,f2])/(ref_point-ideal_point)))            

            
            out["F"] = [f_ASF]
            

    problem_ASF = MyProblem()
    
    # OPTIMIZATION HERE
    
    algorithm = GA(pop_size=10,eliminate_duplicates=True)

    res_ASF = minimize(problem_ASF,
               algorithm,
               seed=1, termination=('n_gen', 50),
               verbose=True)
    return res_ASF


# In[14]:


ASF_Sol = np.zeros([2,2])
ref_p = np.array([[0.4, 0.9],[0.9, 0.4]])
for i in range(0,2):
    res_ASF = ASF_solution(ref_p[i,:])
    ASF_Sol[i,:] = problem.evaluate(res_ASF.X)


# In[15]:


import matplotlib
import matplotlib.pyplot as plt
ref_points = np.array([[0.4, 0.9],[0.9, 0.4]])
plt.figure(figsize=(5, 5))
plt.rcParams['font.size'] = 12
plt.scatter(res.F[:, 0],  res.F[:, 1],  s=15, facecolors='k', edgecolors='k')
plt.scatter(ref_points[:,0], ref_points[:,1], s= 100, marker='*', color='blue', label='Reference Point')
plt.plot([0, ref_points[0,0]], [0, ref_points[0,1]], 'bo', linestyle="--")
plt.plot([0, ref_points[1,0]], [0, ref_points[1,1]], 'bo', linestyle="--")
plt.scatter(ASF_Sol[0, :],  ASF_Sol[1, :],  s=75, facecolors='violet', edgecolors='k')

plt.xlabel('$f_1$',fontsize=16)
plt.ylabel('$f_2$',fontsize=16)
plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)


# ## UB and LB Computation 

# In[16]:


UB = np.zeros([1,2]); LB = np.zeros([1,2]); 

for i in range(0,2):
    UB[0,i] = np.max(res.F[:,i])
    LB[0,i] = np.min(res.F[:,i])
ideal_point = LB[0,:]; nad_point = UB[0,:];


# In[17]:


print(ideal_point)
print(nad_point)
nad_point - ideal_point


# ### R-NSGA-III for MCDM Solutions

# In[18]:


# Define reference points
ref_points = np.array([[0.4, 0.9],[0.9, 0.4]])

algorithm = RNSGA3(
    ref_points=ref_points,
    pop_per_ref_point=20,
    mu=0.1)

resMCDM1 = minimize(problem,
               algorithm=algorithm,
               termination=('n_gen', 100),
               extreme_points_as_reference_points=True,
               seed=10,
               verbose=True)


# In[19]:


plot(res.F,resMCDM1.F,ref_points,ASF_Sol,0.8,0.8)


# ### R-HV Computation

# In[20]:


N = np.zeros([1,2]); N_RoI = np.zeros([1,2]); r = np.zeros([1,2]); 
H_Vol = np.zeros([1,2]); RH_Vol = np.zeros([1,2]); Mod_RH_Vol = np.zeros([1,2]);

MCDM_solns = resMCDM1.F[resMCDM1.F[:,1]<0.8]
MCDM = MCDM_solns[MCDM_solns[:,0]<0.8]

#2-Clusters
clust1 = MCDM[MCDM[:,1]>0.4]; clust2 = MCDM[MCDM[:,1]<0.4]

#Count of MCDM and MCDM in RoI
N[0,:] = [len(clust1),len(clust2)]; N_RoI[0,:] = [len(clust1),len(clust2)]

#Centroid Computation
centro1 = centroid(clust1); centro2 = centroid(clust2)

#Radius Computation
r1 = Radius(clust1); r2 = Radius(clust2)
r[0,:] = np.array([r1, r2])

#HV Computation 
ind = HV(ref_point=ref_points[0,:])
H_Vol[0,0] =  ind(clust1)

ind = HV(ref_point=ref_points[1,:])
H_Vol[0,1] = ind(clust2)

#R-HV Points Computation
RHV_clust1a = Exist_RHV_points(centro1, clust1, ref_points[0,:])
RHV_clust2a = Exist_RHV_points(centro2, clust2, ref_points[1,:])

RHV = np.concatenate([RHV_clust1a,RHV_clust2a],axis=0)

#R-HV Computation
ind = HV(ref_point=ref_points[0,:])
RH_Vol[0,0] =  ind(RHV_clust1a)

ind = HV(ref_point=ref_points[1,:])
RH_Vol[0,1] = ind(RHV_clust2a)


#R-HV Points Computation
Mod_RHV_clust1a = Mod_RHV_points(ASF_Sol[0,:], clust1, ref_points[0,:])
Mod_RHV_clust2a = Mod_RHV_points(ASF_Sol[1,:], clust2, ref_points[1,:])

Mod_RHV = np.concatenate([Mod_RHV_clust1a,Mod_RHV_clust2a],axis=0)

#R-HV Computation
ind = HV(ref_point=ref_points[0,:])
Mod_RH_Vol[0,0] =  ind(Mod_RHV_clust1a)

ind = HV(ref_point=ref_points[1,:])
Mod_RH_Vol[0,1] = ind(Mod_RHV_clust2a)

print(RH_Vol)
print(Mod_RH_Vol)


# ### Distance from ASF point to Ref point

# In[21]:


CR1 = ma.dist(ref_points[0,:],ASF_Sol[0,:])
CR2 = ma.dist(ref_points[1,:],ASF_Sol[1,:])


# ### New R-HV for R-NSGA-III

# In[22]:


res_ASF1 = ASF_solution(centro1)
ASF_sol1 = problem.evaluate(res_ASF1.X)

res_ASF2 = ASF_solution(centro2)
ASF_sol2 = problem.evaluate(res_ASF2.X)

CS1 = ma.dist(centro1,ASF_sol1)
NS3_RHV1 = max(1-CS1/CR1,0)*Mod_RH_Vol[0,0]
CS2 = ma.dist(centro2,ASF_sol2)
NS3_RHV2 = max(1-CS2/CR2,0)*Mod_RH_Vol[0,1]

NS3_RHV = [NS3_RHV1, NS3_RHV2]


# In[23]:


import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 12))
figure, ax = plt.subplots(figsize=(6.5, 6.5))
plt.rcParams['font.size'] = 12
plt.scatter(res.F[:, 0],  res.F[:, 1],  s=5, facecolors='k', edgecolors='k',label="Pareto front")
plt.scatter(ref_points[:,0], ref_points[:,1], s= 100, marker='*', color='blue',label="Ref points")
plt.scatter(MCDM[:,0], MCDM[:,1], s= 10, color='g', edgecolors='g',label="MCDM Solns")
plt.scatter(RHV[:,0], RHV[:,1], s= 10, color='c', edgecolors='c',label="Exist MCDM Shift")
plt.scatter(Mod_RHV[:,0], Mod_RHV[:,1], s= 10, color='r', edgecolors='r',label="New MCDM Shift")
plt.plot([0, ref_points[0,0]], [0, ref_points[0,1]], 'bo', linestyle="--")
plt.plot([0, ref_points[1,0]], [0, ref_points[1,1]], 'bo', linestyle="--")
plt.scatter(ASF_Sol[0, :],  ASF_Sol[1, :],  s=75, facecolors='violet', edgecolors='k')

ax.add_patch(plt.Circle((centro1[0],centro1[1]), r1, facecolor='r', edgecolor='k', alpha=0.2, label="$\delta$-neighbor"))
ax.add_patch(plt.Circle((centro2[0],centro2[1]), r2, facecolor='r', edgecolor='k', alpha=0.2))

HV_cor1=np.array([[np.min(RHV_clust1a[:,0]),np.max(RHV_clust1a[:,1])],[np.max(RHV_clust1a[:,0]),np.min(RHV_clust1a[:,1])],
                    [ref_points[0,0],np.min(RHV_clust1a[:,1])],[ref_points[0,0], ref_points[0,1]], 
                    [np.min(RHV_clust1a[:,0]),ref_points[0,1]]])
p1 = Polygon(HV_cor1, edgecolor='k', facecolor='g', hatch='/', alpha=0.2, label="R-HV")
ax.add_patch(p1)

HV_cor1a=np.array([[np.min(Mod_RHV_clust1a[:,0]),np.max(Mod_RHV_clust1a[:,1])],
                   [np.max(Mod_RHV_clust1a[:,0]),np.min(Mod_RHV_clust1a[:,1])],
                    [ref_points[0,0],np.min(Mod_RHV_clust1a[:,1])],[ref_points[0,0], ref_points[0,1]], 
                    [np.min(Mod_RHV_clust1a[:,0]),ref_points[0,1]]])
p1a = Polygon(HV_cor1a, linestyle='-.', linewidth=2.0, edgecolor='k', facecolor='y', hatch='-', alpha=0.2, label=r"$\tilde{R}$-HV")
ax.add_patch(p1a)

HV_cor2=np.array([[np.min(RHV_clust2a[:,0]),np.max(RHV_clust2a[:,1])],[np.max(RHV_clust2a[:,0]),np.min(RHV_clust2a[:,1])],
                    [ref_points[1,0],np.min(RHV_clust2a[:,1])],[ref_points[1,0], ref_points[1,1]], 
                    [np.min(RHV_clust2a[:,0]),ref_points[1,1]]])
p2 = Polygon(HV_cor2, edgecolor='k', facecolor='g', hatch='/', alpha=0.2)
ax.add_patch(p2)

HV_cor2a=np.array([[np.min(Mod_RHV_clust2a[:,0]),np.max(Mod_RHV_clust2a[:,1])],
                   [np.max(Mod_RHV_clust2a[:,0]),np.min(Mod_RHV_clust2a[:,1])],
                    [ref_points[1,0],np.min(Mod_RHV_clust2a[:,1])],[ref_points[1,0], ref_points[1,1]], 
                    [np.min(Mod_RHV_clust2a[:,0]),ref_points[1,1]]])
p2a = Polygon(HV_cor2a, linestyle='-.', linewidth=2.0, edgecolor='k', facecolor='y', hatch='-', alpha=0.2)
ax.add_patch(p2a)

plt.legend(loc="upper right", fontsize=14)

plt.plot([0.0], [0.0], 'mo', linestyle="--")
plt.text(0.45, 0.9, "$R_1$", fontsize=16)
plt.text(0.95, 0.4, "$R_2$", fontsize=16)
plt.text(-0.045, -0.045, "$O$", fontsize=16)

plt.xlabel('$f_1$',fontsize=18)
plt.ylabel('$f_2$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)
plt.savefig("Ex1_NSGA.pdf", format="pdf", bbox_inches="tight")
# plt.show()


# In[24]:


print(RH_Vol)
print(NS3_RHV)


# In[25]:


print(N)
print(N_RoI)


# ### RVEA for MCDM Solution

# In[26]:


reference_directions = resMCDM1.algorithm.survival.ref_dirs
UB1, LB1 = bounds(reference_directions)

reference_directions = reference_directions[reference_directions[:,1]<0.8,:]
reference_directions = reference_directions[reference_directions[:,0]<0.8,:]

ref_directions = Normal(reference_directions ,UB1[0],LB1[0])
len(ref_directions)


# In[27]:


# ref_directions1


# In[28]:


algorithm = RVEA(ref_directions, alpha=2.0, adapt_freq=0.0001)

resMCDM2 = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               verbose=True)


# In[29]:


plot(res.F,resMCDM2.F,ref_points,ASF_Sol,0.8,0.8)


# ### R-HV Computation

# In[30]:


N = np.zeros([1,2]); N_RoI = np.zeros([1,2]); 
H_Vol = np.zeros([1,2]); RH_Vol = np.zeros([1,2]); Mod_RH_Vol = np.zeros([1,2]);

MCDM_solns = resMCDM2.F[resMCDM2.F[:,1]<0.8]
MCDM = MCDM_solns[MCDM_solns[:,0]<0.8]

#2-Clusters
clust1 = MCDM[MCDM[:,1]>0.4]; clust2 = MCDM[MCDM[:,1]<0.4]

#Count of MCDM and MCDM in RoI
N[0,:] = [len(clust1),len(clust2)]; 

#Centroid Computation
centro1 = centroid(clust1); centro2 = centroid(clust2)

clust1a = RoI_circle(centro1, clust1, r[0,0])
clust2a = RoI_circle(centro2, clust2, r[0,1])
    
#Count of MCDM in RoI
N_RoI[0,:] = [len(clust1a),len(clust2a)]

#HV Computation 
ind = HV(ref_point=ref_points[0,:])
H_Vol[0,0] =  ind(clust1)

ind = HV(ref_point=ref_points[1,:])
H_Vol[0,1] = ind(clust2)

#R-HV Points Computation
RHV_clust1a = Exist_RHV_points(centro1, clust1a, ref_points[0,:])
RHV_clust2a = Exist_RHV_points(centro2, clust2a, ref_points[1,:])

RHV = np.concatenate([RHV_clust1a,RHV_clust2a],axis=0)

#R-HV Computation
ind = HV(ref_point=ref_points[0,:])
RH_Vol[0,0] =  ind(RHV_clust1a)

ind = HV(ref_point=ref_points[1,:])
RH_Vol[0,1] = ind(RHV_clust2a)

#R-HV Points Computation
Mod_RHV_clust1a = Mod_RHV_points(ASF_Sol[0,:], clust1, ref_points[0,:])
Mod_RHV_clust2a = Mod_RHV_points(ASF_Sol[1,:], clust2, ref_points[1,:])

Mod_RHV = np.concatenate([Mod_RHV_clust1a,Mod_RHV_clust2a],axis=0)

#R-HV Computation
ind = HV(ref_point=ref_points[0,:])
Mod_RH_Vol[0,0] =  ind(Mod_RHV_clust1a)

ind = HV(ref_point=ref_points[1,:])
Mod_RH_Vol[0,1] = ind(Mod_RHV_clust2a)

print(RH_Vol)
print(Mod_RH_Vol)


# In[31]:


# Mod_RHV


# ### New R-HV for RVEA

# In[32]:


res_ASF1 = ASF_solution(centro1)
ASF_sol1 = problem.evaluate(res_ASF1.X)

res_ASF2 = ASF_solution(centro2)
ASF_sol2 = problem.evaluate(res_ASF2.X)

CS1 = ma.dist(centro1,ASF_sol1)
RV_RHV1 = max(1-CS1/CR1,0)*Mod_RH_Vol[0,0]
CS2 = ma.dist(centro2,ASF_sol2)
RV_RHV2 = max(1-CS2/CR2,0)*Mod_RH_Vol[0,1]

RV_RHV = [RV_RHV1, RV_RHV2]


# In[33]:


print(RH_Vol)
print(RV_RHV)


# In[34]:


print(N)
print(N_RoI)


# In[35]:


import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 12))
figure, ax = plt.subplots(figsize=(6.5, 6.5))
plt.rcParams['font.size'] = 12
plt.scatter(res.F[:, 0],  res.F[:, 1],  s=5, facecolors='k', edgecolors='k',label="Pareto front")
plt.scatter(ref_points[:,0], ref_points[:,1], s= 100, marker='*', color='blue',label="Ref points")
plt.scatter(MCDM[:,0], MCDM[:,1], s= 10, color='g', edgecolors='g',label="MCDM Solns")
plt.scatter(RHV[:,0], RHV[:,1], s= 10, color='c', edgecolors='c',label="Exist MCDM Shift")
plt.scatter(Mod_RHV[:,0], Mod_RHV[:,1], s= 10, color='r', edgecolors='r',label="New MCDM Shift")
plt.plot([0, ref_points[0,0]], [0, ref_points[0,1]], 'bo', linestyle="--")
plt.plot([0, ref_points[1,0]], [0, ref_points[1,1]], 'bo', linestyle="--")

centr_HV = centroid(RHV_clust2a); centr_Mod_HV = centroid(Mod_RHV_clust2a)
plt.plot([centr_HV[0], ASF_sol2[0]], [centr_HV[1], ASF_sol2[1]], 'k', linestyle="--")
plt.scatter(ASF_Sol[0, :],  ASF_Sol[1, :],  s=75, facecolors='violet', edgecolors='k')
plt.scatter(centr_Mod_HV[0], centr_Mod_HV[1],  s=30, facecolors='b', edgecolors='b')
plt.scatter(ASF_sol2[0], ASF_sol2[1],  s=30, facecolors='b', edgecolors='b')
plt.scatter(centr_HV[0], centr_HV[1],  s=30, facecolors='b', edgecolors='b')



ax.add_patch(plt.Circle((centro1[0],centro1[1]), r1, facecolor='r', edgecolor='k', alpha=0.2, label="$\delta$-neighbor"))
ax.add_patch(plt.Circle((centro2[0],centro2[1]), r2, facecolor='r', edgecolor='k', alpha=0.2))
ax.add_patch(plt.Circle((ASF_Sol[1,0],ASF_Sol[1,1]), ma.dist(centro2,ASF_Sol[1,:]), facecolor='none', 
                        edgecolor='k',linestyle='-.'))

HV_cor1=np.array([[np.min(RHV_clust1a[:,0]),np.max(RHV_clust1a[:,1])],[np.max(RHV_clust1a[:,0]),np.min(RHV_clust1a[:,1])],
                    [ref_points[0,0],np.min(RHV_clust1a[:,1])],[ref_points[0,0], ref_points[0,1]], 
                    [np.min(RHV_clust1a[:,0]),ref_points[0,1]]])
p1 = Polygon(HV_cor1, edgecolor='k', facecolor='g', hatch='/', alpha=0.2, label="R-HV")
ax.add_patch(p1)

HV_cor1a=np.array([[np.min(Mod_RHV_clust1a[:,0]),np.max(Mod_RHV_clust1a[:,1])],
                   [np.max(Mod_RHV_clust1a[:,0]),np.min(Mod_RHV_clust1a[:,1])],
                    [ref_points[0,0],np.min(Mod_RHV_clust1a[:,1])],[ref_points[0,0], ref_points[0,1]], 
                    [np.min(Mod_RHV_clust1a[:,0]),ref_points[0,1]]])
p1a = Polygon(HV_cor1a, linestyle='-.', linewidth=2.0, edgecolor='k', facecolor='y', hatch='-', alpha=0.2, label=r"$\tilde{R}$-HV")
ax.add_patch(p1a)

HV_cor2=np.array([[np.min(RHV_clust2a[:,0]),np.max(RHV_clust2a[:,1])],[np.max(RHV_clust2a[:,0]),np.min(RHV_clust2a[:,1])],
                    [ref_points[1,0],np.min(RHV_clust2a[:,1])],[ref_points[1,0], ref_points[1,1]], 
                    [np.min(RHV_clust2a[:,0]),ref_points[1,1]]])
p2 = Polygon(HV_cor2, edgecolor='k', facecolor='g', hatch='/', alpha=0.2)
ax.add_patch(p2)

HV_cor2a=np.array([[np.min(Mod_RHV_clust2a[:,0]),np.max(Mod_RHV_clust2a[:,1])],
                   [np.max(Mod_RHV_clust2a[:,0]),np.min(Mod_RHV_clust2a[:,1])],
                    [ref_points[1,0],np.min(Mod_RHV_clust2a[:,1])],[ref_points[1,0], ref_points[1,1]], 
                    [np.min(Mod_RHV_clust2a[:,0]),ref_points[1,1]]])
p2a = Polygon(HV_cor2a, linestyle='-.', linewidth=2.0, edgecolor='k', facecolor='y', hatch='-', alpha=0.2)
ax.add_patch(p2a)

plt.legend(loc="upper right", fontsize=14)

plt.plot([0.0], [0.0], 'mo', linestyle="--")
plt.text(0.45, 0.9, "$R_1$", fontsize=16)
plt.text(0.95, 0.4, "$R_2$", fontsize=16)
plt.text(-0.045, -0.045, "$O$", fontsize=16)

plt.xlabel('$f_1$',fontsize=18)
plt.ylabel('$f_2$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)
plt.savefig("Ex1_RVEA.pdf", format="pdf", bbox_inches="tight")
# plt.show()


# ### C-TAEA for MCDM Solutions

# In[36]:


# reference_directions_C3a
algorithm = CTAEA(ref_dirs=ref_directions)

# execute the optimization
resMCDM3 = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True
               )


# In[37]:


plot(res.F,resMCDM3.F,ref_points,ASF_Sol,0.8,0.8)


# ### R-HV Computation

# In[38]:


N = np.zeros([1,2]); N_RoI = np.zeros([1,2]); 
H_Vol = np.zeros([1,2]); RH_Vol = np.zeros([1,2]); Mod_RH_Vol = np.zeros([1,2]);

MCDM_solns = resMCDM3.F[resMCDM3.F[:,1]<0.8]
MCDM = MCDM_solns[MCDM_solns[:,0]<0.8]

#2-Clusters
clust1 = MCDM[MCDM[:,1]>0.4]; clust2 = MCDM[MCDM[:,1]<0.4]

#Count of MCDM and MCDM in RoI
N[0,:] = [len(clust1),len(clust2)]; 

#Centroid Computation
centro1 = centroid(clust1); centro2 = centroid(clust2)

clust1a = RoI_circle(centro1, clust1, r[0,0])
clust2a = RoI_circle(centro2, clust2, r[0,1])
    
#Count of MCDM in RoI
N_RoI[0,:] = [len(clust1a),len(clust2a)]

#HV Computation 
ind = HV(ref_point=ref_points[0,:])
H_Vol[0,0] =  ind(clust1)

ind = HV(ref_point=ref_points[1,:])
H_Vol[0,1] = ind(clust2)

#R-HV Points Computation
RHV_clust1a = Exist_RHV_points(centro1, clust1a, ref_points[0,:])
RHV_clust2a = Exist_RHV_points(centro2, clust2a, ref_points[1,:])

RHV = np.concatenate([RHV_clust1a,RHV_clust2a],axis=0)

#R-HV Computation
ind = HV(ref_point=ref_points[0,:])
RH_Vol[0,0] =  ind(RHV_clust1a)

ind = HV(ref_point=ref_points[1,:])
RH_Vol[0,1] = ind(RHV_clust2a)

#R-HV Points Computation
Mod_RHV_clust1a = Mod_RHV_points(ASF_Sol[0,:], clust1, ref_points[0,:])
Mod_RHV_clust2a = Mod_RHV_points(ASF_Sol[1,:], clust2, ref_points[1,:])

Mod_RHV = np.concatenate([Mod_RHV_clust1a,Mod_RHV_clust2a],axis=0)

#R-HV Computation
ind = HV(ref_point=ref_points[0,:])
Mod_RH_Vol[0,0] =  ind(Mod_RHV_clust1a)

ind = HV(ref_point=ref_points[1,:])
Mod_RH_Vol[0,1] = ind(Mod_RHV_clust2a)

print(RH_Vol)
print(Mod_RH_Vol)


# ### New R-HV for C-TAEA

# In[39]:


res_ASF1 = ASF_solution(centro1)
ASF_sol1 = problem.evaluate(res_ASF1.X)

res_ASF2 = ASF_solution(centro2)
ASF_sol2 = problem.evaluate(res_ASF2.X)

CS1 = ma.dist(centro1,ASF_sol1)
CTA_RHV1 = max(1-CS1/CR1,0)*Mod_RH_Vol[0,0]
CS2 = ma.dist(centro2,ASF_sol2)
CTA_RHV2 = max(1-CS2/CR2,0)*Mod_RH_Vol[0,1]

CTA_RHV = [CTA_RHV1, CTA_RHV2]


# In[40]:


print(RH_Vol)
print(CTA_RHV)


# In[41]:


print(N)
print(N_RoI)


# In[42]:


import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 12))
figure, ax = plt.subplots(figsize=(6.5, 6.5))
plt.rcParams['font.size'] = 12
plt.scatter(res.F[:, 0],  res.F[:, 1],  s=5, facecolors='k', edgecolors='k',label="Pareto front")
plt.scatter(ref_points[:,0], ref_points[:,1], s= 100, marker='*', color='blue',label="Ref points")
plt.scatter(MCDM[:,0], MCDM[:,1], s= 10, color='g', edgecolors='g',label="MCDM Solns")
plt.scatter(RHV[:,0], RHV[:,1], s= 10, color='c', edgecolors='c',label="Exist MCDM Shift")
plt.scatter(Mod_RHV[:,0], Mod_RHV[:,1], s= 10, color='r', edgecolors='r',label="New MCDM Shift")
plt.plot([0, ref_points[0,0]], [0, ref_points[0,1]], 'bo', linestyle="--")
plt.plot([0, ref_points[1,0]], [0, ref_points[1,1]], 'bo', linestyle="--")
plt.scatter(ASF_Sol[0, :],  ASF_Sol[1, :],  s=75, facecolors='violet', edgecolors='k')

ax.add_patch(plt.Circle((centro1[0],centro1[1]), r1, facecolor='r', edgecolor='k', alpha=0.2, label="$\delta$-neighbor"))
ax.add_patch(plt.Circle((centro2[0],centro2[1]), r2, facecolor='r', edgecolor='k', alpha=0.2))

HV_cor1=np.array([[np.min(RHV_clust1a[:,0]),np.max(RHV_clust1a[:,1])],[np.max(RHV_clust1a[:,0]),np.min(RHV_clust1a[:,1])],
                    [ref_points[0,0],np.min(RHV_clust1a[:,1])],[ref_points[0,0], ref_points[0,1]], 
                    [np.min(RHV_clust1a[:,0]),ref_points[0,1]]])
p1 = Polygon(HV_cor1, edgecolor='k', facecolor='g', hatch='/', alpha=0.2, label="R-HV")
ax.add_patch(p1)

HV_cor1a=np.array([[np.min(Mod_RHV_clust1a[:,0]),np.max(Mod_RHV_clust1a[:,1])],
                   [np.max(Mod_RHV_clust1a[:,0]),np.min(Mod_RHV_clust1a[:,1])],
                    [ref_points[0,0],np.min(Mod_RHV_clust1a[:,1])],[ref_points[0,0], ref_points[0,1]], 
                    [np.min(Mod_RHV_clust1a[:,0]),ref_points[0,1]]])
p1a = Polygon(HV_cor1a, linestyle='-.', linewidth=2.0, edgecolor='k', facecolor='y', hatch='-', alpha=0.2, label=r"$\tilde{R}$-HV")
ax.add_patch(p1a)

HV_cor2=np.array([[np.min(RHV_clust2a[:,0]),np.max(RHV_clust2a[:,1])],[np.max(RHV_clust2a[:,0]),np.min(RHV_clust2a[:,1])],
                    [ref_points[1,0],np.min(RHV_clust2a[:,1])],[ref_points[1,0], ref_points[1,1]], 
                    [np.min(RHV_clust2a[:,0]),ref_points[1,1]]])
p2 = Polygon(HV_cor2, edgecolor='k', facecolor='g', hatch='/', alpha=0.2)
ax.add_patch(p2)

HV_cor2a=np.array([[np.min(Mod_RHV_clust2a[:,0]),np.max(Mod_RHV_clust2a[:,1])],
                   [np.max(Mod_RHV_clust2a[:,0]),np.min(Mod_RHV_clust2a[:,1])],
                    [ref_points[1,0],np.min(Mod_RHV_clust2a[:,1])],[ref_points[1,0], ref_points[1,1]], 
                    [np.min(Mod_RHV_clust2a[:,0]),ref_points[1,1]]])
p2a = Polygon(HV_cor2a, linestyle='-.', linewidth=2.0, edgecolor='k', facecolor='y', hatch='-', alpha=0.2)
ax.add_patch(p2a)

plt.legend(loc="upper right", fontsize=14)

plt.plot([0.0], [0.0], 'mo', linestyle="--")
plt.text(0.45, 0.9, "$R_1$", fontsize=16)
plt.text(0.95, 0.4, "$R_2$", fontsize=16)
plt.text(-0.045, -0.045, "$O$", fontsize=16)

plt.xlabel('$f_1$',fontsize=18)
plt.ylabel('$f_2$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)
plt.savefig("Ex1_CTAEA.pdf", format="pdf", bbox_inches="tight")
# plt.show()


# In[ ]:




