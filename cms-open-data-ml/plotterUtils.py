from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def range_float(first,last,n):
    a = range(0,n+1)
    a = map(lambda x : first+x*(last-first)/float(n+1),a)
    return a

<<<<<<< HEAD
def plotMean(data,profile_var_y='jet_jes_ak7',profile_var_x='jet_pt_ak7',n_bins=10,low_bin=100,high_bin=1000,plotRMS=False):    
=======
def plotMean(data,profile_var_y='jet_jes_ak7',profile_var_x='jet_pt_ak7',n_bins=10,low_bin=100,high_bin=1000,plotRMS=False,color='r'):    
>>>>>>> 0ad49aa5b6868e8dcac802635fb96ffe1eb73730
    c=pd.cut(data[profile_var_x],range_float(low_bin,high_bin,n_bins),labels=range(n_bins))
    mean_jes = []
    rms_jes = []
    stderr_jes = []
    for i in range(n_bins):
        mean_jes.append(data[c==i].mean()[profile_var_y])
        rms_jes.append(data[c==i].std()[profile_var_y])
        #print "bin i",i,":",sqrt(len(data[c==i]))
        stderr_jes.append(rms_jes[-1]/sqrt(len(data[c==i])))
    bin_center=range_float(low_bin,high_bin,n_bins)
    bin_center = bin_center[:-1]
    bin_center = map(lambda x : x+(high_bin-low_bin)/n_bins,bin_center)
    #plt.scatter(bin_center,mean_jes)
    if plotRMS :
<<<<<<< HEAD
        plt.errorbar(x=bin_center,y=mean_jes,yerr=rms_jes,fmt='o')
    plt.errorbar(x=bin_center,y=mean_jes,yerr=stderr_jes,fmt='o',color='r')
=======
        plt.errorbar(x=bin_center,y=mean_jes,yerr=rms_jes,fmt='o',color='m')
    plt.errorbar(x=bin_center,y=mean_jes,yerr=stderr_jes,fmt='o',color=color)
>>>>>>> 0ad49aa5b6868e8dcac802635fb96ffe1eb73730
    plt.xlabel(profile_var_x)
    plt.ylabel('Mean '+profile_var_y)

def plot_projection(data,plot_var='jet_pt_ak7',log=True):
    plt.hist(data[plot_var],100,weights=data['mcweight'])
    plt.xlabel(plot_var)
    plt.ylabel('Events')
    if log : 
        plt.gca().set_yscale("log")
