import root_numpy as rnp
import pandas as pd

def init_jra(start_=0,stop_=100000):
    a = rnp.root2array(filenames='AlexxTrees/JRA.root',treename='ak4pfchs/t',start=start_,stop=stop_)
    df = pd.DataFrame(a)

    #print len(a)
    #print df.head()
    
    return df

#9873664
for i in range(100):
    print "i: {0}".format(i)
    df = init_jra(100000*i,100000*i+100000)
    df.to_pickle("AlexxTrees/JRA_pandas_df_{0}.pkl".format(i))
