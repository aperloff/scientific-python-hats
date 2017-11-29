import pandas as pd
import numpy as np
import h5py
from MLJEC_MCTruth_Util import rotate_and_reflect,rotate_all
import matplotlib as mlp
import matplotlib.pyplot as plt
import itertools 

inputs = {}
inputs['QCD80'] = ['../../CMSOpenData/output_QCD120/params0.npy_job0_file0.npy']
inputs['QCD120'] = ['../../CMSOpenData/output_QCD170/params0.npy_job0_file0.npy']
inputs['QCD170'] = ['../../CMSOpenData/output_QCD300/params0.npy_job0_file0.npy']
sample_dfs={}
for key, input_files in inputs.iteritems():
    for in_file in input_files:
        print in_file
        try:
            sample_dfs[key] = pd.DataFrame(np.load(in_file))
        except ValueError:
            print 'bad file: %s'%in_file     
            
df = pd.concat([sample_dfs['QCD80'],sample_dfs['QCD120'],sample_dfs['QCD170']])
#df = df.iloc[:1000]
df.head()

df_vec_dict = {}
df_new=None
group_list=['ak7pfcand_eta','ak7pfcand_phi','ak7pfcand_pt','ak7pfcand_charge','ak7pfcand_id']
for v in group_list : 
    print v
    df_vec_dict[v] = pd.DataFrame(df.groupby(['event','run','lumi','ak7pfcand_ijet'])[v].apply(list))
    df_vec_dict[v].reset_index(inplace=True)
    if df_new is None:
        print "None"
        df_new = df_vec_dict[v]
        df_new.set_index(['event','run','lumi','ak7pfcand_ijet'])
    else :
        print "not None"
        df_new = df_new.join(df_vec_dict[v].set_index(['event','run','lumi','ak7pfcand_ijet']),on=['event','run','lumi','ak7pfcand_ijet'])
df_new.head()    

df_no_pf_cands = df.drop(group_list,axis=1).drop_duplicates()
df_new = df_new.join(df_no_pf_cands.set_index(['event','run','lumi','ak7pfcand_ijet']),on=['event','run','lumi','ak7pfcand_ijet'])

df_new['pfcand_centered_phi'] = map(lambda x : np.subtract(x,x[0]),df_new['ak7pfcand_phi'])
df_new['pfcand_centered_eta'] = map(lambda x : np.subtract(x,x[0]),df_new['ak7pfcand_eta'])
df_new['ak7pfcand_id'] = df_new.apply(lambda row : map(np.abs,row['ak7pfcand_id']),axis=1)

temp_vec = df_new.apply(lambda row : rotate_and_reflect(row['pfcand_centered_eta'],row['pfcand_centered_phi'],row['ak7pfcand_pt']),axis=1)
temp_vec = map(list,temp_vec)
temp_vec = map(list,zip(*temp_vec))

print type(temp_vec)
print "temp_vec:",len(temp_vec)
print "df:",len(df_new['pfcand_centered_phi'])

df_new['pfcand_riz'] = pd.Series(temp_vec)[0]
df_new['pfcand_riy'] = pd.Series(temp_vec)[1]

df_new['jet_image'] = df_new.apply(lambda row : [np.histogram2d(row['pfcand_riz'],row['pfcand_riy'],weights=row['ak7pfcand_eta'],bins=[np.arange(-1.4,1.4+0.00001,2.8/30.),np.arange(-1.4,1.4+0.00001,2.8/30.)])[0]],axis=1)

df_new['photon_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==22,row['ak7pfcand_id']),axis=1)
df_new['electron_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==11,row['ak7pfcand_id']),axis=1)
df_new['muon_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==13,row['ak7pfcand_id']),axis=1)
df_new['charHad_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==211,row['ak7pfcand_id']),axis=1)
df_new['neutHad_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==130,row['ak7pfcand_id']),axis=1)

df_new['jet_photon_mult_ak7'] = df_new.apply(lambda row : sum(row['photon_filter']),axis=1)
df_new['jet_electron_mult_ak7'] = df_new.apply(lambda row : sum(row['electron_filter']),axis=1)
df_new['jet_muon_mult_ak7'] = df_new.apply(lambda row : sum(row['muon_filter']),axis=1)
df_new['jet_charHad_mult_ak7'] = df_new.apply(lambda row : sum(row['charHad_filter']),axis=1)
df_new['jet_neutHad_mult_ak7'] = df_new.apply(lambda row : sum(row['neutHad_filter']),axis=1)

df_new['jet_photon_frac_ak7'] = df_new.apply(lambda row : sum(np.multiply(row['photon_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
df_new['jet_electron_frac_ak7'] = df_new.apply(lambda row : sum(np.multiply(row['electron_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
df_new['jet_muon_frac_ak7'] = df_new.apply(lambda row : sum(np.multiply(row['muon_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
df_new['jet_charHad_frac_ak7'] = df_new.apply(lambda row : sum(np.multiply(row['charHad_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
df_new['jet_neutHad_frac_ak7'] = df_new.apply(lambda row : sum(np.multiply(row['neutHad_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)


df_new.to_pickle("new.pkl")

