import pandas as pd
import numpy as np
import h5py
from MLJEC_MCTruth_Util import rotate_and_reflect,rotate_all
import matplotlib as mlp
import matplotlib.pyplot as plt
import itertools 

inputs = {}
inputs['QCD15'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_15_30.npy']
inputs['QCD30'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_30_50.npy']
inputs['QCD50'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_50_80.npy']
inputs['QCD80'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_80_120.npy']
inputs['QCD120'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_120_170.npy']
inputs['QCD170'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_170_300.npy']
inputs['QCD300'] = ['/Users/awhitbe1/Dropbox/multivariateJECs/ak5_data/QCD_pt_300_470.npy']
sample_dfs={}
for key, input_files in inputs.iteritems():
    for in_file in input_files:
        print in_file
        try:
            sample_dfs[key] = pd.DataFrame(np.load(in_file))
        except ValueError:
            print 'bad file: %s'%in_file     
            
df = pd.concat([sample_dfs['QCD15'],
                sample_dfs['QCD30'],
                sample_dfs['QCD50'],
                sample_dfs['QCD80'],
                sample_dfs['QCD120'],
                sample_dfs['QCD170'],
                sample_dfs['QCD300']

                ])
#df = df.iloc[:500000]
df.head()

df_vec_dict = {}
df_new=None
group_list=['ak5pfcand_eta','ak5pfcand_phi','ak5pfcand_pt','ak5pfcand_charge','ak5pfcand_id']
for v in group_list : 
    print v
    df_vec_dict[v] = pd.DataFrame(df.groupby(['event','run','lumi','ak5pfcand_ijet'])[v].apply(list))
    df_vec_dict[v].reset_index(inplace=True)
    if df_new is None:
        print "None"
        df_new = df_vec_dict[v]
        df_new.set_index(['event','run','lumi','ak5pfcand_ijet'])
    else :
        print "not None"
        df_new = df_new.join(df_vec_dict[v].set_index(['event','run','lumi','ak5pfcand_ijet']),on=['event','run','lumi','ak5pfcand_ijet'])
   
df_no_pf_cands = df.drop(group_list,axis=1).drop_duplicates()
df_new = df_new.join(df_no_pf_cands.set_index(['event','run','lumi','ak5pfcand_ijet']),on=['event','run','lumi','ak5pfcand_ijet'])

df_new['pfcand_centered_phi'] = map(lambda x : np.subtract(x,x[0]),df_new['ak5pfcand_phi'])
df_new['pfcand_centered_eta'] = map(lambda x : np.subtract(x,x[0]),df_new['ak5pfcand_eta'])
df_new['ak5pfcand_id'] = df_new.apply(lambda row : map(np.abs,row['ak5pfcand_id']),axis=1)

temp_vec = df_new.apply(lambda row : rotate_and_reflect(row['pfcand_centered_eta'],row['pfcand_centered_phi'],row['ak5pfcand_pt']),axis=1)
temp_vec = map(list,temp_vec)
temp_vec = map(list,zip(*temp_vec))

print type(temp_vec)
print "temp_vec:",len(temp_vec)
print "df:",len(df_new['pfcand_centered_phi'])

df_new['pfcand_riz'] = pd.Series(temp_vec)[0]
df_new['pfcand_riy'] = pd.Series(temp_vec)[1]

df_new['jet_image'] = df_new.apply(lambda row : [np.histogram2d(row['pfcand_riz'],row['pfcand_riy'],weights=row['ak5pfcand_eta'],bins=[np.arange(-1.4,1.4+0.00001,2.8/30.),np.arange(-1.4,1.4+0.00001,2.8/30.)])[0]],axis=1)

df_new['photon_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==22,row['ak5pfcand_id']) if hasattr(row['ak5pfcand_id'], '__iter__') else False,axis=1)
df_new['electron_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==11,row['ak5pfcand_id']) if hasattr(row['ak5pfcand_id'], '__iter__') else False,axis=1)
df_new['muon_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==13,row['ak5pfcand_id']) if hasattr(row['ak5pfcand_id'], '__iter__') else False,axis=1)
df_new['charHad_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==211,row['ak5pfcand_id']) if hasattr(row['ak5pfcand_id'], '__iter__') else False,axis=1)
df_new['neutHad_filter'] = df_new.apply(lambda row : map(lambda x : abs(x)==130,row['ak5pfcand_id']) if hasattr(row['ak5pfcand_id'], '__iter__') else False,axis=1)

df_new['jet_photon_mult'] = df_new.apply(lambda row : sum(row['photon_filter']) if hasattr(row['photon_filter'], '__iter__') else row['photon_filter'],axis=1)
df_new['jet_electron_mult'] = df_new.apply(lambda row : sum(row['electron_filter']) if hasattr(row['electron_filter'], '__iter__') else row['electron_filter'],axis=1)
df_new['jet_muon_mult'] = df_new.apply(lambda row : sum(row['muon_filter']) if hasattr(row['muon_filter'], '__iter__') else row['muon_filter'],axis=1)
df_new['jet_charHad_mult'] = df_new.apply(lambda row : sum(row['charHad_filter']) if hasattr(row['charHad_filter'], '__iter__') else row['charHad_filter'],axis=1)
df_new['jet_neutHad_mult'] = df_new.apply(lambda row : sum(row['neutHad_filter']) if hasattr(row['neutHad_filter'], '__iter__') else row['neutHad_filter'],axis=1)

df_new['jet_photon_frac'] = df_new.apply(lambda row : (sum(np.multiply(row['photon_filter'],row['ak5pfcand_pt'])) if hasattr(row['photon_filter'], '__iter__') and hasattr(row['ak5pfcand_pt'], '__iter__') else -1.) / row['jet_pt'],axis=1)
df_new['jet_electron_frac'] = df_new.apply(lambda row : (sum(np.multiply(row['electron_filter'],row['ak5pfcand_pt'])) if hasattr(row['electron_filter'], '__iter__') and hasattr(row['ak5pfcand_pt'], '__iter__') else -1.) / row['jet_pt'],axis=1)
df_new['jet_muon_frac'] = df_new.apply(lambda row : (sum(np.multiply(row['muon_filter'],row['ak5pfcand_pt'])) if hasattr(row['muon_filter'], '__iter__') and hasattr(row['ak5pfcand_pt'], '__iter__') else -1.) / row['jet_pt'],axis=1)
df_new['jet_charHad_frac'] = df_new.apply(lambda row : (sum(np.multiply(row['charHad_filter'],row['ak5pfcand_pt'])) if hasattr(row['charHad_filter'], '__iter__') and hasattr(row['ak5pfcand_pt'], '__iter__') else -1.) / row['jet_pt'],axis=1)
df_new['jet_neutHad_frac'] = df_new.apply(lambda row : (sum(np.multiply(row['neutHad_filter'],row['ak5pfcand_pt'])) if hasattr(row['neutHad_filter'], '__iter__') and hasattr(row['ak5pfcand_pt'], '__iter__') else -1.) / row['jet_pt'],axis=1)

df_new.to_pickle("new_ak5.pkl")

