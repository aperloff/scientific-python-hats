import numpy as np
import pandas as pd
<<<<<<< HEAD
import glob
from math import sqrt,tanh
def raw_data():
    '''
    filelist = glob.glob("/inputdata/cms-open-data-ml/output_QCD120/*.npy")
    df = pd.DataFrame()
    list_ = []
    for f in filelist :
        np_data = np.load(f)
        df = pd.DataFrame(np_data)
        list_.append(df)        
    df = pd.concat(list_)
    '''
    np_data={}
    np_data['QCD120'] = np.load("/inputdata/cms-open-data-ml/output_QCD120/params0.npy_job0_file0.npy")
    np_data['QCD170'] = np.load("/inputdata/cms-open-data-ml/output_QCD170/params0.npy_job0_file0.npy")
    np_data['QCD300'] = np.load("/inputdata/cms-open-data-ml/output_QCD300/params0.npy_job0_file0.npy")
    np_data['QCD470'] = np.load("/inputdata/cms-open-data-ml/output_QCD470/params0.npy_job0_file0.npy")
    np_data['QCD120a'] = np.load("/inputdata/cms-open-data-ml/output_QCD120/params0.npy_job1_file0.npy")
    np_data['QCD170a'] = np.load("/inputdata/cms-open-data-ml/output_QCD170/params0.npy_job1_file0.npy")
    np_data['QCD300a'] = np.load("/inputdata/cms-open-data-ml/output_QCD300/params0.npy_job1_file0.npy")
    np_data['QCD470a'] = np.load("/inputdata/cms-open-data-ml/output_QCD470/params0.npy_job1_file0.npy")
    np_data['QCD120b'] = np.load("/inputdata/cms-open-data-ml/output_QCD120/params0.npy_job2_file0.npy")
    np_data['QCD170b'] = np.load("/inputdata/cms-open-data-ml/output_QCD170/params0.npy_job2_file0.npy")
    np_data['QCD300b'] = np.load("/inputdata/cms-open-data-ml/output_QCD300/params0.npy_job2_file0.npy")
    np_data['QCD470b'] = np.load("/inputdata/cms-open-data-ml/output_QCD470/params0.npy_job2_file0.npy")
    pd_data={}
    for sample in np_data :
        pd_data[sample] = pd.DataFrame(np_data[sample])
    df = pd.concat([pd_data['QCD120'],pd_data['QCD170'],pd_data['QCD300'],pd_data['QCD470'],
                   pd_data['QCD120a'],pd_data['QCD170a'],pd_data['QCD300a'],pd_data['QCD470a'],
                   pd_data['QCD120b'],pd_data['QCD170b'],pd_data['QCD300b'],pd_data['QCD470b']])

=======
from math import sqrt,tanh
def raw_data():
    np_data={}
    np_data['QCD120'] = np.load("/Users/awhitbe1/deepJEC/CMSOpenData/output_QCD120/params0.npy_job0_file0.npy")
    np_data['QCD170'] = np.load("/Users/awhitbe1/deepJEC/CMSOpenData/output_QCD170/params0.npy_job0_file0.npy")
    np_data['QCD300'] = np.load("/Users/awhitbe1/deepJEC/CMSOpenData/output_QCD300/params0.npy_job0_file0.npy")
    np_data['QCD470'] = np.load("/Users/awhitbe1/deepJEC/CMSOpenData/output_QCD470/params0.npy_job0_file0.npy")
    pd_data={}
    for sample in np_data :
        pd_data[sample] = pd.DataFrame(np_data[sample])
    df = pd.concat([pd_data['QCD120'],pd_data['QCD170'],pd_data['QCD300'],pd_data['QCD470']])
>>>>>>> 0ad49aa5b6868e8dcac802635fb96ffe1eb73730
    df.reset_index(inplace=True)
        
    df['jetPx'] = map(np.cos,df['jet_phi_ak7'])*df['jet_pt_ak7']
    df['jetPy'] = map(np.sin,df['jet_phi_ak7'])*df['jet_pt_ak7']
    df['ak7pfcand_id'] = map(abs,df['ak7pfcand_id'])

    return df

def jet_level_data():
        
    df = raw_data()

    means = df.groupby(['event', 'ak7pfcand_ijet']).mean()
    means.reset_index(inplace=True)
    removalList = ['run','lumi','ak7pfcand_pt','ak7pfcand_eta','ak7pfcand_phi','jet_ncand_ak7','ak7pfcand_charge','ak7pfcand_id']
    for col in removalList:
        del means[col]
        
    sums = df.groupby(['event', 'ak7pfcand_ijet', 'ak7pfcand_id']).sum()
    sums.reset_index(inplace=True)
    temp = sums.columns
    temp = map(lambda x : "{0}_sum".format(x),temp)
    temp[0] = 'event'
    temp[1] = 'ak7pfcand_ijet'
    temp[2] = 'ak7pfcand_id'
    sums.columns = temp
    
    sums_pivot = sums.pivot_table(index = ['event', 'ak7pfcand_ijet'], columns = 'ak7pfcand_id', values = ['ak7pfcand_pt_sum'])
    sums_pivot.reset_index(inplace=True)
    
    sums_pivot.columns = ['event','ak7pfcand_ijet',
                          'jet_electronFrac_ak7', 'jet_muonFrac_ak7', 'jet_photonFrac_ak7',
                          'jet_neuHadronFrac_ak7', 'jet_charHadronFrac_ak7']
    sums_pivot.fillna(0,inplace=True)
    sums_pivot.head()
    
    counts = df.groupby(['event', 'ak7pfcand_ijet', 'ak7pfcand_id']).count()
    counts.reset_index(inplace=True)
    saveList = ['ak7pfcand_charge','event','ak7pfcand_ijet','ak7pfcand_id']
    for i in counts.columns:
        if i in saveList : continue
        del counts[i]
    counts.columns = ['event','ak7pfcand_ijet','ak7pfcand_id','ak7pfcand_mult']
    counts_pivot = counts.pivot_table(index = ['event', 'ak7pfcand_ijet'], columns = 'ak7pfcand_id', values = ['ak7pfcand_mult'])
    counts_pivot.reset_index(inplace=True)
    
    counts_pivot.columns = ['event','ak7pfcand_ijet',
                            'jet_electronMult_ak7', 'jet_muonMult_ak7', 'jet_photonMult_ak7',
                            'jet_neuHadronMult_ak7', 'jet_charHadronMult_ak7']
    counts_pivot.fillna(0,inplace=True)

    df_perjet = sums_pivot.set_index(['event','ak7pfcand_ijet']).join(means.set_index(['event','ak7pfcand_ijet']).join(counts_pivot.set_index(['event','ak7pfcand_ijet'])))
    df_perjet.reset_index(inplace=True)
    df_perjet.head()
    
    df_perjet['jet_electronFrac_ak7'] = df_perjet['jet_electronFrac_ak7']/df_perjet['jet_pt_ak7']
    df_perjet['jet_muonFrac_ak7'] = df_perjet['jet_muonFrac_ak7']/df_perjet['jet_pt_ak7']
    df_perjet['jet_photonFrac_ak7'] = df_perjet['jet_photonFrac_ak7']/df_perjet['jet_pt_ak7']
    df_perjet['jet_neuHadronFrac_ak7'] = df_perjet['jet_neuHadronFrac_ak7']/df_perjet['jet_pt_ak7']
    df_perjet['jet_charHadronFrac_ak7'] = df_perjet['jet_charHadronFrac_ak7']/df_perjet['jet_pt_ak7']

    return df_perjet

def event_level_data():

    df_perjet = jet_level_data()

    df_perevt = df_perjet.groupby(['event']).sum()
    df_perevt['MHT'] = map(np.sqrt,df_perevt['jetPy']*df_perevt['jetPy']+df_perevt['jetPx']*df_perevt['jetPx'])
    df_perevt.reset_index(inplace=True)

    return df_perevt
