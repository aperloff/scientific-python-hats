from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import numpy as np
import pandas as pd
import glob

####################
# Global Variables #
####################
nx = 30 # size of image in eta
ny = 30 # size of image in phi
xbins = np.linspace(-1.4,1.4,nx+1)
ybins = np.linspace(-1.4,1.4,ny+1)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# rotation + (possible) reflection needed later
def rotate_and_reflect(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    maxPt = -1
    for ix, iy, iw in zip(x, y, w):
        dv = np.matrix([[ix],[iy]])-np.matrix([[x[0]],[y[0]]])
        dR = np.linalg.norm(dv)
        thisPt = iw
        if dR > 0.25 and thisPt > maxPt:
            maxPt = thisPt
            # rotation in eta-phi plane c.f  https://arxiv.org/abs/1407.5675 and https://arxiv.org/abs/1511.05190:
            # theta = -np.arctan2(iy,ix)-np.radians(90)
            # rotation by lorentz transformation c.f. https://arxiv.org/abs/1704.02124:
            px = iw * np.cos(iy)
            py = iw * np.sin(iy)
            pz = iw * np.sinh(ix)
            theta = np.arctan2(py,pz)+np.radians(90)
            #print "px,py,pz,theta:",px,py,pz,theta

    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    for ix, iy, iw in zip(x, y, w):
        # rotation in eta-phi plane:
        #rot = R*np.matrix([[ix],[iy]])
        #rix, riy = rot[0,0], rot[1,0]
        # rotation by lorentz transformation
        px = iw * np.cos(iy)
        py = iw * np.sin(iy)
        pz = iw * np.sinh(ix)
        rot = R*np.matrix([[py],[pz]])
        rix, riy = np.arcsinh(rot[1,0]/iw), np.arcsin(rot[0,0]/iw)
        rot_x.append(rix)
        rot_y.append(riy)

    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0:
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y
    else:
        ref_x = rot_x
        ref_y = rot_y

    return (np.array(ref_x), np.array(ref_y))

def rotate_all(x_vec,y_vec,w_vec):
    x_res=[]
    y_res=[]
    i=0
    mat = zip(x_vec,y_vec,w_vec)
    print "total:",len(mat)
    for x,y,w in mat:
        i+=1
        if i%1000==0 : 
            print "i:",i
        x_temp,y_temp=rotate_and_reflect(x,y,w)
        x_res.append(x_temp)
        y_res.append(y_temp)
        if i>10000 : 
            break

    return x_res,y_res

def prepare_df_dict(params, verbose):
    # now let's prepare some jet images
    if verbose:
        print (params['QCD120'].dtype.names)

    df_dict_jet = {}
    df_dict_cand = {}
    '''
    df_dict_jet['TT'] = pd.DataFrame(params['TT'],columns=['run', 'lumi', 'event', 'met', 'sumet', 'rho', 'pthat', 'mcweight', 'njet_ak7', 'jet_pt_ak7', 'jet_eta_ak7', 'jet_phi_ak7', 'jet_E_ak7', 'jet_msd_ak7', 'jet_area_ak7', 'jet_jes_ak7', 'jet_tau21_ak7', 'jet_isW_ak7', 'jet_ncand_ak7','ak7pfcand_ijet'])
    df_dict_jet['TT'] = df_dict_jet['TT'].drop_duplicates()
    df_dict_jet['TT'] =  df_dict_jet['TT'][(df_dict_jet['TT'].jet_pt_ak7 > 200) & (df_dict_jet['TT'].jet_pt_ak7 < 500) &  (df_dict_jet['TT'].jet_isW_ak7==1)]
    
    df_dict_cand['TT'] = pd.DataFrame(params['TT'],columns=['event', 'jet_pt_ak7', 'jet_isW_ak7', 'ak7pfcand_pt', 'ak7pfcand_eta', 'ak7pfcand_phi', 'ak7pfcand_id', 'ak7pfcand_charge', 'ak7pfcand_ijet'])
    df_dict_cand['TT'] =  df_dict_cand['TT'][(df_dict_cand['TT'].jet_pt_ak7 > 200) & (df_dict_cand['TT'].jet_pt_ak7 < 500) &  (df_dict_cand['TT'].jet_isW_ak7==1)]
    '''


    for QCDbin in ['QCD120','QCD170','QCD300','QCD470']:
        df_dict_jet[QCDbin] = pd.DataFrame(params[QCDbin],columns=['run', 'lumi', 'event', 'met', 'sumet', 'rho', 'pthat', 'mcweight', 'njet_ak7', 'jet_pt_ak7', 'jet_eta_ak7', 'jet_phi_ak7', 'jet_E_ak7', 'jet_msd_ak7', 'jet_area_ak7', 'jet_jes_ak7', 'jet_tau21_ak7', 'jet_isW_ak7', 'jet_ncand_ak7','ak7pfcand_ijet'])
        df_dict_jet[QCDbin] = df_dict_jet[QCDbin].drop_duplicates()
        df_dict_jet[QCDbin] =  df_dict_jet[QCDbin][(df_dict_jet[QCDbin].jet_pt_ak7 > 100) & (df_dict_jet[QCDbin].jet_pt_ak7 < 600) &  (df_dict_jet[QCDbin].jet_isW_ak7==0)]
        # take every 20th jet just to make the training faster and have a sample roughly the size of W jets
        #df_dict_jet[QCDbin] = df_dict_jet[QCDbin].iloc[::20, :]
        
        df_dict_cand[QCDbin] = pd.DataFrame(params[QCDbin],columns=['event', 'jet_pt_ak7', 'jet_isW_ak7', 'ak7pfcand_pt', 'ak7pfcand_eta', 'ak7pfcand_phi', 'ak7pfcand_id', 'ak7pfcand_charge', 'ak7pfcand_ijet'])
        df_dict_cand[QCDbin] =  df_dict_cand[QCDbin][(df_dict_cand[QCDbin].jet_pt_ak7 > 100) & (df_dict_cand[QCDbin].jet_pt_ak7 < 600) &  (df_dict_cand[QCDbin].jet_isW_ak7==0)]
    
    df_dict_jet['QCD'] = pd.concat([df_dict_jet['QCD120'],df_dict_jet['QCD170'],df_dict_jet['QCD300'],df_dict_jet['QCD470']])
    df_dict_cand['QCD'] = pd.concat([df_dict_cand['QCD120'],df_dict_cand['QCD170'],df_dict_cand['QCD300'],df_dict_cand['QCD470']])
    if verbose:
        print ("Length of QCD jets = ")
        print (len(df_dict_jet['QCD']))
    return df_dict_jet, df_dict_cand

class JetImageGenerator(object):
    def __init__(self,batch_size=32):
        # structure of dataframe
        self.jet_columns = ['run', 'lumi', 'event', 'met', 'sumet', 'rho', 'pthat', 'mcweight',
                            'njet_ak7', 'jet_pt_ak7', 'jet_eta_ak7', 'jet_phi_ak7', 'jet_E_ak7',
                            'jet_msd_ak7', 'jet_area_ak7', 'jet_jes_ak7', 'jet_tau21_ak7',
                            'jet_isW_ak7','jet_ncand_ak7','ak7pfcand_ijet']
        #self.NDIM = self.jet_columns.index('jet_isW_ak7')
        self.NDIM = self.jet_columns.index('jet_jes_ak7')

        self.cand_columns = ['event', 'jet_pt_ak7', 'jet_isW_ak7', 'ak7pfcand_pt', 'ak7pfcand_eta',
                             'ak7pfcand_phi', 'ak7pfcand_id', 'ak7pfcand_charge', 'ak7pfcand_ijet']

        self.batch_size = batch_size

        self.categories = ['QCD']
        self.file_pattern = {
            #'TT': 'output_TT/*.npy',
            'QCD': 'output_QCD*/*.npy'
        }
        
        self.preselection = {
            #'TT': lambda df: df[(df.jet_pt_ak7 > 200) & (df.jet_pt_ak7 < 500) &  (df.jet_isW_ak7==1)],
            'QCD': lambda df: df[(df.jet_pt_ak7 > 100) & (df.jet_pt_ak7 < 600) &  (df.jet_isW_ak7==0)],
        }
                
        # list of files
        self.inputs = {}
        for cat in self.categories:
            self.inputs[cat] = glob.glob(self.file_pattern[cat])
        
        
        
    def _load_category(self,category,i):
        if i<len(self.inputs[category]):
            try:
                fname = self.inputs[category][i]
                #print category, i, fname
                params = np.load(fname)
                jet_df = pd.DataFrame(params,columns=self.jet_columns)
                cand_df = pd.DataFrame(params,columns=self.cand_columns)
                jet_df.drop_duplicates(inplace=True)
                jet_df = self.preselection[category](jet_df)
                cand_df = self.preselection[category](cand_df)
                return jet_df, cand_df
            except:
                print ('bad file: %s'%fname)
                if fname in self.inputs[category]: self.inputs[category].remove(fname)
        return pd.DataFrame(), pd.DataFrame()
    
    def generator(self,test=False,crossvalidation=False):

        percat = self.batch_size/len(self.categories)

        icat = {cat:0 for cat in self.categories}
        cat_X = {cat:np.array([]) for cat in self.categories}
        cat_y = {cat:np.array([]) for cat in self.categories}
        cat_z = {cat:np.array([]) for cat in self.categories}
        cat_t = {cat:np.array([]) for cat in self.categories}
        cat_u = {cat:np.array([]) for cat in self.categories}
        
        kfold = StratifiedKFold(n_splits=2, shuffle=True,  random_state=seed)
        
        while True:
            # stop iteration
            #for cat in self.categories:
            #    if len(cat_y[cat])<percat and icat[cat]>=len(self.inputs[cat]):
            #        raise StopIteration
            
            # load data
            for cat in self.categories:
                while len(cat_y[cat])<percat:
                    # load from file
                    jet_df, cand_df = self._load_category(cat,icat[cat]%len(self.inputs[cat])) # allow infinite looping
                    # get the image from pf cands
                    if K.image_dim_ordering()=='tf':
                        jet_images = np.zeros((len(jet_df), nx, ny, 1))
                    else:        
                        jet_images = np.zeros((len(jet_df), 1, nx, ny))
                    njets = 0
                    for i in range(0,len(jet_df)):
                        njets+=1
                        # get the ith jet
                        df_cand_i = cand_df[(cand_df['ak7pfcand_ijet'] == jet_df['ak7pfcand_ijet'].iloc[i]) & (cand_df['event'] == jet_df['event'].iloc[i])]
                        # relative eta
                        x = df_cand_i['ak7pfcand_eta']-df_cand_i['ak7pfcand_eta'].iloc[0]
                        # relative phi
                        y = df_cand_i['ak7pfcand_phi']-df_cand_i['ak7pfcand_phi'].iloc[0]
                        weights = df_cand_i['ak7pfcand_pt'] # pt of candidate is the weight
                        x,y = rotate_and_reflect(x,y,weights)
                        hist, xedges, yedges = np.histogram2d(x, y,weights=weights, bins=(xbins,ybins))
                        for ix in range(0,nx):
                            for iy in range(0,ny):
                                if K.image_dim_ordering()=='tf':
                                    jet_images[i,ix,iy,0] = hist[ix,iy]
                                else:
                                    jet_images[i,0,ix,iy] = hist[ix,iy]
                    # split them to test and train
                    X = jet_images
                    y = jet_df.values[:,self.NDIM]
                    z = jet_df.values[:,self.jet_columns.index('jet_pt_ak7')]
                    t = jet_df.values[:,self.jet_columns.index('jet_eta_ak7')]
                    u = jet_df.values[:,self.jet_columns.index('jet_phi_ak7')]
                    #encoder = LabelEncoder()
                    #encoder.fit(y)
                    #encoded_y = encoder.transform(y)
                    #data_train, data_test = list(kfold.split(X, encoded_y))[int(crossvalidation)]
                    mixed = list(zip(X,y,z,t,u))
                    np.random.shuffle(mixed) 
                    data_train = mixed[:int(len(mixed)*0.4)]
                    data_test = mixed[int(len(mixed)*0.4):]
                    # select test or train
                    sample = data_test if test else data_train
                    X = np.array([C[0] for C in sample])
                    y = np.array([C[1] for C in sample])
                    z = np.array([C[2] for C in sample])
                    t = np.array([C[3] for C in sample])
                    u = np.array([C[4] for C in sample])
                    cat_X[cat] = np.vstack((cat_X[cat],X)) if cat_X[cat].size else X
                    cat_y[cat] = np.hstack((cat_y[cat],y)) if cat_y[cat].size else y
                    cat_z[cat] = np.hstack((cat_z[cat],z)) if cat_z[cat].size else z
                    cat_t[cat] = np.hstack((cat_t[cat],t)) if cat_t[cat].size else t
                    cat_u[cat] = np.hstack((cat_u[cat],u)) if cat_u[cat].size else u
                    icat[cat] += 1

            # build combined sample based on batch_size
            all_X = np.array([])
            all_y = np.array([])
            all_z = np.array([])
            all_t = np.array([])
            all_u = np.array([])
            for cat in self.categories:
                X = cat_X[cat][:percat]
                y = cat_y[cat][:percat]
                z = cat_z[cat][:percat]
                t = cat_t[cat][:percat]
                u = cat_u[cat][:percat]
                cat_X[cat] = cat_X[cat][percat:]
                cat_y[cat] = cat_y[cat][percat:]
                cat_z[cat] = cat_z[cat][percat:]
                cat_t[cat] = cat_t[cat][percat:]
                cat_u[cat] = cat_u[cat][percat:]
                all_X = np.vstack((all_X,X)) if all_X.size else X
                all_y = np.hstack((all_y,y)) if all_y.size else y
                all_z = np.hstack((all_z,z)) if all_z.size else z
                all_t = np.hstack((all_t,t)) if all_t.size else t
                all_u = np.hstack((all_u,u)) if all_u.size else u

            yield [all_X, all_z, all_t, all_u], all_y
            #yield all_X, all_y

    def prepare_jet_images_for_training(self,jet_df,cand_df,test=False):
        icat = {cat:0 for cat in self.categories}
        cat_X = {cat:np.array([]) for cat in self.categories}
        cat_y = {cat:np.array([]) for cat in self.categories}
        cat_z = {cat:np.array([]) for cat in self.categories}
        cat_t = {cat:np.array([]) for cat in self.categories}
        cat_u = {cat:np.array([]) for cat in self.categories}

        kfold = StratifiedKFold(n_splits=2, shuffle=True,  random_state=seed)
        
        for cat in self.categories:
            if K.image_dim_ordering()=='tf':
                jet_images = np.zeros((len(jet_df), nx, ny, 1))
            else:
                jet_images = np.zeros((len(jet_df), 1, nx, ny))
            njets = 0
            for i in range(0,len(jet_df)):
                njets+=1
                df_cand_i = cand_df[cat][(cand_df[cat]['ak7pfcand_ijet'] == jet_df[cat]['ak7pfcand_ijet'].iloc[i]) & (cand_df[cat]['event'] == jet_df[cat]['event'].iloc[i])]
                # relative eta
                x = df_cand_i['ak7pfcand_eta']-df_cand_i['ak7pfcand_eta'].iloc[0]
                # relative phi
                y = df_cand_i['ak7pfcand_phi']-df_cand_i['ak7pfcand_phi'].iloc[0]
                weights = df_cand_i['ak7pfcand_pt'] # pt of candidate is the weight
                x,y = rotate_and_reflect(x,y,weights)
                hist, xedges, yedges = np.histogram2d(x, y,weights=weights, bins=(xbins,ybins))

                for ix in range(0,nx):
                    for iy in range(0,ny):
                        if K.image_dim_ordering()=='tf':
                            jet_images[i,ix,iy,0] = hist[ix,iy]
                        else:
                            jet_images[i,0,ix,iy] = hist[ix,iy]

            # split them to test and train
            X = jet_images
            y = jet_df[cat].values[:,self.jet_columns.index('jet_jes_ak7')]
            z = jet_df[cat].values[:,self.jet_columns.index('jet_pt_ak7')]
            t = jet_df[cat].values[:,self.jet_columns.index('jet_eta_ak7')]
            u = jet_df[cat].values[:,self.jet_columns.index('jet_phi_ak7')]
            mixed = list(zip(X,y,z,t,u))
            np.random.shuffle(mixed)
            data_train = mixed[:int(len(mixed)*0.4)]
            data_test = mixed[int(len(mixed)*0.4):]
            # select test or train
            sample = data_test if test else data_train
            X = np.array([C[0] for C in sample])
            y = np.array([C[1] for C in sample])
            z = np.array([C[2] for C in sample])
            t = np.array([C[3] for C in sample])
            u = np.array([C[4] for C in sample])
            cat_X[cat] = np.vstack((cat_X[cat],X)) if cat_X[cat].size else X
            cat_y[cat] = np.hstack((cat_y[cat],y)) if cat_y[cat].size else y
            cat_z[cat] = np.hstack((cat_z[cat],z)) if cat_z[cat].size else z
            cat_t[cat] = np.hstack((cat_t[cat],t)) if cat_t[cat].size else t
            cat_u[cat] = np.hstack((cat_u[cat],u)) if cat_u[cat].size else u
            icat[cat] += 1
            
        # build combined sample based on batch_size
        all_X = np.array([])
        all_y = np.array([])
        all_z = np.array([])
        all_t = np.array([])
        all_u = np.array([])
        for cat in self.categories:
            X = cat_X[cat][:percat]
            y = cat_y[cat][:percat]
            z = cat_z[cat][:percat]
            t = cat_t[cat][:percat]
            u = cat_u[cat][:percat]
            cat_X[cat] = cat_X[cat][percat:]
            cat_y[cat] = cat_y[cat][percat:]
            cat_z[cat] = cat_z[cat][percat:]
            cat_t[cat] = cat_t[cat][percat:]
            cat_u[cat] = cat_u[cat][percat:]
            all_X = np.vstack((all_X,X)) if all_X.size else X
            all_y = np.hstack((all_y,y)) if all_y.size else y
            all_z = np.hstack((all_z,z)) if all_z.size else z
            all_t = np.hstack((all_t,t)) if all_t.size else t
            all_u = np.hstack((all_u,u)) if all_u.size else u

        return [all_X, all_z, all_t, all_u], all_y
