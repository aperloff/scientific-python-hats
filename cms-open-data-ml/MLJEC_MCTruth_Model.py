# coding: utf-8
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.layers import Merge, merge
from keras import backend as K
import numpy as np
import pandas as pd
import sys, glob, argparse, os
from itertools import cycle
from scipy import interp
from ipywidgets import FloatProgress
from IPython.display import display
from MLJEC_MCTruth_Util import rotate_and_reflect, prepare_df_dict, JetImageGenerator
import MLJEC_MCTruth_Plot as plotter
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def getInputs():
    # get input numpy arrays
    inputs = {}
    inputs['TT'] = glob.glob('output_TT/*job5*.npy')
    inputs['QCD120'] = glob.glob('output_QCD120/*job0*.npy')
    inputs['QCD170'] = glob.glob('output_QCD170/*job0*.npy')
    inputs['QCD300'] = glob.glob('output_QCD300/*job0*.npy')
    inputs['QCD470'] = glob.glob('output_QCD470/*job0*.npy')
    return inputs

def openFiles(inputs):
    list_params = {}
    params = {}
    for key, input_files in inputs.iteritems():
        list_params[key] = []
        for in_file in input_files:
            try:
                arr = np.load(in_file)
                list_params[key].append(arr)
            except ValueError:
                print 'bad file: %s'%in_file
        params[key] = np.concatenate(list_params[key])
    return params

def convertToPandas(params,verbose):
    # convert to pandas dataframe
    df_dict = {}
    df_dict['TT'] = pd.DataFrame(params['TT'],columns=['jet_pt_ak7','jet_tau21_ak7','jet_msd_ak7','jet_ncand_ak7','jet_isW_ak7', 'pthat','mcweight'])
    for QCDbin in ['QCD120','QCD170','QCD300','QCD470']:
        df_dict[QCDbin] = pd.DataFrame(params[QCDbin],columns=['jet_pt_ak7','jet_tau21_ak7','jet_msd_ak7','jet_ncand_ak7','jet_isW_ak7', 'pthat','mcweight'])

    df_dict['TT'] = df_dict['TT'].drop_duplicates()
    df_dict['TT'] =  df_dict['TT'][(df_dict['TT'].jet_pt_ak7 > 200) & (df_dict['TT'].jet_pt_ak7 < 500) &  (df_dict['TT'].jet_isW_ak7==1)]

    for QCDbin in ['QCD120','QCD170','QCD300','QCD470']:
        df_dict[QCDbin] = df_dict[QCDbin].drop_duplicates()
        df_dict[QCDbin] =  df_dict[QCDbin][(df_dict[QCDbin].jet_pt_ak7 > 200) & (df_dict[QCDbin].jet_pt_ak7 < 500) & (df_dict[QCDbin].jet_isW_ak7==0)]
        # take every 20th jet just to make the training faster and have a sample roughly the size of W jets
        df_dict[QCDbin] = df_dict[QCDbin].iloc[::20, :]
    
    df_dict['QCD'] = pd.concat([df_dict['QCD120'],df_dict['QCD170'],df_dict['QCD300'],df_dict['QCD470']])
    df = pd.concat([df_dict['TT'],df_dict['QCD']])

    if verbose:
        print params['TT'].dtype.names
        print 'number of W jets: %i'%len(df_dict['TT'])
        for QCDbin in ['QCD120','QCD170','QCD300','QCD470']:
            print 'number of QCD jets in bin %s: %i'%( QCDbin, len(df_dict[QCDbin]))
        print df_dict['TT'].iloc[:3]
        print df_dict['QCD'].iloc[:3]

    return df

# Model
def build_conv_model(nx=30, ny=30):
    """Test model.  Consists of several convolutional layers followed by dense layers and an output layer"""
    if K.image_dim_ordering()=='tf':
        input_layer = Input(shape=(nx, ny, 1))
    else:
        input_layer = Input(shape=(1, nx, ny))
    layer = Convolution2D(20, 7, 7, border_mode='same')(input_layer)
    layer = Activation('tanh')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Convolution2D(8, 3, 3, border_mode='same')(layer)
    layer = Activation('tanh')(layer)
    layer = MaxPooling2D(pool_size=(3,3))(layer)
    layer = Convolution2D(8, 3, 3, border_mode='same')(layer)
    layer = Activation('tanh')(layer)
    layer = MaxPooling2D(pool_size=(3,3))(layer)
    layer = Flatten()(layer)
    # additional features input
    jet_pt_ak7_input = Input(shape=(1,), name='jet_pt_ak7_input')
    jet_eta_ak7_input = Input(shape=(1,), name='jet_eta_ak7_input')
    layer = merge([layer, jet_pt_ak7_input, jet_eta_ak7_input], mode='concat')
    #layer = Dropout(0.20)(layer)
    layer = Dense(20)(layer)
    layer = Dropout(0.10)(layer)
    #other activation option 'sigmoid' better for bounded problem
    output_layer = Dense(1, activation='relu', name='main_output')(layer)
    model = Model(input=[input_layer,jet_pt_ak7_input,jet_eta_ak7_input], output=output_layer)
    #model = Model(input=input_layer, output=output_layer)
    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['accuracy','precision','mse','msle'])
    return model

def fitModels(df_dict_jet, df_dict_cand,nx,ny,generator,verbosity,debug):
    # Run classifier with cross-validation and plot ROC curves
    #kfold = StratifiedKFold(n_splits=2, shuffle=True,  random_state=seed)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'black', 'green', 'brown'])
    histories = []
    models = []
    for cv, color in zip(range(0,1), colors):
        conv_model = build_conv_model(nx,ny)
        if verbosity:
            conv_model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        jetImageGenerator = JetImageGenerator()
        if generator:
            history = conv_model.fit_generator(jetImageGenerator.generator(crossvalidation=int(cv)), 128, validation_data=jetImageGenerator.generator(test=True), nb_val_samples=128, nb_epoch=25, verbose=verbosity, callbacks=[early_stopping])
        else:
            if not os.path.exists("X_train.npz"):
                gen1 = jetImageGenerator.generator(crossvalidation=int(cv))
                X_train = np.empty(3200,dtype=object)
                X_test = np.empty(3200,dtype=object)
                f = FloatProgress(min=0, max=3200)
                display(f)
                for i in range(3200):
                    f.value += 1
                    X_train[i], encoded_Y_train = gen1.next()
                    X_test[i], encoded_Y_test = gen1.next()
                data_train = np.load("X_train.npz")
                X_train = [data_train[0],data_train[1],data_train[2]]
                encoded_Y_train = data_train[3]
                data_test = np.load("X_test.npz")
                X_test = [data_test[0],data_test[1],data_test[2]]
                encoded_Y_test = data_test[3]
            else:
                np.savez("X_train.npz",*(X_train + [encoded_Y_train]))
                np.savez("X_test.npz",*(X_test + [encoded_Y_test]))

            scaler = StandardScaler()
            X_train[1][:] = scaler.fit_transform(X_train[1])
            X_test[1][:] = scaler.transform(X_test[1])

            history = conv_model.fit(X_train, encoded_Y_train, validation_data=(X_test, encoded_Y_test), nb_epoch=10, batch_size=32, verbose=verbosity, callbacks=[early_stopping])
        histories.append(history)
        models.append(conv_model)
    return models

def saveModel(model,verbose):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def loadModel(verbose):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    if verbose:
        print("Loaded model from disk")
    return [loaded_model]

####################
# Global Variables #
####################
nx = 30 # size of image in eta
ny = 30 # size of image in phi
xbins = np.linspace(-1.4,1.4,nx+1)
ybins = np.linspace(-1.4,1.4,ny+1)
def main(open_models,train_models,save_models,plot,generator,debug,verbose):
    #Get the inputs
    inputs = getInputs()
    params = openFiles(inputs)
    df = convertToPandas(params,verbose)
    df_dict_jet, df_dict_cand = prepare_df_dict(params, verbose)

    #Make, load, and/or save the models
    if open_models:
        models = loadModel(verbose)
    else:
        models = fitModels(df_dict_jet,df_dict_cand,nx,ny,generator,verbose,debug)
    if save_models and len(models)>=1:
        saveModel(models[0],verbose)

    #Make all of the plots
    if plot:
        #plotter.plotJet(df_dict_jet, df_dict_cand,process='TT', njets_to_plot=1, nx=nx, ny=ny, xbins=xbins, ybins=ybins)
        #plotter.plotJet(df_dict_jet, df_dict_cand,process='QCD', njets_to_plot=1, nx=nx, ny=ny, xbins=xbins, ybins=ybins)
        #plotter.plot_ROC_curves(models[0])
        plotter.plot_JES(models[0])

if __name__ == '__main__':
    #program name available through the %(prog)s command
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="""
Open files and train models for maching learning (ML) based JEC.""",
                                     epilog="""
And those are the options available. Deal with it.
""")
    parser.add_argument("-g", "--generator", help="use a generator rather than loading all data into memory", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-o", "--open_models", help="load the models from a file", action="store_true")
    group.add_argument("-t", "--train_models", help="refit the models", action="store_true")
    parser.add_argument("-s", "--save_models", help="save the models and the weights", action="store_true")
    parser.add_argument("-p", "--plot", help="plot the ROC curves after training and testing", action="store_true")
    parser.add_argument("-d","--debug", help="Shows extra information in order to debug this program.",
                        action="store_true")
    parser.add_argument("-v","--verbose", help="print out additional information", action="store_true")
    parser.add_argument('--version', action='version', version='%(prog)s 2.0b')
    args = parser.parse_args()

    if(args.debug):
         print 'Number of arguments:', len(sys.argv), 'arguments.'
         print 'Argument List:', str(sys.argv)
         print "Argument ", args

    main(open_models=args.open_models,train_models=args.train_models,save_models=args.save_models,plot=args.plot,generator=args.generator,debug=args.debug,verbose=args.verbose)



