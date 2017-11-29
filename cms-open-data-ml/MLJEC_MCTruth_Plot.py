from sklearn.externals import joblib
from sklearn.preprocessing import scale
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from MLJEC_MCTruth_Util import rotate_and_reflect, prepare_df_dict, JetImageGenerator
from itertools import cycle
from ipywidgets import FloatProgress
from IPython.display import display
from sklearn.metrics import roc_curve, auc

lw=2

def plotJet(df_dict_jet, df_dict_cand, process='TT',njets_to_plot=-1, nx=30, ny=30, xbins=[], ybins=[]):
    jet_images = {}
    # 4D tensor
    # 1st dim is jet index
    # 2nd dim is pt value (or rgb, etc.)
    # 3rd dim is eta bin
    # 4th dim is phi bin

    list_x = []
    list_y = []
    list_w = []
    njets = 0

    if K.image_dim_ordering()=='tf':
        jet_images[process] = np.zeros((len(df_dict_jet[process]), nx, ny, 1))
    else:        
        jet_images[process] = np.zeros((len(df_dict_jet[process]), 1, nx, ny))
    
    njets_ = []
    if njets_to_plot == -1:
        njets_ = range(0,len(df_dict_jet[process]))
    else:
        njets_ = range(0,njets_to_plot)
    
    for i in njets_:
        njets+=1
        # get the ith jet
        df_dict_cand_i = {}
        df_dict_cand_i[process] = df_dict_cand[process][(df_dict_cand[process]['ak7pfcand_ijet'] == df_dict_jet[process]['ak7pfcand_ijet'].iloc[i]) & (df_dict_cand[process]['event'] == df_dict_jet[process]['event'].iloc[i]) ]
        # relative eta
        x = df_dict_cand_i[process]['ak7pfcand_eta']-df_dict_cand_i[process]['ak7pfcand_eta'].iloc[0]
        # relative phi
        y = df_dict_cand_i[process]['ak7pfcand_phi']-df_dict_cand_i[process]['ak7pfcand_phi'].iloc[0]
        weights = df_dict_cand_i[process]['ak7pfcand_pt'] # pt of candidate is the weight
        x,y = rotate_and_reflect(x,y,weights)
        list_x.append(x)
        list_y.append(y)
        list_w.append(weights)
        hist, xedges, yedges = np.histogram2d(x, y,weights=weights, bins=(xbins,ybins))
        for ix in range(0,nx):
            for iy in range(0,ny):
                if K.image_dim_ordering()=='tf':
                    jet_images[process][i,ix,iy,0] = hist[ix,iy]
                else:
                    jet_images[process][i,0,ix,iy] = hist[ix,iy]
    all_x = np.concatenate(list_x)
    all_y = np.concatenate(list_y)
    all_w = np.concatenate(list_w)
    all_w = 1.*all_w/njets # to get average
    plt.figure('W') 
    plt.hist2d(all_x, all_y, weights=all_w, bins=(xbins,ybins), norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.show()
<<<<<<< HEAD

def plot_JES(conv_model, verbose):
    colors = cycle(['seagreen','cyan', 'indigo', 'yellow', 'blue', 'darkorange', 'red', 'black', 'green', 'brown'])
    for cv, color in zip(range(0,1), colors):
        nbatches = 200
        jetImageGenerator2 = JetImageGenerator(2)
        gen = jetImageGenerator2.generator(test=True)
        y_predict = []
        y_score = []

        #Progress bar
        #maxval = 100
        maxval = nbatches
        #f = FloatProgress(min=0, max=maxval)
        #display(f)

        all_pT = np.array([])
        all_pT_noscale = np.array([])
        all_eta = np.array([])
        all_eta_noscale = np.array([])
        for i in tqdm(range(nbatches)):
            #f.value += 1
            #if i%10==0:
                #print "Jet",i
            Xp, yp = gen.next()
            Xp_ns, yp_ns = Xp, yp
            #if verbose:
                #print ("Before transformation:")
                #print ("\t",Xp)

            #Do all of the scaling
            np.clip(Xp[0],0,100,out=Xp[0])
            scaler = joblib.load('scaler.pkl')
            Xp[1] = scaler.transform(Xp[1].reshape(-1,1))
            #Xp[1] -= 200
            #Xp[1] /= 100
            Xp[2] /= np.max(np.abs(2.5),axis=0)

            #if verbose:
                #print ("After transformation:")
                #print ("\t",Xp)
                #print ("y_predict",yp)
                #print ("y_score", conv_model.predict(Xp))

            y_predict += [yp]
            y_score += [conv_model.predict(Xp)]
            all_pT = np.concatenate([all_pT.reshape(-1,1), Xp[1]])
            all_eta = np.concatenate([all_eta.reshape(-1,1), Xp[2]])
            #all_pT_noscale = np.concatenate([all_pT_noscale, Xp_ns[1]])
            #all_eta_noscale = np.concatenate([all_eta_noscale, Xp_ns[2]])

        y_predict = np.concatenate(y_predict)
        y_score = np.concatenate(y_score)
        print y_predict
        print y_score
    plt.figure()
    plt.scatter(y_predict, y_score, color=color, label='CNN')
    plt.xlim([0.8, 1.2])
    plt.ylim([0.8, 1.2])
    plt.ylabel('Predicted JEC')
    plt.xlabel('True JEC')
    plt.title('Predicted Vs. True JEC')
    plt.legend(loc="lower right")
    plt.savefig("JES_4.pdf")
    #plt.show()

    '''
    plt.figure()
    plt.xlabel('pT')
    plt.ylabel('Entries')
    plt.title('Jet pT')
    bins = np.linspace(200, 500, 60)
    plt.hist(all_pT_noscale,bins=bins, alpha=1, label='Mixed Sample',normed=True)
    plt.legend(loc='upper right')
    plt.savefig("pT.pdf")

    plt.figure()
    plt.xlabel('eta')
    plt.ylabel('Entries')
    plt.title('Jet eta')
    bins = np.linspace(-3, 3, 20)
    plt.hist(all_eta_noscale,bins=bins, alpha=1, label='Mixed Sample',normed=True)
    plt.legend(loc='upper right')
    plt.savefig("eta.pdf")

    # plot input JES and output JES vs pT and eta scatter plots
    plt.figure()
    plt.xlim([180, 550])
    plt.ylim([0.0, 3.0])
    plt.xlabel(r'Jet $p_{T}$ [GeV/c]')
    plt.ylabel('JEC')
    plt.title('JetMET vs DCNN')
    plt.scatter(y_predict, all_pT_noscale, color=blue, label='JetMET')
    plt.scatter(y_score, all_pT_noscale, color=green, label='DCNN')
    plt.legend(loc="upper right")
    plt.savefig("jec_vs_jetpt.pdf")
    '''
    # plot input JES and output JES vs pT and eta scatter plots
    plt.figure()
    plt.xlim([-1, 1])
    plt.ylim([0.0, 3.0])
    plt.xlabel(r'Scaled Jet $p_{T}$ [GeV/c]')
    plt.ylabel('JEC')
    plt.title('JetMET vs DCNN')
    plt.scatter(all_pT, y_predict, color='blue', label='JetMET')
    plt.scatter(all_pT, y_score, color='green', label='DCNN')
    plt.legend(loc="upper right")
    plt.savefig("jec_vs_jetptscaled.pdf")
    '''
    # plot input JES and output JES vs pT and eta scatter plots
    plt.figure()
    plt.xlim([-3, 3])
    plt.ylim([0.0, 3.0])
    plt.xlabel(r'Jet $\eta$ ')
    plt.ylabel('JEC')
    plt.title('JetMET vs DCNN')
    plt.scatter(y_predict, all_eta_noscale, color=blue, label='JetMET')
    plt.scatter(y_score, all_eta_noscale, color=green, label='DCNN')
    plt.legend(loc="upper right")
    plt.savefig("jec_vs_jeteta.pdf")
    '''
    # plot input JES and output JES vs pT and eta scatter plots
    plt.figure()
    plt.xlim([-3, 3])
    plt.ylim([0.0, 3.0])
    plt.xlabel(r'Scaled Jet $\eta$ [GeV/c]')
    plt.ylabel('JEC')
    plt.title('JetMET vs DCNN')
    plt.scatter(all_eta, y_predict, color='blue', label='JetMET')
    plt.scatter(all_eta, y_score, color='green', label='DCNN')
    plt.legend(loc="upper right")
    plt.savefig("jec_vs_jetetascaled.pdf")

=======
>>>>>>> 0ad49aa5b6868e8dcac802635fb96ffe1eb73730
    
def plot_ROC_curves(conv_model):
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'black', 'green', 'brown'])
    #Plot the ROC curves for the training above
    for cv, color in zip(range(0,1), colors):
        nbatches = 100
        jetImageGenerator2 = JetImageGenerator()
        gen = jetImageGenerator2.generator(test=True)
        y_predict = []
        y_score = []

        #Progress bar
        #maxval = 100
        maxval = nbatches
        f = FloatProgress(min=0, max=maxval)
        display(f)

        for i in range(nbatches):
            f.value += 1
            #if i%10==0:
            #    print "Jet",i
            Xp, yp = gen.next()
            y_predict += [yp]
            y_score += [conv_model.predict(Xp)]
        y_predict = np.concatenate(y_predict)
        y_score = np.concatenate(y_score)
        print y_predict
        print y_score
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_predict, y_score)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    mean_tpr /= kfold.get_n_splits(X, encoded_Y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    #x *= 255
    if K.image_dim_ordering()=='tf':
        x = x.transpose((1, 2, 0))
        x = x.transpose((1, 0, 2))
    #x = np.clip(x, 0, 255).astype('uint8')
    return x

def plot_layer(conv_model):
    # dimensions of the generated pictures for each filter.
    img_width = nx
    img_height = ny

    layer_dict = dict([(layer.name, layer) for layer in conv_model.layers])
    input_img = conv_model.input

    pics = {}
    num_filters = {}
    for key, layer in layer_dict.iteritems():
        if 'convolution2d' in key:
            num_filters[key] = 8
    layer_name = num_filters.keys()[1]
    print layer_name

    kept_filters = []

    # can be any integer from 0 to 7, as there are 8 filters in that layer
    for filter_index in range(0,num_filters[layer_name]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_dim_ordering()=='tf':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
    
        if K.image_dim_ordering()=='tf':
            input_img_data = np.random.random((1, img_width, img_height, 1))
        else:
            input_img_data = np.random.random((1, 1, img_width, img_height))
        
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 100 steps
        for i in range(100):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        # decode the resulting input image
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    
    plt.figure(figsize=(15,15))
    i = 0
    j = 0
    for img, loss  in kept_filters:
        i+=1
        plt.subplot(3,3,i)
        plt.imshow(img[0])
        plt.colorbar()
    plt.show()

def plot_loss(histories):
    val_loss = np.asarray(histories[0].history['val_loss'])
    loss = np.asarray(histories[0].history['loss'])
    plt.plot(np.log(val_loss), label='validation')
    plt.plot(np.log(loss), label='train')
    plt.legend()
    plt.xlabel('epoch')
    plt.savefig("history.pdf")





  

