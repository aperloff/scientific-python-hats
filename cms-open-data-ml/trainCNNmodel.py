from MLJEC_MCTruth_Model import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES']='2'

model = loadModel('model_eta_dense_pt_dense_updatedJuly14')

model.summary()

df = pd.read_pickle('new.pkl')

scaler = StandardScaler()
df['jet_eta_ak7_scaled'] = df['jet_eta_ak7'] / 2.5
df['jet_pt_ak7_exp'] = map(np.log,df['jet_pt_ak7'])
df['jet_pt_ak7_scaled'] = scaler.fit_transform(df['jet_pt_ak7_exp'].reshape(-1, 1))

df = df.sample(frac=1)
df_train,df_test = np.array_split(df,2)

jet_image_train = np.array(map(lambda x : x[0] , df_train['jet_image']))
inputs_train = [jet_image_train.reshape([-1,30,30,1]),np.array(df_train['jet_pt_ak7_scaled']),np.array(df_train['jet_eta_ak7_scaled'])]
jet_image_test = np.array(map(lambda x : x[0] , df_test['jet_image']))
inputs_test = [jet_image_test.reshape([-1,30,30,1]),np.array(df_test['jet_pt_ak7_scaled']),np.array(df_test['jet_eta_ak7_scaled'])]

#import keras.optimizers
#sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','msle'])

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(inputs_train, np.array(df_train['jet_jes_ak7']), validation_data=(inputs_test, np.array(df_test['jet_jes_ak7'])), 
          nb_epoch=400, batch_size=256, verbose=1)#, callbacks=[early_stopping])

val_loss = np.asarray(history.history['val_loss'])
loss = np.asarray(history.history['loss'])
plt.plot(np.log(val_loss), label='test')
plt.plot(np.log(loss), label='train')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['test','train'])
fig = plt.gcf()
plt.draw()
fig.savefig('training_summary.png',dpi=100)

model_json = model.to_json()
with open("model_eta_dense_pt_dense_updatedSept20.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_eta_dense_pt_dense_updatedSept20.h5")
print("Saved model to disk")

df_train['pred'] = model.predict(inputs_train)
df_train['residual'] = df_train['pred']-df_train['jet_jes_ak7']
df_test['pred'] = model.predict(inputs_test)
df_test['residual'] = df_test['pred']-df_test['jet_jes_ak7']

df = pd.concat([df_train,df_test])
df.to_pickle("new_withCNN.pkl")

