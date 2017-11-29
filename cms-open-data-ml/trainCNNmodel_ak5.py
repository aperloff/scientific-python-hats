from MLJEC_MCTruth_Model import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.environ['KERAS_BACKEND'] = 'tensorflow'
model = loadModel('model_eta_dense_pt_dense_updatedJuly14')
model.summary()

df = pd.read_pickle('new_ak5.pkl')

scaler = StandardScaler()
df['jet_eta_scaled'] = df['jet_eta'] / 2.5
df['jet_pt_exp'] = map(np.log,df['jet_pt'])
df['jet_pt_scaled'] = scaler.fit_transform(df['jet_pt_exp'].reshape(-1, 1))
df.head()

df = df.sample(frac=1)
df_train,df_test = np.array_split(df,2)
#print df_train.head()
print df_test.head()

jet_image_train = np.array(map(lambda x : x[0] , df_train['jet_image']))
inputs_train = [jet_image_train.reshape([-1,30,30,1]),np.array(df_train['jet_pt_scaled']),np.array(df_train['jet_eta_scaled'])]

jet_image_test = np.array(map(lambda x : x[0] , df_test['jet_image']))
inputs_test = [jet_image_test.reshape([-1,30,30,1]),np.array(df_test['jet_pt_scaled']),np.array(df_test['jet_eta_scaled'])]

model.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','msle'])

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

history=model.fit(inputs_train, np.array(df_train['jet_jes']), validation_data=(inputs_test, np.array(df_test['jet_jes'])), 
                     nb_epoch=1, batch_size=1024, verbose=1, callbacks=[early_stopping])

val_loss = np.asarray(history.history['val_loss'])
loss = np.asarray(history.history['loss'])
plt.plot(np.log(val_loss), label='test')
plt.plot(np.log(loss), label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
fig = plt.gcf()
plt.draw()
fig.savefig('ak5_training_summary.png',dpi=100)

model_json = model.to_json()
with open("model_ak5_eta_dense_pt_dense_updatedSept19.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_ak5_eta_dense_pt_dense_updatedSept19.h5")
print("Saved model to disk")

