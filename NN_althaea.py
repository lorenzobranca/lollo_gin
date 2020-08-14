import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.backend import tensorflow_backend as K
import keras
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization

dataset=np.loadtxt('dataset_althaea_flux_rho_useX.dat')
print(np.shape(dataset))



X=dataset[:,0:3]


y=np.log10((dataset[:,4:-1]))
'''
for i in range(8):
	plt.plot(X[200:400,0],y[200:400,i],'o')
	plt.show()

exit()
'''
del(dataset)
X_train_full, X_test, y_train_full, y_test = train_test_split(
X,y)
X_train, X_valid, y_train, y_valid = train_test_split(
X_train_full, y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)



early_stopping_cb = keras.callbacks.EarlyStopping(patience=25,
restore_best_weights=True)

epochs=100000


model=Sequential()
model.add(Dense(200,input_shape=(3,)))
#model.add( BatchNormalization())
#model.add(Dropout(0.2))
model.add(LeakyReLU())
model.add(Dense(150))
#model.add(Dropout(0.2))
model.add(LeakyReLU())
model.add(Dense(100))
#model.add(Dropout(0.2))
model.add(LeakyReLU())
model.add(Dense(70))
#model.add(Dropout(0.2))
model.add(LeakyReLU())
model.add(Dense(8))
model.summary()
opt=keras.optimizers.Adam(learning_rate=0.001)

lrate = ReduceLROnPlateau( monitor='loss', factor=0.3, patience=10, min_lr=2.e-5 )


model.compile(optimizer=opt, loss='mean_squared_error')

history = model.fit( X_train_scaled, y_train,
                     epochs=epochs,
                     batch_size=16,
                     callbacks=[lrate,early_stopping_cb],
                     validation_data=( X_valid_scaled, y_valid ),
                     verbose=True )

mse=model.evaluate(X_test_scaled,y_test)
print(mse)





'''
plt.plot(y_pred[:,specie],y_test[:,specie],'o')
plt.plot([min(y_pred[:,specie]),max(y_pred[:,specie])],[min(y_pred[:,specie]),max(y_pred[:,specie])])
plt.show()
'''
y_pred_train=model.predict(X_train_scaled)
y_pred_test=model.predict(X_test_scaled)
'''
plt.plot(y_pred_train[:,specie],y_train[:,specie],'o')
plt.plot([min(y_pred_train[:,specie]),max(y_pred_train[:,specie])],[min(y_pred_train[:,specie]),max(y_pred_train[:,specie])])
plt.show()

for i in range(7):
    plt.plot(y_pred_train[:,i],y_train_scaled[:,i],'o')
    plt.plot(y_pred_test[:,i],y_test_scaled[:,i],'^')
    plt.plot([min(y_train_scaled[:,i]),max(y_train_scaled[:,i])],[min(y_train_scaled[:,i]),max(y_train_scaled[:,i])])
    
    plt.show()
'''
distance_train=abs((y_train-y_pred_train)/(y_train))
distance_test=abs((y_test-y_pred_test)/(y_test))
distance_mean=np.array([np.mean(distance_train),np.mean(distance_test)])
#np.savetxt('distance_althaea_logscale.txt',distance_mean)

name=["E", "H-", "H", "HE", "H2", "D", "HD", "H+", "HE+", "H2+", "D+"]
for i in range(8):
        print(np.mean(distance_train[:,i]),np.mean(distance_test[:,i]))
        plt.plot([min(y_train[:,i]),max(y_train[:,i])],[min(y_train[:,i]),max(y_train[:,i])])
        plt.plot(y_train[:,i],y_pred_train[:,i],'o')
        plt.plot(y_test[:,i],y_pred_test[:,i],'o')
	#plt.savefig('image_map/name[i].png')
	plt.show()
        

#model.save('keras_althaea_flux_rho_useX_logscale.h5')
