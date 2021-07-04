# Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read les donnée

train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

# On observe les 5 première occurence

train.head()

# Reduction de l'usage mémoire

def reduction_mem(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[ float_cols].astype(np.float16)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


train = reduction_mem(train)

# Recupération des colone qui on un nom

day_columns = [col for col in train.columns if 'd_' in col]

# Récupération des donnée de vente seulement

train = train[day_columns]


train = train.T
train.shape

# Scaling the features using min-max scaler in range 0-1

## Import de MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

## Fit les data et transform pour les scale
train_scaled = sc.fit_transform(train)

def process_data_by_timesteps(train_data, timesteps):

    print(f"Forming batches of data every {timesteps} days")

    X_train = []
    y_train = []

    ## Exploration des data
    for i in range(timesteps, 1913-timesteps):

        X_train.append(train_data[i-timesteps:i])
        y_train.append(train_data[i][0:30490])

    print(f"X_train shape: {X_train[0].shape}")
    print(f"y_train shape: {y_train[0].shape}")
    print("Converting to arrays...")

    X_train = np.array(X_train, dtype = 'float16')
    y_train = np.array(y_train, dtype = 'float16')

    print(f"X_train array shape: {X_train.shape}")
    print(f"y_train array shape: {y_train.shape}")

    print("COMPLETE!!!")

    return X_train, y_train
	timesteps=28

# Get le process du train Data

X_train_f, y_train_f = process_data_by_timesteps(train_data=train_scaled, timesteps=14)

# Crée le model

## imports tensorflow
import tensorflow as tf

input_shape = (np.array(X_train_f).shape[1], np.array(X_train_f).shape[2])

## Création du model sequential
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(30490),
])


## Compile le model
model.summary()
model.compile(optimizer='adam', metrics=['mse'], loss='mean_squared_error')

# création du callback
callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=3, min_delta=0.001)

## Entrainement du model

model.fit(X_train_f, y_train_f, epochs = 20, batch_size = 10, callbacks=[callback])

def process_test_data_and_get_predictions(train_data, timesteps):

    inputs = train_data[-timesteps:]

    inputs = sc.transform(inputs)

    # Créatin du dataset
    X_test = []
    X_test.append(inputs[0:timesteps])
    X_test = np.array(X_test)


    predictions = []

    ## Prédiction 28jour
    for j in range(timesteps,timesteps + 28):

        print(f"Observation Num: {j}")

        ### Predicting jour suivant
        forecast = model.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 30490))
        print(f"Forecast shape: {forecast.shape}")

        # Ajout des data
        X_test = np.append(X_test, forecast).reshape(1,j+1,30490)
        print(f"X_test shape: {X_test.shape}")

        # Invertion du scaling
        forecast = sc.inverse_transform(forecast)[:,0:30490]
        print(f"Forecast shape: {forecast.shape}")

        predictions.append(forecast)

    return predictions

# Recupération des predictions
predictions = process_test_data_and_get_predictions(train, timesteps)

#############################################################################################
#	Création d'un graph qui montre a quoi ressemblent les feature de sine/cosine and special-day #
#############################################################################################

fig, ax = plt.subplots(2, 1, facecolor='w', figsize=(16,6))
cal_df[['year_sin', 'year_cos']].plot(ax=ax[0])
ax[0].set_title("Sine and cosine features")
cal_df[['pre_Easter', 'post_Easter']].plot(ax=ax[1])
ax[1].set_title("Example of special-day encoding")
fig.tight_layout(pad=3.0)
plt.show()

#############################################################################################
#	Création d'un graph qui montre a quoi ressemblent les feature de sine/cosine and special-day #
#############################################################################################


import time

dataPath = '../input/m5-forecasting-accuracy/'

submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))

submission = submission.T

submission = pd.concat((submission, submission), ignore_index=True)

sample_submission = pd.read_csv(dataPath + "/sample_submission.csv")

idColumn = sample_submission[["id"]]

submission[["id"]] = idColumn

cols = list(submission.columns)
cols = cols[-1:] + cols[:-1]
submission = submission[cols]

colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]

submission.columns = colsdeneme

currentDateTime = time.strftime("%d%m%Y_%H%M%S")

submission.to_csv("submission.csv", index=False)
