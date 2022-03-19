import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def create_model(model_version = 0):

    model = []

    if model_version == 0:
        model = Sequential()

        model.add(Conv1D(filters = 16, kernel_size = 5, activation ='relu', input_shape = (35, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(8, 5, activation ='relu'))
        model.add(Conv1D(8, 5, activation ='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(32, activation ='tanh'))
        model.add(Dropout(0.5))
        model.add(Flatten())

        model.add(Dense(5, activation = 'softmax'))

    if model_version == 1:
        model = Sequential()

        model.add(LSTM(32, input_shape=(35, 1), activation='tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(5, activation='softmax'))

    return model

def train_ACA_model(X_AC_train, y_train, NN, NN_model):

    if NN:

        clf_AC = NN_model

        clf_AC.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        X_ = X_AC_train.copy()
        y_ = y_train.copy()

        h = clf_AC.fit(X_, y_, shuffle=True, batch_size=128, epochs=150,
                       verbose=1, validation_split=0.25, callbacks=EarlyStopping(monitor='val_loss', patience=15))

        clf_AC_proba = clf_AC.predict(X_AC_train)

    else:
        clf_AC = KNeighborsClassifier(n_neighbors=5)
        clf_AC.fit(X_AC_train, y_train)

        clf_AC_proba = clf_AC.predict_proba(X_AC_train)

    return clf_AC, clf_AC_proba


def train_MCA_model(X_MC_train, y_train, NN):

    clf_MC = KNeighborsClassifier(n_neighbors=5)

    clf_MC.fit(X_MC_train, y_train)

    clf_MC_proba = clf_MC.predict_proba(X_MC_train)

    return clf_MC, clf_MC_proba


def train_AMCA_model(X_AC_MC_train, y_train):

    clf = LogisticRegression(max_iter=1000, fit_intercept=False)

    clf.fit(X_AC_MC_train, y_train)

    clf_proba = clf.predict_proba(X_AC_MC_train)

    return clf, clf_proba


def sample_level_accuracy(df_in, clf_AC, encoder, NMETA=6, sample_groups='Target', true_label='CPE_type'):

    preds_sample_store = []
    cpe_percent_store = []
    samples_store = []
    true_store = []
    cpe_percent = {}

    for sample, df in tqdm(df_in.groupby(sample_groups)):
        curves = df.iloc[:, NMETA+1:].values
        y_preds_sample = encoder.inverse_transform(clf_AC.predict(curves))

        preds_sample_store.append(y_preds_sample)
        true_store.append(df[true_label].iloc[0])
        samples_store.append(sample)

    for preds in preds_sample_store:
        cpe_count = Counter(preds)
        cpe_percent = {}

        for key in list(cpe_count):
            count = cpe_count[key]
            total = sum(cpe_count.values())
            cpe_percent[key] = count/total

        cpe_none = np.setdiff1d(list(encoder.classes_), list(cpe_percent))

        if cpe_none.size > 0:
            for cpe in cpe_none:
                cpe_percent[cpe] = 0.0
    
        cpe_percent_store.append(cpe_percent)

    df_sample_acc = pd.DataFrame(data=cpe_percent_store)
    df_sample_acc['CPE'] = samples_store
    df_sample_acc['true_label'] = true_store

    return df_sample_acc