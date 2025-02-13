import tensorflow as tf
import numpy as np
import gc
from model import get_model
import os
import pandas as pd
import sys
import openpyxl as op
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
import statistics
import joblib

warnings.filterwarnings("ignore")

filename = 'XXX.xlsx'


def op_toexcel(data, filename):
    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]
        ws.append(data)  
        wb.save(filename)

    else:
        wb = op.Workbook() 
        ws = wb['Sheet']  
        ws.append(['MSE', 'MAE', 'RMSE', 'R2', 'PCC', 'P_value'])
        ws.append(data)  
        wb.save(filename)


def data_generator(train_esm, train_prot, train_y, batch_size):
    L = train_esm.shape[0]

    while True:
        for i in range(0, L, batch_size):
            batch_esm = train_esm[i:i + batch_size].copy()
            batch_prot = train_prot[i:i + batch_size].copy()
            batch_y = train_y[i:i + batch_size].copy()

            yield ([batch_esm, batch_prot], batch_y)


def cross_validation(train_esm, train_prot, train_label, valid_esm, valid_prot, valid_label,
                     test_esm, test_prot,
                     test_label, k, i):

    train_size = train_label.shape[0]
    val_size = valid_label.shape[0]
    batch_size = 16
    train_steps = train_size // batch_size
    val_steps = val_size // batch_size

    print(
        f"\n {k}Fold nums：Training samples: {train_esm.shape[0]},Valid samples: {valid_esm.shape[0]} ,Test samples: {test_esm.shape[0]}")
    print("valid_nums:", len(valid_label))
    print(valid_label)
    print("test_nums:", len(test_label))
    print(test_label)


    qa_model = get_model()
    valiBestModel = f'XXX.h5'


    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=valiBestModel,  
        monitor='val_loss',
        save_weights_only=True,  
        verbose=1, 
        save_best_only=True,
        mode='min',
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=0,  
        mode='min' 
    )

    train_generator = data_generator(train_esm, train_prot, train_label, batch_size)
    val_generator = data_generator(valid_esm, valid_prot, valid_label, batch_size)

    history_callback = qa_model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=70,
        verbose=1,
        callbacks=[checkpointer, early_stopping],
        validation_data=val_generator,
        validation_steps=val_steps,
        shuffle=True, 
        workers=1
    )
    

    loss = history_callback.history['loss']
    val_loss = history_callback.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title(f'TEST{i}Fold{k} Training And Validation LOSS')
    plt.xlabel('70 Epochs 20 patience')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    
    feature_layer = tf.keras.models.Model(inputs=qa_model.input, outputs=qa_model.get_layer('concatenate').output)
    train_features = feature_layer.predict([train_esm, train_prot])
    val_features = feature_layer.predict([valid_esm, valid_prot])
    test_features = feature_layer.predict([test_esm, test_prot])


    # XGBOOST
    xgb_model = XGBRegressor(gamma=0.1, learning_rate=0.05, max_depth=3, n_estimators=200)
    xgb_model.fit(train_features, train_label)
    val_pred_xgb = xgb_model.predict(val_features)
    test_pred_xgb = xgb_model.predict(test_features)
    print(f"Fold {k} - Validation Results (XGB):")
    evaluate_regression(val_pred_xgb, valid_label, save_to_file=False)
    print(f"Fold {k} - Testing Results (XGB):")
    evaluate_regression(test_pred_xgb, test_label, save_to_file=True)
    joblib.dump(xgb_model, f'XXX.pkl')

    # XGBOOST
    y_true_flat = np.ravel(test_label)
    y_pred_flat = np.ravel(test_pred_xgb)
    result_df = pd.DataFrame({
        'TRUE_label': y_true_flat,
        'PRED_label': y_pred_flat
    })
    result_end = f'XXX.csv'
    result_df.to_csv(result_end, index=False)


    train_generator.close()
    val_generator.close()
    del train_generator
    del val_generator
    gc.collect()  


def evaluate_regression(predictions, true_labels, save_to_file=False):
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_labels, predictions)
    pearson_corr, p_value = pearsonr(true_labels, predictions)

    print(
        f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, PCC: {pearson_corr:.4f}, P-value: {p_value:.4f}")

    result = [mse, mae, rmse, r2, pearson_corr, p_value]
    if save_to_file:
        op_toexcel(result, filename)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    all_esm = np.lib.format.open_memmap('../features_npy/S1586_6esm.npy')
    all_prot = np.lib.format.open_memmap('../features_npy/S1586_wild_161_prot.npy')
    all_label = np.lib.format.open_memmap('../features_npy/S1586_label.npy')

    all_label = all_label.astype(np.float)
    print(all_label.dtype)

    for i in range(1, 2):
        cv = KFold(n_splits=10, shuffle=True, random_state=42) 
        k = 1
        for train_index, test_index in cv.split(all_esm, all_label):
            train_ESM = all_esm[train_index]
            train_Prot = all_prot[train_index]
            train_Y = all_label[train_index]

            train_ESM, valid_ESM, train_Prot, valid_Prot, train_Y, valid_Y = train_test_split(
                train_ESM, train_Prot,
                train_Y, test_size=0.1,
                random_state=42)

            test_ESM = all_esm[test_index]
            test_Prot = all_prot[test_index]
            test_Y = all_label[test_index]

            cross_validation(train_ESM, train_Prot, train_Y, valid_ESM,
                             valid_Prot, valid_Y,
                             test_ESM, test_Prot,
                             test_Y, k, i)

            k += 1
