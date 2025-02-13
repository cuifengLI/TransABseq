import tensorflow as tf
import numpy as np
import gc
import tensorflow.keras
from model import get_model
import os
import pandas as pd
import sys
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import openpyxl as op
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
import statistics

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
    
    rmse = history_callback.history['root_mean_squared_error']
    val_rmse = history_callback.history['val_root_mean_squared_error']
    epochs = range(1, len(rmse) + 1)

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

    train_generator.close()
    val_generator.close()
    del train_generator
    del val_generator
    gc.collect() 

   
    print(
        f"\n{k}fold valid_results：Validation Loss: {history_callback.history['val_loss'][-1]:.4f}," + f"Validation RMSE: {history_callback.history['val_root_mean_squared_error'][-1]:.4f}")

    print(f"Fold {k} - Testing:")  
    print(test_esm.shape)
    print(test_prot.shape)
    test_pred = qa_model.predict([test_esm, test_prot]).reshape(-1, )

    evaluate_regression(test_pred, test_label)  

    
    y_true_flat = np.ravel(test_label)
    y_pred_flat = np.ravel(test_pred)
    result_df = pd.DataFrame({
        'TRUE_label': y_true_flat,
        'PRED_label': y_pred_flat
    })
    result_end = f'XXX.csv'
    result_df.to_csv(result_end, index=False)


def evaluate_regression(test_pred, test_label):
    y_pred = test_pred
    y_true = test_label

    print("label：" + str(test_label))
    print("pred_label：" + str(test_pred))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearsonr_corr, p_value = pearsonr(y_true, y_pred)

    print(f"\nFold {k} - Mean Squared Error (MSE): {mse:.4f}")
    print(f"Fold {k} - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Fold {k} - RMSE: {rmse:.4f}")
    print(f"Fold {k} - R-squared (R2) Score: {r2:.4f}")
    print(f"Fold {k} - PCC: {pearsonr_corr:.4f}")
    print(f"Fold {k} - P-value: {p_value:.4f}")

    result = mse, mae, rmse, r2, pearsonr_corr, p_value
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
