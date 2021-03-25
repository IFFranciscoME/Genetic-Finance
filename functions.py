
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Genetic Methods for Neural Nets Training for Trading                                       -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/GeneticTraining                                        -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
import data as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from scipy.stats import kurtosis as m_kurtosis
from scipy.stats import skew as m_skew
from gplearn.genetic import SymbolicTransformer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import r2_score

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, log_loss

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, models, regularizers, optimizers

from datetime import datetime
from scipy.stats import kurtosis as m_kurtosis
from scipy.stats import skew as m_skew


# ---------------------------------------------------------------------------- FEATURES BASIC STATISTICS -- #
# --------------------------------------------------------------------------------------------------------- #

def data_profile(p_data, p_type, p_mult):
    """
    OHLC Prices Profiling (Inspired in the pandas-profiling existing library)

    Parameters
    ----------

    p_data: pd.DataFrame
        A data frame with columns of data to be processed

    p_type: str
        indication of the data type: 
            'ohlc': dataframe with TimeStamp-Open-High-Low-Close columns names
            'ts': dataframe with unknown quantity, meaning and name of the columns
    
    p_mult: int
        multiplier to re-express calculation with prices,
        from 100 to 10000 in forex, units multiplication in cryptos, 1 for fiat money based assets
        p_mult = 10000

    Return
    ------

    r_data_profile: dict
        {}
    
    References
    ----------

    https://github.com/pandas-profiling/pandas-profiling

    """

    # copy of input data
    f_data = p_data.copy()

    # interquantile range
    def f_iqr(param_data):
        q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
        q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
        return  q1 - q3
    
    # outliers function (returns how many were detected, not which ones or indexes)
    def f_out(param_data):
        q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
        q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
        lower_out = len(np.where(param_data < q1 - 1.5*f_iqr(param_data))[0])
        upper_out = len(np.where(param_data > q3 + 1.5*f_iqr(param_data))[0])
        return [lower_out, upper_out]


    # -- OHLCV PROFILING -- #
    if p_type == 'ohlc':

        # initial data
        ohlc_data = p_data[['open', 'high', 'low', 'close', 'volume']].copy()

        # data calculations
        ohlc_data['co'] = round((ohlc_data['close'] - ohlc_data['open'])*p_mult, 2)
        ohlc_data['hl'] = round((ohlc_data['high'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ol'] = round((ohlc_data['open'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ho'] = round((ohlc_data['high'] - ohlc_data['open'])*p_mult, 2)

        # original data + co, hl, ol, ho columns
        f_data = ohlc_data.copy()
    
    # basic data description
    data_des = f_data.describe(percentiles=[0.25, 0.50, 0.75, 0.90])

    # add skewness metric
    skews = pd.DataFrame(m_skew(f_data)).T
    skews.columns = list(f_data.columns)
    data_des = data_des.append(skews, ignore_index=False)

    # add kurtosis metric
    kurts = pd.DataFrame(m_kurtosis(f_data)).T
    kurts.columns = list(f_data.columns)
    data_des = data_des.append(kurts, ignore_index=False)
    
    # add outliers count
    outliers = [f_out(param_data=f_data[col]) for col in list(f_data.columns)]
    negative_series = pd.Series([i[0] for i in outliers], index = data_des.columns)
    positive_series = pd.Series([i[1] for i in outliers], index = data_des.columns)
    data_des = data_des.append(negative_series, ignore_index=True)
    data_des = data_des.append(positive_series, ignore_index=True)
    
    # index names
    data_des.index = ['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'p90',
                      'max', 'skew', 'kurt', 'n_out', 'p_out']

    return np.round(data_des, 2)


# ------------------------------------------------------------------------------------ DATA PRE-SCALLING -- #
# ------------------------------------------------------------------------------------ ----------------- -- #

def data_scaler(p_data, p_trans):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada

    Parameters
    ----------
    p_trans: str
        Standard: Para estandarizacion (restar media y dividir entre desviacion estandar)
        Robust: Para estandarizacion robusta (restar mediana y dividir entre rango intercuartilico)

    p_datos: pd.DataFrame
        Con datos numericos de entrada

    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados

    """

    # hardcopy of the data
    data = p_data.copy()
    # list with columns to transform
    lista = data[list(data.columns)]
    # choose to scale from 1 in case timestamp is present
    scale_ind = 1 if 'timestamp' in list(data.columns) else 0
    
    if p_trans == 'standard':
        
        # removes the mean and scales the data to unit variance
        data[list(data.columns[scale_ind:])] = StandardScaler().fit_transform(lista.iloc[:, scale_ind:])
        return data

    elif p_trans == 'robust':

        # removes the meadian and scales the data to inter-quantile range
        data[list(data.columns[scale_ind:])] = RobustScaler().fit_transform(lista.iloc[:, scale_ind:])
        return data

    elif p_trans == 'scale':

        # scales to max value
        data[list(data.columns[scale_ind:])] = MaxAbsScaler().fit_transform(lista.iloc[:, scale_ind:])
        return data
    
    else:
        print('Error in data_scaler, p_trans value is not valid')


# ------------------------------------------------------------------------------ Autoregressive Features -- #
# --------------------------------------------------------------------------------------------------------- #

def autoregressive_features(p_data, p_memory):
    """
    Creacion de variables de naturaleza autoregresiva (resagos, promedios, diferencias)

    Parameters
    ----------
    p_data: pd.DataFrame
        with OHLCV columns: Open, High, Low, Close, Volume

    p_memory: int
        A value that represents the implicit assumption of a "memory" effect in the prices

    Returns
    -------
    r_features: pd.DataFrame
        

    """

    # work with a separate copy of original data
    data = p_data.copy()

    # nth-period final price "movement"
    data['co'] = (data['close'] - data['open'])
    # nth-period uptrend movement
    data['ho'] = (data['high'] - data['open'])
    # nth-period downtrend movement
    data['ol'] = (data['open'] - data['low'])
    # nth-period volatility measure
    data['hl'] = (data['high'] - data['low'])

    # N features with window-based calculations
    for n in range(0, p_memory):

        data['ma_ol'] = data['ol'].rolling(n + 2).mean()
        data['ma_ho'] = data['ho'].rolling(n + 2).mean()
        data['ma_hl'] = data['hl'].rolling(n + 2).mean()
        
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)

        data['sd_ol_' + str(n + 1)] = data['ol'].rolling(n + 1).std()
        data['sd_ho_' + str(n + 1)] = data['ho'].rolling(n + 1).std()
        data['sd_hl_' + str(n + 1)] = data['hl'].rolling(n + 1).std()

        data['lag_vol_' + str(n + 1)] = data['volume'].shift(n + 1)
        data['sum_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).sum()
        data['mean_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).mean()

    # timestamp as index
    data.index = pd.to_datetime(data.index)
    # select columns, drop for NAs, change column types, reset index
    r_features = data.drop(['open', 'high', 'low', 'close', 'hl', 'ol', 'ho', 'volume'], axis=1)
    r_features = r_features.dropna(axis='columns', how='all')
    r_features = r_features.dropna(axis='rows')
    r_features.iloc[:, 1:] = r_features.iloc[:, 1:].astype(float)
    r_features.reset_index(inplace=True, drop=True)

    return r_features


# ---------------------------------------------------------- FUNCTION: Autoregressive Feature Engieering -- #
# ---------------------------------------------------------- ---------------------------------------------- #

def linear_features(p_data, p_memory, p_target):
    """
    autoregressive process for feature engineering

    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos
        p_data = m_folds['periodo_1']

    p_memory: int
        valor de memoria maxima para hacer calculo de variables autoregresivas
        p_memory = 7

    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'val_x': pd.DataFrame, 'val_y': pd.DataFrame}

    References
    ----------

    """

    # hardcopy of data
    data = p_data.copy()

    # funcion para generar variables autoregresivas
    data_ar = autoregressive_features(p_data=data, p_memory=p_memory)

    # y_t = y_t+1 in order to prevent filtration, that is, at time t, the target variable y_t
    # with the label {co_d}_t will be representing the direction of the price movement (0: down, 1: high) 
    # that was observed at time t+1, and so on applies to t [0, n-1]. the last value is droped
    data_ar[p_target] = data_ar[p_target].shift(-1, fill_value=999)
    data_ar = data_ar.drop(data_ar[p_target].index[[-1]])

    # separacion de variable dependiente
    data_y = data_ar[p_target].copy()

    # separacion de variables independientes
    data_arf = data_ar.drop(['timestamp', p_target], axis=1, inplace=False)

    # datos para utilizar en la siguiente etapa
    next_data = pd.concat([data_y.copy(), data_arf.copy()], axis=1)

    # keep the timestamp as index
    next_data.index = data_ar['timestamp'].copy()
  
    return next_data


# ------------------------------------------------------------------------------------ Symbolic Features -- #
# --------------------------------------------------------------------------------------------------------- #

def symbolic_features(p_x, p_y, p_params):
    """
    Feature engineering process with symbolic variables by using genetic programming. 

    Parameters
    ----------
    p_x: pd.DataFrame / np.array / list
        with regressors or predictor variables

        p_x = data_features.iloc[:, 1:]

    p_y: pd.DataFrame / np.array / list
        with variable to predict

        p_y = data_features.iloc[:, 0]

    p_params: dict
        with parameters for the genetic programming function

        p_params = {'functions': ["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
        'population': 5000, 'tournament':20, 'hof': 20, 'generations': 5, 'n_features':20,
        'init_depth': (4,8), 'init_method': 'half and half', 'parsimony': 0.1, 'constants': None,
        'metric': 'pearson', 'metric_goal': 0.65, 
        'prob_cross': 0.4, 'prob_mutation_subtree': 0.3,
        'prob_mutation_hoist': 0.1. 'prob_mutation_point': 0.2,
        'verbose': True, 'random_cv': None, 'parallelization': True, 'warm_start': True }

    Returns
    -------
    results: dict
        With response information

        {'fit': model fitted, 'params': model parameters, 'model': model,
         'data': generated data with variables, 'best_programs': models best programs}

    References
    ----------
    https://gplearn.readthedocs.io/en/stable/reference.html#gplearn.genetic.SymbolicTransformer
    
    
    **** NOTE ****

    simplified internal calculation for correlation (asuming w=1)
    
    y_pred_demean = y_pred - np.average(y_pred)
    y_demean = y - np.average(y)

                              np.sum(y_pred_demean * y_demean)
    pearson =  ---------------------------------------------------------------
                np.sqrt((np.sum(y_pred_demean ** 2) * np.sum(y_demean ** 2)))  

    """
     
    # Function to produce Symbolic Features
    model = SymbolicTransformer(function_set=p_params['functions'], population_size=p_params['population'],
                                tournament_size=p_params['tournament'], hall_of_fame=p_params['hof'],
                                generations=p_params['generations'], n_components=p_params['n_features'],

                                init_depth=p_params['init_depth'], init_method=p_params['init_method'],
                                parsimony_coefficient=p_params['parsimony'],
                                const_range=p_params['constants'],
                                
                                metric=p_params['metric'], stopping_criteria=p_params['metric_goal'],

                                p_crossover=p_params['prob_cross'],
                                p_subtree_mutation=p_params['prob_mutation_subtree'],
                                p_hoist_mutation=p_params['prob_mutation_hoist'],
                                p_point_mutation=p_params['prob_mutation_point'],
                                max_samples=p_params['max_samples'],

                                verbose=p_params['verbose'], warm_start=p_params['warm_start'],
                                random_state=123, n_jobs=-1 if p_params['parallelization'] else 1,
                                feature_names=p_x.columns)

    # SymbolicTransformer fit
    model_fit = model.fit_transform(p_x, p_y)

    # output data of the model
    data = pd.DataFrame(model_fit)

    # parameters of the model
    model_params = model.get_params()

    # best programs dataframe
    best_programs = {}
    for p in model._best_programs:
        factor_name = 'sym' + str(model._best_programs.index(p))
        best_programs[factor_name] = {'raw_fitness': p.raw_fitness_, 'reg_fitness': p.fitness_, 
                                      'expression': str(p), 'depth': p.depth_, 'length': p.length_}

    # format and sorting
    best_programs = pd.DataFrame(best_programs).T
    best_programs = best_programs.sort_values(by='raw_fitness', ascending=False)

    # results
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data,
               'best_programs': best_programs, 'details': model.run_details_}

    return results


# ----------------------------------------------------------- Genetic Programming for Feature Engieering -- #
# --------------------------------------------------------------------------------------------------------- #

def genetic_programed_features(p_data, p_target, p_params):
    """
    El uso de programacion genetica para generar variables independientes simbolicas

    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos
        
        p_data = m_folds['periodo_1']

    p_split: int
        split in val

        p_split = '0'

    p_params:
        parameters for symbolic_features process 

    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'val_x': pd.DataFrame, 'val_y': pd.DataFrame}

    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """
   
    # separacion de variable dependiente
    datos_y = p_data[p_target].copy().astype(int)

    # separacion de variables independientes
    datos_had = p_data.copy().drop([p_target], axis=1, inplace=False)

    # Lista de operaciones simbolicas
    sym_data = symbolic_features(p_x=datos_had, p_y=datos_y, p_params=p_params)

    # Symbolic variables output
    datos_sym = sym_data['data'].copy()
    datos_sym.columns = ['sym_' + str(i) for i in range(0, len(sym_data['data'].iloc[0, :]))]
    datos_sym.index = datos_y.index
   
    return {'sym_data': sym_data, 'sym_features': datos_sym}


# ------------------------------------------------------------------------------------------- Data Split -- #
# --------------------------------------------------------------------------------------------------------- #

def data_split(p_data, p_target, p_split):
    
    # separacion de variable dependiente
    datos_y = p_data[p_target].copy().astype(float)

    # if size != 0 then an inner fold division is performed with size*100 % as val and the rest for train
    size = float(p_split)/100
           
    # automatic data sub-sets division according to inner-split
    xtrain, xval, ytrain, yval = train_test_split(p_data, datos_y, test_size=size, shuffle=False)

    return {'train_x': xtrain, 'train_y': ytrain, 'val_x': xval, 'val_y': yval}



# -------------------------- MODEL: Multivariate Linear Regression Model with ELASTIC NET regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def logistic_net(p_data, p_params):
    """
    Funcion para ajustar varios modelos lineales

    Parameters
    ----------
    p_data: dict
        Diccionario con datos de entrada como los siguientes:

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]

    p_params: dict
        Diccionario con parametros de entrada para modelos, como los siguientes

        p_alpha: float
                alpha for the models
                p_alpha = 0.1

        p_ratio: float
            elastic net ratio between L1 and L2 regularization coefficients
            p_ratio = 0.1

        p_iterations: int
            Number of iterations until stop the model fit process
            p_iter = 200

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    References
    ----------
    ElasticNet
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    """

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # alpha, l1_ratio,

    # computations parameters
    # fit_intercept, normalize, precompute, copy_X, tol, warm_start, positive, selection

    # Fit model
    en_model = LogisticRegression(l1_ratio=p_params['ratio'], C=p_params['c'], tol=1e-3,
                                  penalty='elasticnet', solver='saga', multi_class='ovr', n_jobs=-1,
                                  max_iter=1e6, fit_intercept=False, random_state=123)

    # model fit
    en_model.fit(p_data['train_x'].copy(), p_data['train_y'].copy())

   # performance metrics of the model
    metrics_en_model = model_metrics(p_model=en_model, p_model_data=p_data.copy())

    return metrics_en_model


# --------------------------------------------------- MODEL: Artificial Neural Net Multilayer Perceptron -- #
# --------------------------------------------------------------------------------------------------------- #

def ann_mlp(p_data, p_params):
   
    x_train = p_data['train_x']
    x_test = p_data['val_x']
    y_train = p_data['train_y']
    y_test = p_data['val_y']

    candidate = 8
    learning_rate = 0.001
    epochs = 200
    batch = 32

    # Neural network architecture
    model = Sequential()
    model.add(Dense(candidate*2, activation='sigmoid', input_dim=x_train.shape[1]))
    model.add(Dense(1, activation='linear'))
    opt = SGD(lr=learning_rate)

    model.compile(loss = 'mean_squared_error', optimizer=opt, metrics=['mse'])

    # fit the model
    model_history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch,
                            validation_data=(x_test, y_test), verbose=2)

    model_score = model.evaluate(x_test, y_test)

    return {'history': model_history, 'score': model_score}