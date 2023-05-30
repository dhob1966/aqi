import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)  
import pandas as pd
pd.options.mode.chained_assignment = None  
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from scipy.signal import detrend
from sklearn.base import BaseEstimator, TransformerMixin
# Create a linear model and train
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None  
pd.options.mode.copy_on_write = True
from IPython.display import Image
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle
import statistics
from statistics import median
######################
#######################
######################
#######################
def rsquared(y_true, y_pred):    
    return np.mean((y_true-y_pred)**2)
################
################
################
################
################
################
def any_station_day(general_file,num_past_observation_prepro,
num_past_observations_model,N,name,ml_algor,path_store_model,num_estim_rf, boost_round):

    '''
    general_file               == csv file with the AQI info by stations hour/day
    num_past_observation_prepro== number of past observations used to preprocess the training data
    num_past_observations_model== number of past observations used as predictors to model our data
    N                          == denoising factor. N times rolling average used to smooth data
    name                       == name of station to be analysed 
    ml_algor                   == machine learning model to use

    returns:
    *a plot of the ground truth overlapped with the predictions made by the model on the test data 
    *a plot of the train targets used for training overlapped with the predictions made by the model on them
    *Corresponding MSE values
    * model path
        '''
    
    data = pd.read_csv(general_file)
    data = data[['StationId','Datetime','PM2.5','PM10','SO2','NOx','NH3','CO','O3','AQI','AQI_Bucket']]
          
    try: 
         #start
        train_data,test_data =get_station_training_test_data(station_name=name,full_data=data)

        #replace nans with median
        train_data['AQI'] = train_data['AQI'].fillna(train_data['AQI'].median())
        train_data['AQI_Bucket'] = train_data['AQI_Bucket'].fillna('Satisfactory')
        #replace nans with median
        test_data['AQI'] = test_data['AQI'].fillna(test_data['AQI'].median())
        test_data['AQI_Bucket'] = test_data['AQI_Bucket'].fillna('Satisfactory')

        #outliers treatment
        outliers_df=outliers_filter(train_data['AQI'],num_past_observation_prepro,3)
        #actual outliers
        outliers=outliers_df[outliers_df['outlier_signal']==1]
        #Create list with the outliers indexes to be removed
        list_outliers_indexes = outliers.index.to_list()
        #train data without outliers
        train_data_no_outliers = train_data.drop(index=list_outliers_indexes)

        #Denoising
        train_data_no_outliers_convolved=pd.Series(np.convolve(train_data_no_outliers['AQI'], np.ones(N)/N, mode='valid'))
        test_data_convolved=pd.Series(np.convolve(test_data['AQI'], np.ones(N)/N, mode='valid'))
        
        #modelling
        #data_to_model = detrend(normalized_AQI_train_data_no_outliers[num_past_observation_prepro:])
        data_to_model = train_data_no_outliers_convolved
        df_data_to_model = pd.DataFrame(data_to_model)
        df_data_to_model.rename(columns={0: "AQI"},inplace=True)

        AQI_test_data=test_data_convolved
        AQI_test_data = pd.DataFrame(AQI_test_data)
        AQI_test_data.rename(columns={0: "AQI"},inplace=True)        #TRAIN DATA
        
        #TRAIN DATA
        # we create a lagged data to use past data to predict future data, we use a window
        # e.g. if equal to 24, meaning 24 previous measures used to predict next AQI
        predictors,targets=create_lagged_data(window=num_past_observations_model,dataframe=df_data_to_model,feature="AQI")
        
        #TEST DATA
        #here we have the previous meausurement that we will input in our trained model
        #whilst targets_test represents the ground truth to verify how good our model is
        predictors_test,targets_test_smoothed = create_lagged_data(window=num_past_observations_model,dataframe=AQI_test_data,feature="AQI") 
        #we must used the actual test_data (not the smoothed one) to check the precission of our model
        targets_test=test_data[N+(num_past_observations_model-1):]['AQI']
        #trainig our model with our training data
        
        if ml_algor=='lr':

            model = LinearRegression()
            model.fit(X=predictors, y=targets)
            # save the model to disk
            filename = path_store_model#'model_LR_v-1.pkl'
            pickle.dump(model, open(filename, 'wb'))

            # using model on the same training data
            train_pred = model.predict(predictors)
            #predicting on test data
            test_pred = model.predict(predictors_test)
         
        elif ml_algor=='random_for':
                model = RandomForestRegressor(n_estimators=num_estim_rf)
                model.fit(X=predictors, y=targets)
                    # save the model to disk
                filename = path_store_model
                pickle.dump(model, open(filename, 'wb'))
                # using model on the same training data
                train_pred = model.predict(predictors)
                # Plot prediction with test data
                test_pred = model.predict(predictors_test)

        elif ml_algor=='xgboost':
            # Create regression matrices
            dtrain_reg = xgb.DMatrix(predictors, targets, enable_categorical=True)
            dtest_reg = xgb.DMatrix(predictors_test, targets_test, enable_categorical=True)
            # Define hyperparameters
            params = {"objective": "reg:squarederror", "tree_method": "hist"}

            model = xgb.train(params=params, dtrain=dtrain_reg,num_boost_round=boost_round)
            filename = path_store_model#'model_LR_v-1.pkl'
            pickle.dump(model, open(filename, 'wb'))

            train_pred = model.predict((dtrain_reg))
            test_pred = model.predict(dtest_reg)

        else:
            print('model name incorrect or model not implemented')
                    # Plot prediction with train data
        plt.figure(figsize=(14,6))
        plt.title('Train data')
        plt.plot(range(len(targets)), targets, label='train')
        plt.plot(range(len(train_pred)), train_pred, label='train_prediction')
        plt.legend()
        #print("train r2 squared error is ", rsquared(targets, train_pred))

         #Plot prediction with test data
        plt.figure(figsize=(14,6))
        plt.title('Test data')
        plt.plot(range(len(targets_test)), targets_test, label='test')
        plt.plot(range(len(test_pred)), test_pred, label='test_prediction')

        plt.legend()
        yield rsquared(targets, train_pred), rsquared(targets_test, test_pred),test_data,test_pred

    except ValueError:
        print("ERROR", "Not sufficient data for rolling averages in"+name)
        
######################
#######################
######################
#######################
def any_station_month(general_file,num_past_observations_model,name,ml_algor,path_store_model,num_estim_rf,boost_round):
    """
    Similar to any_station_day function but added
    preprocessing in order to get the AQI-median value for each 15th day of a month.
    The 15_th day is choosen arbitrarily just to have month-like data.
    
    Not enough data to include the following parameters:
    num_past_observation_prepro
    N
    """
    try: 
        data = pd.read_csv(general_file)
        data = data[data['StationId']==name]

        # To facilitate the pattern grabbing we create a column with only the date, no time.
        dates = pd.DatetimeIndex(data=data['Datetime'])
        only_date = dates.date   
        data['Date']=only_date

        # we extract the unique dates values from the dataframe
        only_date_unique =list(only_date)
        only_date_unique = list(dict.fromkeys(only_date_unique))

        #we calculate the median value of each 15th day 
        #for each unique date, leaving only one value per date
        medians_15_th=[]
        for i in range(len(only_date_unique)):
            df = data.where(data["Date"]==only_date_unique[i])
            medians_15_th.append(df['AQI'].median())

        #Now we leave only one value per date of the dataframe, we choose '00:00:00', 
        #this is arbitrary
        #could be any since the value we will focus from now on is the median value
        mask=data.Datetime.str.contains('00:00:00')
        data_with_one_entry_per_day = data[mask]
        data_with_one_entry_per_day['AQI_median']=medians_15_th

        #we extract only the data we will use
        data_clean = data_with_one_entry_per_day[["Datetime","StationId","AQI_median"]]

        train_data=data_clean[:round(0.75*len(data_clean))]    
        test_data=data_clean[round(0.75*len(data_clean)):]  

        #replace nans with median
        train_data['AQI_median'] = train_data['AQI_median'].fillna(train_data['AQI_median'].median()) 

        # we recalculate the AQI_bucket value for the AQI_median
        aqi_bucket_median_train=[]
        for i in train_data["AQI_median"]:
                if round(i)<=50:
                    aqi_bucket_median_train.append("Good")
                elif 51 <= round(i) <=100:
                    aqi_bucket_median_train.append('Satisfactory')
                elif 101<=round(i)<=200:
                    aqi_bucket_median_train.append("Moderate")
                elif 201<=round(i)<=300:
                    aqi_bucket_median_train.append('Poor')
                elif 301<=round(i)<=400:
                    aqi_bucket_median_train.append('Very poor')
                elif round(i)>=401:
                    aqi_bucket_median_train.append('Severe')
                else:
                    aqi_bucket_median_train.append('ERROR',i)
        # we recalculate the AQI_bucket value for the AQI_median
        aqi_bucket_median_test=[]
        for i in test_data["AQI_median"]:
                if round(i)<=50:
                    aqi_bucket_median_test.append("Good")
                elif 51 <= round(i) <=100:
                    aqi_bucket_median_test.append('Satisfactory')
                elif 101<=round(i)<=200:
                    aqi_bucket_median_test.append("Moderate")
                elif 201<=round(i)<=300:
                    aqi_bucket_median_test.append('Poor')
                elif 301<=round(i)<=400:
                    aqi_bucket_median_test.append('Very poor')
                elif round(i)>=401:
                    aqi_bucket_median_test.append('Severe')
                else:
                    aqi_bucket_median_test.append('ERROR',i)

        train_data["AQI_Bucket_median"]=aqi_bucket_median_train
        test_data["AQI_Bucket_median"]=aqi_bucket_median_test

        #data to build our model
        data_to_model = train_data["AQI_median"] #data_to_model = detrend(train_data_no_outliers['AQI'])
        df_data_to_model = pd.DataFrame(data_to_model)
        df_data_to_model.rename(columns={0: "AQI"},inplace=True)

        #data to test our model
        AQI_test_data = test_data["AQI_median"] #data_to_model = detrend(train_data_no_outliers['AQI'])
        AQI_test_data = pd.DataFrame(AQI_test_data)
        AQI_test_data.rename(columns={0: "AQI"},inplace=True)
        # we create a lagged data to use past data to predict future data, if we use a window equal to 30, leans 30 previous measurements were taken to predict AQI the next day.
        #TRAIN DATA
        #num_past_observations_model=2
        predictors,targets=create_lagged_data(window=num_past_observations_model,dataframe=df_data_to_model,feature="AQI_median")

        #TEST DATA
        #here we have the previous meausurement that we will input in our trained model
        #whilst targets_test represents the ground truth to verify how good our model is
        predictors_test,targets_test = create_lagged_data(window=num_past_observations_model,dataframe=AQI_test_data,feature="AQI_median") 

        #modelling
        if ml_algor=='lr':
            model = LinearRegression()
            model.fit(X=predictors, y=targets)
            # save the model to disk
            filename = path_store_model#'model_LR_v-1.pkl'
            pickle.dump(model, open(filename, 'wb'))

            # using model on the same training data
            train_pred = model.predict(predictors)
            test_pred = model.predict(predictors_test)


            rsquare_train= rsquared(targets, train_pred) #wrong! train data is train data! rsquared(train_data_no_outliers[N+(num_past_observations_model-1):]['AQI'], train_pred) # 
            rsquare_test= rsquared(targets_test, test_pred)# rsquared(targets_test, test_pred)
 

        elif ml_algor=='random_for':
            model = RandomForestRegressor(n_estimators=num_estim_rf)
            model.fit(X=predictors, y=targets)
                # save the model to disk
            filename = path_store_model#'model_LR_v-1.pkl'
            pickle.dump(model, open(filename, 'wb'))
            # using model on the same training data
            train_pred = model.predict(predictors)
            # Plot prediction with test data
            test_pred = model.predict(predictors_test)
        # R squared score for test data

        elif ml_algor=='xgboost':
            # Create regression matrices
            dtrain_reg = xgb.DMatrix(predictors, targets, enable_categorical=False)
            dtest_reg = xgb.DMatrix(predictors_test, targets_test, enable_categorical=False)
            # Define hyperparameters
            params = {"objective": "reg:squarederror", "tree_method": "hist"}

            model = xgb.train(params=params, dtrain=dtrain_reg,num_boost_round=boost_round)
            
            filename = path_store_model#'model_LR_v-1.pkl'
            pickle.dump(model, open(filename, 'wb'))
            train_pred = model.predict((dtrain_reg))
            test_pred = model.predict(dtest_reg)

        else:
            print('model name incorrect or model not implemented')
                     # Plot prediction with train data
        plt.figure(figsize=(14,6))
        plt.title('Train data')
        plt.plot(range(len(targets)), targets, label='train')
        plt.plot(range(len(train_pred)), train_pred, label='train_prediction')
        plt.legend()
        #print("train r2 squared error is ", rsquared(targets, train_pred))

        # Plot prediction with test data
        plt.figure(figsize=(14,6))
        plt.title('Test data')
        plt.plot(range(len(targets_test)), targets_test, label='test')
        plt.plot(range(len(test_pred)), test_pred, label='test_prediction')

        plt.legend()   

        yield rsquared(targets, train_pred), rsquared(targets_test, test_pred),test_data,test_pred
    
    except ValueError:
        print("ERROR", "Not sufficient data for rolling averages in "+name)
    
    #return test_data,test_pred
######################
#######################
######################
#######################
def precision_array(num_past_observations_model,N,test_data_AQI, test_pred_AQI,day=True):  #AQI_Bucket,AQI_Bucket_median
    """
    num_past_observation_prepro== number of past observations used to preprocess the training data.

    N                          == denoising factor. N times rolling average used to smooth data.
    
    test_data_AQI              == test_data including AQI_Bucket info
    test_pred_AQI              == numerical AQI test predictions
    Day= True for daily predictions, False otherwise.
    
    n factor accounts for previous laggs on the data used to grab past data.

    Returns: array with score difference. If the score is 0 the prediction is accurate.
    If it's negative means that we predicted cleaner air quality than actually is. e.g. -1 could mean we predicted "good" air quality but in reality was just satisfactory.

    If it's positive means that we predicted less clean air quality than actually is. e.g. 1 could mean we predicted "satisfactory" air quality but in reality was good.

    The larger the number the larger we missed the right category (i.e. good,satifactpry, moderate, etc.)

    On these grounds we will prefer always a positive difference over a negative one.
    """

    n=num_past_observations_model+N-1
    
    aqi_bucket_ground_truth = test_data_AQI[n:]
    if day:

        aqi_bucket_ground_truth=np.array(aqi_bucket_ground_truth['AQI_Bucket'])
    else:
        aqi_bucket_ground_truth=np.array(aqi_bucket_ground_truth['AQI_Bucket_median'])
    enconded_ground=[]
    for i in aqi_bucket_ground_truth:
        if i=="Good":
            enconded_ground.append(1)
        elif i=="Satisfactory":
            enconded_ground.append(2)
        elif i=="Moderate":
            enconded_ground.append(3)
        elif i=="Poor":
            enconded_ground.append(4)
        elif i=="Very poor":
            enconded_ground.append(5)
        elif i=="Severe":
            enconded_ground.append(6)
        else:
            print("ERROR")

    #converting predicted aqi to aqi_bucket to precission score
    aqi_bucket_pred=[]
    encoded_predictions=[]
    for i in test_pred_AQI:
        if round(i)<=50:
            aqi_bucket_pred.append("Good")
            encoded_predictions.append(1)
        elif 51 <= round(i) <=100:
            aqi_bucket_pred.append('Satisfactory')
            encoded_predictions.append(2)
        elif 101<=round(i)<=200:
            aqi_bucket_pred.append("Moderate")
            encoded_predictions.append(3)
        elif 201<=round(i)<=300:
            aqi_bucket_pred.append('Poor')
            encoded_predictions.append(4)
        elif 301<=round(i)<=400:
            aqi_bucket_pred.append('Very poor')
            encoded_predictions.append(5)
        elif round(i)>=401:
            aqi_bucket_pred.append('Severe')
            encoded_predictions.append(6)
        else:
            aqi_bucket_pred.append('ERROR',i)
    yield aqi_bucket_pred, np.array(encoded_predictions) - np.array(enconded_ground)
######################
#######################
######################
#######################
def create_lagged_data(window,dataframe,feature):
    '''
    takes a lagging-window, a dataframe to be lagged and the feature/s of interest
    returns dataframe with data-0 and n lagged features
    '''
    lags_list=[]
    for i in range(window):
        lags_list.append('lag{}'.format(i+1))
        dataframe['lag{}'.format(i+1)]=dataframe[feature].shift(i+1)
        predictors=dataframe[lags_list][window:]
        targets=dataframe[feature][window:]
    return predictors,targets
######################
#######################
######################
#######################
def get_station_training_test_data(station_name,full_data):
    '''
    gets the name of the desired station to be analized plus the dataframe with
    all station's data. 
    Returns two sets one for train one for test
    '''
    data_station=full_data[full_data['StationId']==station_name]
    train_data=data_station[:round(0.75*len(data_station))]
    test_data=data_station[round(0.75*len(data_station)):]
    return train_data,test_data
######################
#######################
######################
#######################

######################
#######################
######################
#######################
# code used from:
#https://medium.com/swlh/5-tips-for-working-with-time-series-in-python-d889109e676d
def outliers_filter(data,window,threshold,mode='rolling'):
    """ouliers Filter.
    
    Mark as outliers the points that are out of the interval:
    (mean - threshold * std, mean + threshold * std ).
    
    Parameters
    ----------
    data : pandas.Series
        The time series to filter.
    mode : str, optional, default: 'rolling'
        Whether to filter in rolling or expanding basis.
    window : int, optional, default: 262
        The number of periods to compute the mean and standard
        deviation.
    threshold : int, optional, default: 3
        The number of standard deviations above the mean.
        
    Returns
    -------
    series : pandas.DataFrame
        Original series and marked outliers.
    """
    msg = f"Type must be of pandas.Series but {type(data)} was passed."
    assert isinstance(data, pd.Series), msg
    
    series = data.copy()
    
    # rolling/expanding objects
    pd_object = getattr(series, mode)(window=window)
    mean = pd_object.mean()
    std = pd_object.std()
    
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    outliers = ~series.between(lower_bound, upper_bound)
    # fill false positives with 0
    outliers.iloc[:window] = np.zeros(shape=window)
    
    series = series.to_frame()
    series['outliers'] = np.array(outliers.astype('int').values)
    series.columns = ['Value', 'outlier_signal']
    
    return series
############################
############################
############################