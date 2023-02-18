def Cnn_Svr (en,cycle):




# Import libarires : 
    import tensorflow as tf
    from tensorflow.keras import layers,models 
    from tensorflow.keras.models import Sequential

    import numpy as np
    import pandas as pd
    import sklearn

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    import joblib





# Import sample Data: 

    columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
            ,"sensor20","sensor21"]



    #with open(test_cv, "r") as f:

        #test_data = pd.read_csv(f, sep= "\s+", header = None, names=columns) 
    test_data =pd.read_csv("/app/data/test_FD001.txt", sep= "\s+", header = None,names=columns)# lets see how to do this 

    #test_data=test_cv
    t=test_data.groupby('id').last().reset_index()
    train_data =pd.read_csv("/app/data/train_FD001.txt", sep= "\s+", header = None,names=columns) 
    true_rul = pd.read_csv("/app/data/RUL_FD001.txt", sep= "\s+", header = None) 
     # preprocessing 
    window_length = 30
    shift = 1
    early_rul = 125            
    processed_train_data = []
    processed_train_targets = []

    #train_data=pd.read_csv("/content/drive/MyDrive/Project/data/train_FD001.txt", sep= "\s+", header = None,names=columns)   
    #test_data =pd.read_csv("/content/drive/MyDrive/Project/data/test_FD001.txt", sep= "\s+", header = None,names=columns) 
    #true_rul = pd.read_csv("/content/drive/MyDrive/Project/data/RUL_FD001.txt", sep= "\s+", header = None) 



###### 
    def process_targets(data_length, early_rul = None):
            """ 
            Takes datalength and earlyrul as input and 
            creates target rul.
            """
            if early_rul == None:
                return np.arange(data_length-1, -1, -1)
            else:
                early_rul_duration = data_length - early_rul
                if early_rul_duration <= 0:
                    return np.arange(data_length-1, -1, -1)
                else:
                    return np.append(early_rul*np.ones(shape = (early_rul_duration,)), np.arange(early_rul-1, -1, -1))

    def process_input_data_with_targets(input_data, target_data = None, window_length = 1, shift = 1):
    
        num_batches = np.int(np.floor((len(input_data) - window_length)/shift)) + 1
        num_features = input_data.shape[1]
        output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches, window_length,
                                                                                                    num_features)
        if target_data is None:
            for batch in range(num_batches):
                output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
            return output_data
        else:
            output_targets = np.repeat(np.nan, repeats = num_batches)
            for batch in range(num_batches):
                output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
                output_targets[batch] = target_data[(shift*batch + (window_length-1))]
            return output_data, output_targets

    def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows = 1):
    
        max_num_test_batches = np.int(np.floor((len(test_data_for_an_engine) - window_length)/shift)) + 1
        if max_num_test_batches < num_test_windows:
            required_len = (max_num_test_batches -1)* shift + window_length
            batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                            target_data = None,
                                                                            window_length = window_length, shift = shift)
            return batched_test_data_for_an_engine, max_num_test_batches
        else:
            required_len = (num_test_windows - 1) * shift + window_length
            batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                            target_data = None,
                                                                            window_length = window_length, shift = shift)
            return batched_test_data_for_an_engine, num_test_windows


    num_test_windows = 5     
    processed_test_data = []
    num_test_windows_list = []

    columns_to_be_dropped =['id','op1','op2','op3', 'sensor1', 'sensor5', 'sensor6','sensor7','sensor10',
                    'sensor16', 'sensor18', 'sensor19']

    train_data_first_column = train_data['id']
    test_data_first_column = test_data['id']

    # Scale data for all engines
    scaler = MinMaxScaler(feature_range = (-1,1))

    train_data = scaler.fit_transform(train_data.drop(columns = columns_to_be_dropped))
    test_data = scaler.transform(test_data.drop(columns = columns_to_be_dropped))

    train_data = pd.DataFrame(data = np.c_[train_data_first_column, train_data])
    test_data = pd.DataFrame(data = np.c_[test_data_first_column, test_data])

    num_train_machines = len(train_data[0].unique())
    num_test_machines = len(test_data[0].unique())




    # Process training and test data sepeartely as number of engines in training and test set may be different.
    # As we are doing scaling for full dataset, we are not bothered by different number of engines in training and test set.

    # Process trianing data
    for i in np.arange(1, num_train_machines + 1):
        temp_train_data = train_data[train_data[0] == i].drop(columns = [0]).values
        
        # Verify if data of given window length can be extracted from training data
        if (len(temp_train_data) < window_length):
            print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
            raise AssertionError("Window length is larger than number of data points for some engines. "
                                "Try decreasing window length.")
            
        temp_train_targets = process_targets(data_length = temp_train_data.shape[0], early_rul = early_rul)
        data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets, 
                                                                                    window_length = window_length, shift = shift)
        
        processed_train_data.append(data_for_a_machine)
        processed_train_targets.append(targets_for_a_machine)

    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)

    # Process test data
    for i in np.arange(1, num_test_machines + 1):
        temp_test_data = test_data[test_data[0] == i].drop(columns = [0]).values
        
        # Verify if data of given window length can be extracted from test data
        if (len(temp_test_data) < window_length):
            print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
            raise AssertionError("Window length is larger than number of data points for some engines. "
                                "Try decreasing window length.")
        
        # Prepare test data
        test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length = window_length, shift = shift,
                                                                num_test_windows = num_test_windows)
        
        processed_test_data.append(test_data_for_an_engine)
        num_test_windows_list.append(num_windows)

    processed_test_data = np.concatenate(processed_test_data)
    true_rul = true_rul[0].values

    # Shuffle training data
    index = np.random.permutation(len(processed_train_targets))
    processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]


    #st.write(processed_train_data.shape)
    #st.write(processed_test_data.shape)
    ########## Model####################


    #load trained models 
    #loaded_model_SVR = joblib.load( "/mnt/c/Users/safsa/Desktop/RUL/SVR_model.sav")
    loaded_model_SVR = joblib.load("SVR3.joblib")
    loaded_model_CNN = joblib.load("cnn3.joblib")

    y_pred_cnn=loaded_model_CNN.predict(processed_test_data)
    y_final=loaded_model_SVR.predict(y_pred_cnn)


    #######################################################


    #Evaluation 

    preds_for_each_engine = np.split(y_final, np.cumsum(num_test_windows_list)[:-1])
    mean_pred_for_each_engine = [np.average(ruls_for_each_engine,weights=np.repeat(1/num_windows, num_windows),axis=0)
                                for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]

    RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))


    indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
    preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]

    RMSE_new = np.sqrt(mean_squared_error(true_rul, preds_for_last_example))
    #st.write(RMSE)
    #st.write(preds_for_last_example.shape)
    y=pd.DataFrame(preds_for_last_example)
    y.columns=["RUL"]
    y["engine"]=range(1,101,1)
    y=y[["engine","RUL"]]
    #st.write(y)
    y["cycle"]=t["cycle"]
    y["max_cycle"]=(y["RUL"]+y["cycle"]).astype(int)
    #st.write(y)
    #st.write(t)
    def compute_s_score(rul_true, rul_pred):

   
        diff = rul_pred - rul_true
        return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    s_score = compute_s_score(true_rul, preds_for_last_example)
#print("S-score: ", s_score)    
    #st.write(s_score)




    ######################################## Stermlit app########################### 
    
    eng=range(1,101,1)
    ################################################################
    
    #st.write(en)
    
    ######################################################################
    m=y[y["engine"]==en]["max_cycle"]
    rul=m-cycle
    
    m=y[y["engine"]==en]["max_cycle"]

    rul=m-cycle
    value = rul.iloc[0]
    value_m=m.iloc[0]
    return value
def Cnn_Svr2 (en,cycle):




# Import libarires : 
    import tensorflow as tf
    from tensorflow.keras import layers,models 
    from tensorflow.keras.models import Sequential

    import numpy as np
    import pandas as pd
    import sklearn

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    import joblib





# Import sample Data: 

    columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
            ,"sensor20","sensor21"]



    #with open(test_cv, "r") as f:

        #test_data = pd.read_csv(f, sep= "\s+", header = None, names=columns) 
    test_data =pd.read_csv("/app/data/test_FD001.txt", sep= "\s+", header = None,names=columns)# lets see how to do this 

    #test_data=test_cv
    t=test_data.groupby('id').last().reset_index()
    train_data =pd.read_csv("/app/data/train_FD001.txt", sep= "\s+", header = None,names=columns) 
    true_rul = pd.read_csv("/app/data/RUL_FD001.txt", sep= "\s+", header = None) 
     # preprocessing 
    window_length = 30
    shift = 1
    early_rul = 125            
    processed_train_data = []
    processed_train_targets = []

    #train_data=pd.read_csv("/content/drive/MyDrive/Project/data/train_FD001.txt", sep= "\s+", header = None,names=columns)   
    #test_data =pd.read_csv("/content/drive/MyDrive/Project/data/test_FD001.txt", sep= "\s+", header = None,names=columns) 
    #true_rul = pd.read_csv("/content/drive/MyDrive/Project/data/RUL_FD001.txt", sep= "\s+", header = None) 



###### 
    def process_targets(data_length, early_rul = None):
            """ 
            Takes datalength and earlyrul as input and 
            creates target rul.
            """
            if early_rul == None:
                return np.arange(data_length-1, -1, -1)
            else:
                early_rul_duration = data_length - early_rul
                if early_rul_duration <= 0:
                    return np.arange(data_length-1, -1, -1)
                else:
                    return np.append(early_rul*np.ones(shape = (early_rul_duration,)), np.arange(early_rul-1, -1, -1))

    def process_input_data_with_targets(input_data, target_data = None, window_length = 1, shift = 1):
    
        num_batches = np.int(np.floor((len(input_data) - window_length)/shift)) + 1
        num_features = input_data.shape[1]
        output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches, window_length,
                                                                                                    num_features)
        if target_data is None:
            for batch in range(num_batches):
                output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
            return output_data
        else:
            output_targets = np.repeat(np.nan, repeats = num_batches)
            for batch in range(num_batches):
                output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
                output_targets[batch] = target_data[(shift*batch + (window_length-1))]
            return output_data, output_targets

    def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows = 1):
    
        max_num_test_batches = np.int(np.floor((len(test_data_for_an_engine) - window_length)/shift)) + 1
        if max_num_test_batches < num_test_windows:
            required_len = (max_num_test_batches -1)* shift + window_length
            batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                            target_data = None,
                                                                            window_length = window_length, shift = shift)
            return batched_test_data_for_an_engine, max_num_test_batches
        else:
            required_len = (num_test_windows - 1) * shift + window_length
            batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                            target_data = None,
                                                                            window_length = window_length, shift = shift)
            return batched_test_data_for_an_engine, num_test_windows


    num_test_windows = 5     
    processed_test_data = []
    num_test_windows_list = []

    columns_to_be_dropped =['id','op1','op2','op3', 'sensor1', 'sensor5', 'sensor6','sensor7','sensor10',
                    'sensor16', 'sensor18', 'sensor19']

    train_data_first_column = train_data['id']
    test_data_first_column = test_data['id']

    # Scale data for all engines
    scaler = MinMaxScaler(feature_range = (-1,1))

    train_data = scaler.fit_transform(train_data.drop(columns = columns_to_be_dropped))
    test_data = scaler.transform(test_data.drop(columns = columns_to_be_dropped))

    train_data = pd.DataFrame(data = np.c_[train_data_first_column, train_data])
    test_data = pd.DataFrame(data = np.c_[test_data_first_column, test_data])

    num_train_machines = len(train_data[0].unique())
    num_test_machines = len(test_data[0].unique())




    # Process training and test data sepeartely as number of engines in training and test set may be different.
    # As we are doing scaling for full dataset, we are not bothered by different number of engines in training and test set.

    # Process trianing data
    for i in np.arange(1, num_train_machines + 1):
        temp_train_data = train_data[train_data[0] == i].drop(columns = [0]).values
        
        # Verify if data of given window length can be extracted from training data
        if (len(temp_train_data) < window_length):
            print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
            raise AssertionError("Window length is larger than number of data points for some engines. "
                                "Try decreasing window length.")
            
        temp_train_targets = process_targets(data_length = temp_train_data.shape[0], early_rul = early_rul)
        data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets, 
                                                                                    window_length = window_length, shift = shift)
        
        processed_train_data.append(data_for_a_machine)
        processed_train_targets.append(targets_for_a_machine)

    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)

    # Process test data
    for i in np.arange(1, num_test_machines + 1):
        temp_test_data = test_data[test_data[0] == i].drop(columns = [0]).values
        
        # Verify if data of given window length can be extracted from test data
        if (len(temp_test_data) < window_length):
            print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
            raise AssertionError("Window length is larger than number of data points for some engines. "
                                "Try decreasing window length.")
        
        # Prepare test data
        test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length = window_length, shift = shift,
                                                                num_test_windows = num_test_windows)
        
        processed_test_data.append(test_data_for_an_engine)
        num_test_windows_list.append(num_windows)

    processed_test_data = np.concatenate(processed_test_data)
    true_rul = true_rul[0].values

    # Shuffle training data
    index = np.random.permutation(len(processed_train_targets))
    processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]


    #st.write(processed_train_data.shape)
    #st.write(processed_test_data.shape)
    ########## Model####################


    #load trained models 
    #loaded_model_SVR = joblib.load( "/mnt/c/Users/safsa/Desktop/RUL/SVR_model.sav")
    loaded_model_SVR = joblib.load("SVR3.joblib")
    loaded_model_CNN = joblib.load("cnn3.joblib")

    y_pred_cnn=loaded_model_CNN.predict(processed_test_data)
    y_final=loaded_model_SVR.predict(y_pred_cnn)


    #######################################################


    #Evaluation 

    preds_for_each_engine = np.split(y_final, np.cumsum(num_test_windows_list)[:-1])
    mean_pred_for_each_engine = [np.average(ruls_for_each_engine,weights=np.repeat(1/num_windows, num_windows),axis=0)
                                for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]

    RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))


    indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
    preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]

    RMSE_new = np.sqrt(mean_squared_error(true_rul, preds_for_last_example))
    #st.write(RMSE)
    #st.write(preds_for_last_example.shape)
    y=pd.DataFrame(preds_for_last_example)
    y.columns=["RUL"]
    y["engine"]=range(1,101,1)
    y=y[["engine","RUL"]]
    #st.write(y)
    y["cycle"]=t["cycle"]
    y["max_cycle"]=(y["RUL"]+y["cycle"]).astype(int)
    #st.write(y)
    #st.write(t)
    def compute_s_score(rul_true, rul_pred):

   
        diff = rul_pred - rul_true
        return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    s_score = compute_s_score(true_rul, preds_for_last_example)
#print("S-score: ", s_score)    
    #st.write(s_score)




    ######################################## Stermlit app########################### 
    
    eng=range(1,101,1)
    ################################################################
    
    #st.write(en)
    
    ######################################################################
    m=y[y["engine"]==en]["max_cycle"]
    rul=m-cycle
    
    m=y[y["engine"]==en]["max_cycle"]

    rul=m-cycle
    value = rul.iloc[0]
    value_m=m.iloc[0]
    return value_m
                
        

   




