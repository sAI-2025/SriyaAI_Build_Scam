import math
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from numpy import array, argsort
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
import xgboost as xg
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Initializer
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# tv= ''
# classes = []

class sxirl_engine:
    
    def __init__(self, target_type, target, buynobuy, sxi_dataframe, classes,primkey):
            self.target_type = target_type
            self.target = target
            self.buynobuy = buynobuy
            self.sxi_dataframe = sxi_dataframe
            self.classes = classes
            self.primkey = primkey

    def single_correlation(self,x,y):
        oa = pearsonr(x, y)[0]
        if np.isnan(oa) or np.isinf(oa):
            print(f"Number of Inf in pearsonr(x, y)[0]: {np.isinf(oa).sum()}")
            print(f"Number of NaN in pearsonr(x, y)[0]: {np.isnan(oa).sum()}")
            print("Pearson correlation returned NaN or Inf. Check data for constant values or invalid inputs.")
        return oa

    def bivarient_correlation(self,df):
        data = df.to_numpy().astype(float)
        length_of_parameter = data.shape[1]
        
        # Initialize matrices for correlations
        raw = np.zeros((length_of_parameter, length_of_parameter))
        abss = np.zeros((length_of_parameter, length_of_parameter))
        
        # Compute pairwise correlations
        for i in range(length_of_parameter):
            for j in range(length_of_parameter):
                if i != j:
                    corr = self.single_correlation(data[:, i], data[:, j])
                    raw[i, j] = corr
                    abss[i, j] = abs(corr)
        
        # Compute the mean of absolute correlations, ignoring NaNs
        abs_bivarient_correlation = np.nanmean(abss, axis=1) / 2
        return abs_bivarient_correlation.tolist()

    def weight(self,df,b):
        W=[]
        for i in range(df.shape[1]):
            w=1-(b[i]/2)
            W.append(w)
        return W

    def dxi(self,new_df,w):
        new_df = np.array(new_df, dtype=float)
        w = np.array(w, dtype=float)
        # Compute the dot product
        w_df = np.dot(new_df, w.T)
        # Calculate dxi using vectorized operations
        print("Length of the",len(w),type(w))
        dxi = (w_df * 100) / len(w)
        return dxi.tolist()

    def create_label_forward_time_weighted(self, target, decay_factor=0.95):
        forward_values = []
        weighted_sum = 0
        weight_sum = 0

        for i in range(len(target)):
            weight = decay_factor ** (len(target) - i - 1)  # Apply a decaying weight
            weighted_sum += target[i] * weight
            weight_sum += weight

            # Normalize the weighted sum by total weight
            forward_value = weighted_sum / weight_sum if weight_sum != 0 else target.mean()
            forward_values.append(forward_value)
        
        return forward_values

    def normalize(self,df):
        min_max = df.iloc[2].values
        max_values = df.iloc[1].astype(float).values
        column_names = df.columns
        
        # Create a copy of the dataframe starting from the 5th row
        new_df = df.iloc[4:].copy()
        
        # Vectorized operations for MAX and MIN normalization
        max_mask = min_max == 'MAX'
        min_mask = ~max_mask  # MIN mask is the inverse of MAX mask
        
        # Apply normalization
        new_df.iloc[:, max_mask] = new_df.iloc[:, max_mask].astype(float) / max_values[max_mask]
        new_df.iloc[:, min_mask] = (max_values[min_mask] - new_df.iloc[:, min_mask].astype(float)) / max_values[min_mask] 
        return new_df.astype(float)

    def create_label(self,total_dxi,Avg_dxi):
        y=[]
        for i in range(len(total_dxi)):
            if np.all(total_dxi[i]>Avg_dxi):
                y.append(1)
            else:
                y.append(0)
        return y

    def create_label_forward(self,composite_dxi1,avg_composite_dxi):
        y_forward=[]
        for i in range(len(composite_dxi1)):

            if np.all(composite_dxi1[i] > avg_composite_dxi):
                y_forward.append(0)
            else:
                y_forward.append(1)
        return y_forward

    def total_dxi(self,df1,weight):
        total_dxi=self.dxi(df1,weight)
        Avg_dxi=np.mean(np.array(total_dxi))
        return total_dxi, Avg_dxi,weight

    def catogorical_dxi(self,df,weight,parameter):
        column_name= df.columns
        behaviour_column_name=[]
        transactional_column_name=[]
        visual_column_name=[]
        kpi_column_name=[]
        
        weight_behaviour=[]
        weight_transactional=[]
        weight_visual=[]
        weight_kpi=[]
        
        for i in range(len(parameter)):
            if parameter[i]=='Behavior':
                behaviour_column_name.append(column_name[i]) 
                weight_behaviour.append(weight[i])
            elif parameter[i]=='Transactional':
                transactional_column_name.append(column_name[i])
                weight_transactional.append(weight[i])
            elif parameter[i]=='visual':
                visual_column_name.append(column_name[i])
                weight_visual.append(weight[i])
            else:
                kpi_column_name.append(column_name[i])
                weight_kpi.append(weight[i])
            
        new_df=df
        
        df_behaviour = new_df[behaviour_column_name]
        df_transactional=new_df[transactional_column_name]
        df_visual=new_df[visual_column_name]
        df_kpi=new_df[kpi_column_name]
    
        behaviour_dxi=self.dxi(df_behaviour,weight_behaviour)
        #avg_behaviour_dxi=mean(behaviour_dxi)
        transactional_dxi=self.dxi(df_transactional,weight_transactional)
        visual_dxi=self.dxi(df_visual,weight_visual)
        kpi_dxi=self.dxi(df_kpi,weight_kpi)
        return behaviour_dxi,transactional_dxi,kpi_dxi,visual_dxi

    def svm_feature_selection(self,x,y):
        from sklearn.svm import LinearSVC,SVC
        #lsvc=SVC(gamma='auto')
        lsvc = LinearSVC(C=0.4,penalty='l1', dual=False).fit(x, y)
        coeff=lsvc.coef_
        coeff = abs(coeff)
        return coeff.reshape(-1)

    def mutual_information(self,x,y):
        from sklearn.feature_selection import mutual_info_classif
        x= mutual_info_classif(x, y,n_jobs =-1)
        for i in range(len(x)):
            if x[i] < .1 :
                x[i]=0
        return x
    
    def mutual_informationreg(self,x,y):
        from sklearn.feature_selection import mutual_info_regression
        x= mutual_info_regression(x, y,n_jobs =-1)
        for i in range(len(x)):
            if x[i] < .1 :
                x[i]=0
        return x


    def tree_feature_selection(self,x,y):
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=100)
        clf = clf.fit(x, y)          
        coeff = clf.feature_importances_ 
        return coeff

    def pca_feature_selection(self,x,y):
        from sklearn.feature_selection import VarianceThreshold 
        selector = VarianceThreshold() 
        selector.fit_transform(x) 
        print(selector.variances_)
        x=selector.variances_
        for i in range(len(x)):
            if x[i] < .1 :
                
                x[i]=0
        return x

    def lasso_feature_selection(self,x,y):
        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=0.01)
        clf.fit(x,y)
        coeff = clf.coef_
        return abs(coeff)

    def composite_weight(self,df,svm_weight,avg_dxi):
        svm_index=(argsort(svm_weight))
        
        def Reverse(lst): 
            new_lst = lst[::-1] 
            return new_lst
        
        svm_ind=Reverse(svm_index)
        
        column_name=df.columns
        svm_indd=[]
        svm_weights=[]
        
        for i in range(len(svm_index)):
            if svm_weight[svm_ind[i]] !=0:
                svm_indd.append(svm_ind[i])
                svm_weights.append(svm_weight[svm_ind[i]])
            
        update_column=column_name[svm_indd]
        xx=df[update_column]
        
        while(1):
            svm_dxi,avg_svm_dxi,w_svm=self.total_dxi(xx,svm_weights)
            
            if (avg_svm_dxi >= 0.9*avg_dxi) and  (avg_svm_dxi <= 1.1*avg_dxi) :
                print(11)
                xx=xx.iloc[:,0:-1]
                svm_weights=svm_weights[:-1]
            else:
                break
        column_update=list(xx.columns)
        weight_svm=[]
        for i in range(len(column_name)):
            
            if column_name[i] in column_update :
                a=column_update.index(column_name[i])
                weight_svm.append(w_svm[a])
            else:
                weight_svm.append(0)
        return svm_dxi,avg_svm_dxi,w_svm,weight_svm

    def composite_dxi(self,x,weight_svm,weight_pca,weight_mi,weight_lasso,weigth_xgb):
        final_weights=[]
        for i in range(x.shape[1]):
            w= [weight_svm[i],weight_pca[i],weight_mi[i],weight_lasso[i],weigth_xgb[i]]      
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1

            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            
            final_weights.append(final_weight)
            
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))
        print(composite_avg_dxi)   
        return composite_dxi, composite_avg_dxi

    def sample_composite_dxi(self,x,weight_svm,weight_mi,weight_lasso,weight_pca,weight_nb):
    
        final_weights=[]
        for i in range(x.shape[1]):
            
            w= [weight_svm[i],weight_mi[i],weight_lasso[i],weight_pca[i],weight_nb[i]]
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1
            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            final_weights.append(final_weight)
        print(f'length of FInal Weight: {len(final_weights)}')
        print(x.columns)
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))    
        print(composite_avg_dxi) 
        fnlwgts = dict(zip(list(x.columns),final_weights))  
        return composite_dxi, composite_avg_dxi,fnlwgts

    def sample_composite_dxireg(self,x,weight_svm,weight_mi,weight_pca,weight_nb):
    
        final_weights=[]
        for i in range(x.shape[1]):
            
            w= [weight_svm[i],weight_mi[i],weight_pca[i],weight_nb[i]]
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1
            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            final_weights.append(final_weight)
        print(f'length of FInal Weight: {len(final_weights)}')
        print(x.columns)
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))    
        print(composite_avg_dxi) 
        fnlwgts = dict(zip(list(x.columns),final_weights))  
        return composite_dxi, composite_avg_dxi,fnlwgts


    def rflsample_composite_dxi(self,x,weight_svm,weight_mi,weight_lasso,weight_pca,weight_nb,weight_rfl):

        final_weights=[]
        for i in range(x.shape[1]):
            
            w= [weight_svm[i],weight_mi[i],weight_lasso[i],weight_pca[i],weight_nb[i],weight_rfl[i]]
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1

            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            final_weights.append(final_weight)
        print(f'length of FInal Weight: {len(final_weights)}')
            
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))
        print('SXI: ',composite_avg_dxi)   
        fnlwgts = dict(zip(list(x.columns),final_weights))
        return composite_dxi, composite_avg_dxi,fnlwgts

    def tensorflownn(self,X, y,feature_importance):

        from sklearn.model_selection import train_test_split
        import copy

        from keras.layers import Dense, Dropout,Dense
        from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
        from sklearn.metrics import make_scorer, accuracy_score
        from bayes_opt import BayesianOptimization
        from sklearn.model_selection import StratifiedKFold
        from keras.layers import LeakyReLU,PReLU
        from keras.initializers import Constant
        from scikeras.wrappers import KerasClassifier
        from tensorflow.keras import initializers
        from tensorflow.keras.initializers import Initializer
        from sklearn.model_selection import cross_val_score
        LeakyReLU = LeakyReLU(alpha=0.1)
        import warnings
        warnings.filterwarnings('ignore')
        pd.set_option("display.max_columns", None)
        from keras.models import Sequential
        from bayes_opt import BayesianOptimization
        
        class CustomInitializer(Initializer):
            def __init__(self, feature_importance):
                self.feature_importance = feature_importance
                print(feature_importance)

            def __call__(self, shape, dtype=None):
                input_size, output_size = shape
                if len(self.feature_importance) != input_size:
                    raise ValueError("Length of feature_importance should match the input_size")

                effective_input_size = int(np.sum(np.square(self.feature_importance)) * input_size)
                std_dev = np.sqrt(2.0 / (effective_input_size + output_size))
                weights = np.random.normal(0, std_dev, size=(input_size, output_size))
                return weights.astype(float32)

        X_train, X_val, y_train, y_val = train_test_split(X , y, test_size=0.2, random_state=42, stratify=y)
        input_size = X_train.shape[1]
        output_size = X_train.shape[1]
        def score_acc(y_true, y_pred):
            return accuracy_score(y_true, (y_pred > 0.5).astype(int))

        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu', LeakyReLU,
                    PReLU(alpha_initializer=Constant(value=0.25))]

        def nn_cl_bo(neurons, activation, optimizer, learning_rate, batch_size, epochs, dropout, dropout_rate):
            activation = activationL[int(round(activation))]
            neurons = round(neurons)
            batch_size = round(batch_size)
            epochs = round(epochs)
            learning_rate = learning_rate

            optimizerD = {
                'Adam': Adam(learning_rate=learning_rate),
                'SGD': SGD(learning_rate=learning_rate),
                'RMSprop': RMSprop(learning_rate=learning_rate),
                'Adadelta': Adadelta(learning_rate=learning_rate),
                'Adagrad': Adagrad(learning_rate=learning_rate),
                'Adamax': Adamax(learning_rate=learning_rate),
                'Nadam': Nadam(learning_rate=learning_rate),
                'Ftrl': Ftrl(learning_rate=learning_rate)
            }

            def nn_cl_fun():
                try:
                    opt = optimizerD.get(optimizer, Adam)(learning_rate=learning_rate)
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(f"Optimizer value: {optimizer}")
                    print(f"Available optimizers: {optimizerD.keys()}")
                    raise

                input_dim = X_train.shape[1]

                nn = Sequential()
                nn.add(Dense(neurons, input_dim=input_dim, activation=activation))

                if  X_train.shape[0] > 20000:
                    nn.add(Dense(neurons,  activation=activation))
                else:
                    nn.add(Dense(neurons, kernel_initializer=initializers.GlorotNormal(seed=None), activation=activation))
                
                if dropout > 0.5 and X_train.shape[0] < 20000:
                    nn.add(Dropout(dropout_rate, seed=123))

                if  X_train.shape[0] > 20000:
                    nn.add(Dense(neurons,  activation=activation))
                else:
                    nn.add(Dense(neurons, kernel_initializer=initializers.GlorotNormal(seed=None), activation=activation))
                
                if dropout > 0.5 and X_train.shape[0] < 20000:
                    nn.add(Dropout(dropout_rate, seed=123))

                if  X_train.shape[0] > 20000:
                    nn.add(Dense(neurons,  activation=activation))
                else:
                    nn.add(Dense(neurons, kernel_initializer=initializers.GlorotNormal(seed=None), activation=activation))

                nn.add(Dense(1, activation='sigmoid'))

                nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

                return nn

            nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
            score = cross_val_score(nn, X_train, y_train, scoring=make_scorer(score_acc), cv=kfold).mean()

            return score

        params_nn = {
            'neurons': (30, 200),
            'activation': (0, 8),
            'optimizer': (0, 8),
            'learning_rate': (0.01, 1),
            'batch_size': (50, 120),
            'epochs': (20, 100),
            'dropout_rate': (0, 0.3),
            'dropout': (0, 1)
        }

        nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=111)

        nn_bo.maximize(init_points=3, n_iter=2)
        
        print('---------------------------Params--------------------------')
        params_nn_ = nn_bo.max['params']

        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                    'elu', LeakyReLU, PReLU(alpha_initializer=Constant(value=0.25))] 

        params_nn_['activation'] = activationL[round(params_nn_['activation'])]

        learning_rate = params_nn_['learning_rate']

        optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
                    'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
                    'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
                    'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}

        optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']

        params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]

        params_nn_['batch_size'] = round(params_nn_['batch_size'])
        params_nn_['epochs'] = round(params_nn_['epochs'])
        params_nn_['neurons'] = round(params_nn_['neurons'])

        print('---------------------Layers--------------------------\n')

        def print_layer_weights(model):
            first_layer = model.layers[0]
            first_layer_weights = first_layer.get_weights()
            print(f"First Layer weights:")
            if len(model.layers) < 2:
                print("Model does not have enough layers.")
                return

            second_layer = model.layers[1]
            second_layer_weights = second_layer.get_weights()

            print(f"Second Layer weights:")
            return first_layer_weights, second_layer_weights

        print('-----------------------Training----------------------------\n')

        def nn_cl_fun(X_train, y_train, X_val, y_val):
            input_dim = X_train.shape[1]
            nn = Sequential()

            nn.add(Dense(params_nn_['neurons'], input_dim=input_dim, activation= LeakyReLU,kernel_initializer= CustomInitializer(feature_importance)))

            if X_train.shape[0] > 20000:
                nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
            else:
                nn.add(Dense(params_nn_['neurons'],kernel_initializer= initializers.GlorotNormal(seed=None), activation=params_nn_['activation']))

            if params_nn_['dropout'] > 0.5 and X_train.shape[0] < 20000:
                nn.add(Dropout(params_nn_['dropout_rate'], seed=123))

            if X_train.shape[0] > 20000:
                nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
            else:
                nn.add(Dense(params_nn_['neurons'],kernel_initializer=initializers.GlorotNormal(seed=None), activation=params_nn_['activation']))

            if params_nn_['dropout'] > 0.5 and X_train.shape[0] < 20000:
                nn.add(Dropout(params_nn_['dropout_rate'], seed=123))

            if X_train.shape[0] > 20000:
                nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
            else:
                nn.add(Dense(params_nn_['neurons'],kernel_initializer=initializers.GlorotNormal(seed=None), activation=params_nn_['activation']))

            nn.add(Dense(1, activation='sigmoid'))

            optimizerD = {
                'Adam': Adam,
                'SGD': SGD,
                'RMSprop': RMSprop,
                'Adadelta': Adadelta,
                'Adagrad': Adagrad,
                'Adamax': Adamax,
                'Nadam': Nadam,
                'Ftrl': Ftrl
            }

            try:
                opt = optimizerD.get(params_nn_['optimizer'], Adam)(learning_rate=params_nn_['learning_rate'])
            except KeyError as e:
                print(f"KeyError: {e}")
                print(f"Optimizer value: {params_nn_['optimizer']}")
                print(f"Available optimizers: {optimizerD.keys()}")
                raise

            nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

            
            nn.summary()

            history = nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params_nn_['epochs'],
                            batch_size=params_nn_['batch_size'],  verbose=1)

            _, accuracy = nn.evaluate(X_val, y_val)

            first_layer_weights, last_layer_weights = print_layer_weights(nn)

            return accuracy, first_layer_weights, last_layer_weights

        accuracy, first_layer_weights, last_layer_weights = nn_cl_fun(X_train, y_train, X_val, y_val)
        return accuracy, [list(j) for j in zip(*first_layer_weights[0])][:10]
    
    """
    def pytorchnn(self,X,y):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
    #     X = df.drop([tv,'EmployeeNumber'],axis=1)
    #     y = df[tv]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1)
        xtrain = torch.tensor(xtrain.values, dtype=torch.float32)
        xval = torch.tensor(xval.values, dtype=torch.float32)
        ytrain = torch.tensor(ytrain.values, dtype=torch.long)
        yval = torch.tensor(yval.values, dtype=torch.long)
        class ClassificationModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(ClassificationModel, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.layer2 = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                return x
        input_size = len(X.columns) # specify the number of features in your input
        num_classes = 2 # specify the number of classes in your classification task
        num_epochs = 100
        hidden_sizes = [32, 64, 128, 256,512]
        accs = []
        for hidden_size in hidden_sizes:
            modelval = ClassificationModel(input_size, hidden_size, num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(modelval.parameters(), lr=0.01)

            for epoch in range(num_epochs):
                # Forward pass
                outputs = modelval(xtrain)
                loss = criterion(outputs, ytrain)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            modelval.eval()  # set the model to evaluation mode
            with torch.no_grad():
                _, predicted = torch.max(modelval(xval), 1)

            # Calculate accuracy
            accuracy = (predicted == yval).sum().item() / yval.size(0)
            print('Accuracy on Validation data: {:.2f}%'.format(accuracy * 100))
            accs.append(accuracy * 100)
                    # Record or print the validation loss and other metrics
        print(accs)
        hidden_size = hidden_sizes[accs.index(max(accs))]
        model = ClassificationModel(input_size, hidden_size, num_classes)
        # CrossEntropyLoss for classification
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            _, predicted = torch.max(model(X_test), 1)

        # Calculate accuracy
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))
        weights_layer1 = model.layer1.weight.data
        return list(weights_layer1)[0:3],accuracy
    """

    def pytorchnn(self,X, y,feature_importance):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        class CustomInitializer:
            def __init__(self, feature_importance):
                self.feature_importance = feature_importance
                print(feature_importance)

            def __call__(self, tensor):
                input_size, output_size = tensor.shape
                input_size = len(X.columns)
                print(input_size)
                if len(self.feature_importance) != input_size:
                    raise ValueError("Length of feature_importance should match the input_size")

                effective_input_size = int(np.sum(np.square(self.feature_importance)) * input_size)
                std_dev = np.sqrt(2.0 / (effective_input_size + output_size))
                with torch.no_grad():
                    tensor.normal_(0, std_dev)

        # Assuming feature_importance is defined based on your data
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1)
        xtrain = torch.tensor(xtrain.values, dtype=torch.float32)
        xval = torch.tensor(xval.values, dtype=torch.float32)
        ytrain = torch.tensor(ytrain.values, dtype=torch.long)
        yval = torch.tensor(yval.values, dtype=torch.long)

        class ClassificationModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes, initializer):
                super(ClassificationModel, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.layer2 = nn.Linear(hidden_size, num_classes)
                initializer(self.layer1.weight)

            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                return x

        input_size = len(X.columns)  # specify the number of features in your input
        num_classes = 2  # specify the number of classes in your classification task
        num_epochs = 100
        hidden_sizes = [32, 64, 128, 256, 512]
        accs = []

        initializer = CustomInitializer(feature_importance)
        print(feature_importance)
        for hidden_size in hidden_sizes:
            modelval = ClassificationModel(input_size, hidden_size, num_classes, initializer)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(modelval.parameters(), lr=0.01)

            for epoch in range(num_epochs):
                # Forward pass
                outputs = modelval(xtrain)
                loss = criterion(outputs, ytrain)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
            
            modelval.eval()  # set the model to evaluation mode
            with torch.no_grad():
                _, predicted = torch.max(modelval(xval), 1)

            # Calculate accuracy
            accuracy = (predicted == yval).sum().item() / yval.size(0)
            print('Accuracy on Validation data: {:.2f}%'.format(accuracy * 100))
            accs.append(accuracy * 100)

        print(accs)
        hidden_size = hidden_sizes[accs.index(max(accs))]
        model = ClassificationModel(input_size, hidden_size, num_classes, initializer)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            _, predicted = torch.max(model(X_test), 1)

        # Calculate accuracy
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))
        weights_layer1 = model.layer1.weight.data
        return list(weights_layer1)[0:2], accuracy

    def xregenerate(self,dfa,tv,primkey):
        xa = dfa.drop([tv,primkey],axis=1)#dropped sku
        ya= dfa[tv]
        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=0.2)
        clf.fit(xa,ya)
        lasso_weight= clf.coef_
        #update dataframe
        min_max = []
        max_f=[]
        dff=xa.columns
        for i in range(len(dff)):
            if lasso_weight[i] < 0 :
                min_max.append(dff[i])
            else:
                max_f.append('MAX')
        
        dfa.drop([tv],axis=1,inplace=True)
        features=list(dfa.columns)

        minimum_value=[]
        maximum_value=[]
        minmax_value=[]
        catogorical_value=[]
        for i in range(len(features)):
        
            if features[i] in min_max:
                minmax_value.append('MIN')
                catogorical_value.append('Behavior')
            else:
                minmax_value.append('MAX') 
                catogorical_value.append('Behavior')
            try:
                minimum=min(list(dfa[features[i]]))
                maximum=max(list(dfa[features[i]]))
            except:
                minimum=''
                maximum=''
            minimum_value.append(minimum)
            maximum_value.append(maximum)
        minimum_value[0]='Min_Value'
        maximum_value[0]='Max_Value'
        minmax_value[0] = 'minmax'
        catogorical_value[0]='Category'
        
        df_mains=pd.DataFrame()
        df_mains['0']=minimum_value
        df_mains['1']=maximum_value
        df_mains['2']=minmax_value
        df_mains['3']=catogorical_value
        
        df_m=df_mains.T
        df_m.columns=dfa.columns
        df_m.reset_index(drop=True, inplace=True)
        dfa.reset_index(drop=True, inplace=True)
        frames=[df_m,dfa]

        results = pd.concat(frames, keys=['x', 'y'])
        print(results.shape)
        dfg = results.iloc[:, 1:]
        minmax=list(dfg.iloc[1])
        # print(minmax)
        column=dfg.columns
        # print(f'Column entering: {column}')
        dropcols = []
        for i in range(len(minmax)):
            if float(minmax[i]) == 0:    
                dfg=dfg.drop([column[i]],axis=1)
                dropcols.append(i)
                # print(dropcols)
        dfg=dfg.dropna(how='any')
        new_df=self.normalize(dfg)
        return new_df,dropcols

    def rfLagent(self,sxi,chkcls,tv,rfagent,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                xx,avg_base_dxi,r_start,r_ends,selcls,twds,classes,misclassif,intv,dropsind,primkey):

        dfb=[]
        wgtper = []
        classified = []
        sxplswgt = []
        brkloop = False
        for i in range(r_start,r_ends,intv):
            print(i)
            if i > r_start and intv == 5:
            
                if dfb[-1].empty:
                    finalmisclasif = dfb[-1]
                    break;
            
                newdff = dfb[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index','Datasetname'],axis=1,errors = 'ignore')
                xx,dropind=self.xregenerate(newdff,tv,primkey)
                print(xx.shape)
                misclassif = dfb[-1]
            
            elif i < r_start and intv == -5:
                if dfb[-1].empty:
                    finalmisclasif = dfb[-1]
                    break;
                newdff = dfb[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index','Datasetname'],axis=1,errors = 'ignore')
                xx,dropind=self.xregenerate(newdff,tv,primkey)
                misclassif = dfb[-1]
        
            else:
                dropind = dropsind
                pass          
            
            wgtper.append(i) # Stores the % increase
            rfgupd = map(lambda x: x+(x*(i/100)), rfagent) # Weights from the Neural network- with % increase/decrease (Reinforcement L)
            """
            laswg = list(map(add, lasso_weight, rfgupd)) # lasso weight + Weights(Reinforcement L)
            miwg = list(map(add, mi_weight, rfgupd)) # MI weight + Weights(Reinforcement L)
            pcawg = list(map(add, pca_weight, rfgupd))# PCA weight + Weights(Reinforcement L)
            nbwg = list(map(add, nb_weight, rfgupd)) # Naive Bayes weight + Weights(Reinforcement L)
            xgbwg = list(map(add, xgb_weight, rfgupd)) # XGBoost weight + Weights(Reinforcement L)
            """
            # print(len(lasso_weight))
            # print(len(mi_weight))
            # print(len(pca_weight))
            # print(len(xgb_weight))
            laswg = lasso_weight
            miwg = mi_weight
            pcawg = pca_weight
            nbwg = nb_weight
            xgbwg = xgb_weight
            rfagupd = list(rfgupd)
            # print(len(laswg))
            # print(len(miwg))
            # print(len(pcawg))
            # print(len(xgbwg))
            laswg = [value for i, value in enumerate(laswg) if i not in dropind]
            miwg = [value for i, value in enumerate(miwg) if i not in dropind]
            pcawg = [value for i, value in enumerate(pcawg) if i not in dropind]
            nbwg = [value for i, value in enumerate(nbwg) if i not in dropind]
            xgbwg = [value for i, value in enumerate(xgbwg) if i not in dropind]  
            rfagupd = [value for i, value in enumerate(rfagupd) if i not in dropind]
            xx = xx.reset_index(drop=True)

            ##############SXI Score calculation where the above weights taken as inputs###############
            lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = self.composite_weight(xx,laswg,avg_base_dxi)
            # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi)
            mi_dxi,avg_mi_dxi,w_mi,weight_mi = self.composite_weight(xx,miwg,avg_base_dxi)
            pca_dxi,avg_pca_dxi,w_pca,weight_pca = self.composite_weight(xx,pcawg,avg_base_dxi)
            nb_dxi,avg_nb_dxi,w_nb,weight_nb = self.composite_weight(xx,nbwg,avg_base_dxi)
            xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = self.composite_weight(xx,xgbwg,avg_base_dxi)
            
            rfl_dxi, avg_rfl_dxi,w_rfl,weight_rfl = self.composite_weight(xx,rfagupd,avg_base_dxi)
            
            sample_composite_dxi1=self.rflsample_composite_dxi(xx,weight_lasso, weight_xgb, weight_mi, weight_pca,weight_nb,weight_rfl)
            composite_dxi1=sample_composite_dxi1[0]
            nan_inds = [i for i, value in enumerate(composite_dxi1) if math.isnan(value)]
            if any(x == float('-inf') for x in composite_dxi1):
                if i == r_start:
                    brkloop = True
                    print('Entering SXI-NON')
                    finalmisclasif = misclassif
                    classified = []
                    wgtper = []
                    if len(dfb) > 0:
                        finalmisclasif = dfb[-1] 
                    else:
                        finalmisclasif = misclassif
                else:
                    pass
                continue    
    #             brkloop = True
    #             break;
            elif nan_inds:
                if i == r_start:
                    brkloop = True
                    print('Entering SXI-NON')
                    finalmisclasif = misclassif
                    classified = []
                    wgtper = []
                    if len(dfb) > 0:
                        finalmisclasif = dfb[-1] 
                    else:
                        finalmisclasif = misclassif
                else:
                    pass
                continue    
    #             brkloop = True
    #             break;
            else:
                sxipluswgts = sample_composite_dxi1[2] 
                sxplswgt.append(sxipluswgts)  
            #composite_dxi1=composite_dxi[0]
            avg_composite_dxi=np.mean(composite_dxi1)
            print('SXI: ',avg_composite_dxi)
            ##################SXI Calculation Done################################
            update_y_com=self.create_label_forward(composite_dxi1,avg_composite_dxi)  
            df_buynobuya = misclassif
            df_buynobuya = df_buynobuya.reset_index(drop=True)
    #         df_buynobuy=df_buynobuy.dropna(how='any')
            df_buynobuya['composite_dxi_label'] = update_y_com #Adding SXI scores to the dataframe
            df_buynobuya['composite_dxi'] = composite_dxi1
            
            if chkcls == 'class1' and ((selcls =='cls1' and twds =='abv') or (selcls =='cls2' and twds =='bel')):
                ckf=df_buynobuya.loc[(df_buynobuya['composite_dxi'] >= sxi) & (df_buynobuya[tv] == 1)]
            elif chkcls == 'class2' and ((selcls =='cls1' and twds =='abv') or (selcls =='cls2' and twds =='bel')):
                ckf = df_buynobuya.loc[(df_buynobuya['composite_dxi'] < sxi) & (df_buynobuya[tv] == 0)]
            elif chkcls == 'class1' and ((selcls =='cls1' and twds =='bel') or (selcls =='cls2' and twds =='abv')):
                ckf = df_buynobuya.loc[(df_buynobuya['composite_dxi'] < sxi) & (df_buynobuya[tv] == 1)]
            elif chkcls == 'class2' and ((selcls =='cls1' and twds =='bel') or (selcls =='cls2' and twds =='abv')):
                ckf = df_buynobuya.loc[(df_buynobuya['composite_dxi'] >= sxi) & (df_buynobuya[tv] == 0)]
            
            if chkcls == 'class1':    
                print(f'Classified Correctly {len(ckf)} out of {len(df_buynobuya)} for {classes[1]}')
            else:
                print(f'Classified Correctly {len(ckf)} out of {len(df_buynobuya)} for {classes[0]}')
            
            if len(ckf) == 0:
                dfb.append(df_buynobuya)
            else:
                classified.append(ckf)
                filtered_df = df_buynobuya[~df_buynobuya.index.isin(ckf.index)]
                dfb.append(filtered_df)
        if len(dfb) > 0:
            finalmisclasif = dfb[-1] 
        else:
            finalmisclasif = misclassif
        return classified,wgtper,finalmisclasif,sxplswgt

    def wrecalibrate(self,sxi,chkcls1,misclassif1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                    avg_base_dxi,selcls,twds,classes,primkey):
        xxs = misclassif1.drop(['composite_dxi_label','composite_dxi','index', 'needed_index','Datasetname'],axis=1,errors = 'ignore')
        xxs,dropsind=self.xregenerate(xxs,tv,primkey)
        print(xxs.shape)
        r_start,r_ends = 0,100
        inv=5
        clsfd,wgtpr,fnlmisclasif,sxplswgt = self.rfLagent(sxi,chkcls1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,
                                            xgb_weight,xxs,avg_base_dxi,r_start,r_ends,selcls,twds,classes,misclassif1,inv,dropsind,primkey)
        nan_indices = [i for i, value in enumerate(fnlmisclasif['composite_dxi'].values) if math.isnan(value)]
        if nan_indices:
            fnlmisclasif1 = [misclassif1]
        else:
            fnlmisclasif1 = [fnlmisclasif]
        clsfdfnl = [clsfd]
        
        trigbrk = False

        if len(clsfd) == 0:
            r_start,r_ends = 600,500
            # r_start,r_ends = 0,100
            while len(fnlmisclasif) != 0:
                if trigbrk:
                    print('Reinforcement Stopped')
                    break 
                else:
                    #Looping towards negative side - percentage decrease
                    r_start= r_start-100 
                    r_ends = r_ends -100 
                    print('Rstart: ', r_start)
                    print('Rends: ', r_ends)
                    if r_start < -600: #Loop breaks at -5000 % decrease if the loop goes on...
                        break
                    else:

                        xxsit = fnlmisclasif1[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index','Datasetname'],axis=1,errors = 'ignore')
                        if len(xxsit) == 0:
                            break
                        else:
                            pass
                        xxsit = xxsit.reset_index(drop=True)
                        xxsit,dropsind=self.xregenerate(xxsit,tv,primkey)
                        xxsit = xxsit.reset_index(drop=True)
                        clsfd1,wgtpr1,fnlmisclasif2,sxplswgt1 = self.rfLagent(sxi,chkcls1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,
                                            xgb_weight,xxsit,avg_base_dxi,r_start,r_ends,selcls,twds,classes,fnlmisclasif1[-1],-5,dropsind,primkey)
                        fnlmisclasif2 = fnlmisclasif2.reset_index(drop=True)
                        nans_indices = [i for i, value in enumerate(fnlmisclasif2['composite_dxi'].values) if math.isnan(value)]
                        if nans_indices:
                            break
                        else:
                            fnlmisclasif1.append(fnlmisclasif2)
                            clsfdfnl.append(clsfd1)
                            sxplswgt = sxplswgt+sxplswgt1
        else:
            r_start,r_ends = -600,-500
            while len(fnlmisclasif) != 0:
                if trigbrk:
                    print('Reinforcement Stopped')
                    break 
                else:
                    #Looping towards negative side - percentage decrease
                    r_start= r_start+100 
                    r_ends = r_ends + 100 
                    print('Rstart: ', r_start)
                    print('Rends: ', r_ends)
                    if r_start > 600: #Loop breaks at -5000 % decrease if the loop goes on...
                        break
                    else:

                        xxsit = fnlmisclasif1[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index','Datasetname'],axis=1,errors='ignore')
                        if len(xxsit) == 0:
                            break
                        else:
                            pass
                        xxsit = xxsit.reset_index(drop=True)
                        xxsit,dropsind=self.xregenerate(xxsit,tv,primkey)
                        xxsit = xxsit.reset_index(drop=True)
                        clsfd1,wgtpr1,fnlmisclasif2,sxplswgt2 = self.rfLagent(sxi,chkcls1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,
                                            xgb_weight,xxsit,avg_base_dxi,r_start,r_ends,selcls,twds,classes,fnlmisclasif1[-1],-5,dropsind,primkey)
                        nans_indices = [i for i, value in enumerate(fnlmisclasif2['composite_dxi'].values) if math.isnan(value)]
                        if nans_indices:
                            break
                        fnlmisclasif2 = fnlmisclasif2.reset_index(drop=True)
                        fnlmisclasif1.append(fnlmisclasif2)
                        clsfdfnl.append(clsfd1)
                        sxplswgt = sxplswgt+sxplswgt2
        
        return clsfdfnl,fnlmisclasif1[-1],sxplswgt

    def generate_sxirl(self,target,buynobuy,sxi_dataframe,classes,primkey):
        dropcols = []
        tv = target
        df_buynobuy = buynobuy ### Read BUYNOBUY DATA
        df = sxi_dataframe ### Read DXI DATA
        df = df.iloc[:, 1:]
        minmax=list(df.iloc[1])
        
        print(f'Minmax1: {minmax}')
        print(f'Len Minmax1: {len(minmax)}')
        column=df.columns

        for i in range(len(minmax)):
            if float(minmax[i]) == 0:
                df=df.drop([column[i]],axis=1)
                dropcols.append(column[i])
        df=df.dropna(how='any')
        new_df=self.normalize(df)
        b=self.bivarient_correlation(new_df) #not required
        w=self.weight(new_df,b)  #Weight 1 to save
        before_base_dxi=self.dxi(new_df,w)
        avg_dxi=np.mean(np.array(before_base_dxi))
        x=new_df

        y=self.create_label_forward(before_base_dxi,avg_dxi)
        clf = linear_model.Lasso(alpha=0.2)
        clf.fit(x,y)
        prelasso_weight= clf.coef_ # save lasso weight 1
        
        min_max= (df.iloc[2])

        print(f'Minmax2: {minmax}')
        print(f'Len Minmax2: {len(minmax)}')

        for i in range(len(min_max)):
            if prelasso_weight[i] < 0 :
                if min_max[i] != 'MIN' :
                    (df.loc[2])[i]='MIN'
        new_df=self.normalize(df)

        update_b=self.bivarient_correlation(new_df)  
        parameter=df.iloc[3]
        update_w=self.weight(new_df,update_b) #weight 2 to save
        base_dxi=self.dxi(new_df,update_w)
        behaviour_dxi,transactional_dxi,kpi_dxi,visual_dxi=self.catogorical_dxi(new_df,update_w,parameter)
        avg_behaviour=np.mean(np.array(behaviour_dxi))
        avg_kpi=np.mean(np.array(kpi_dxi))
        avg_transactional=np.mean(np.array(transactional_dxi))
        avg_visual=np.mean(np.array(visual_dxi))
        avg_base_dxi=np.mean(np.array(base_dxi))
        update_y = self.create_label_forward(base_dxi,avg_base_dxi)

        data=df.iloc[4:]
        data['Base_dxi_label']=update_y
        data['Base_dxi']=base_dxi
        data[tv]=df_buynobuy[tv]
        op = new_df
        print(f'Len of Cols: {len(new_df.columns)}')
        weightcols = new_df.columns
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(op)
        if (x.values < 0).any() == True:
            a = x_train
        else:
            a = x
        cnb = ComplementNB().fit(a, update_y)
        logprobs = cnb.feature_log_prob_
        avgprob =[]
        for i in range(len(logprobs[0])):
            avgprob.append((logprobs[0][i]+logprobs[1][i])/2)
        nb_weight = list(np.exp(avgprob) / (np.exp(avgprob)).sum())  

        xgb = xg.XGBClassifier().fit(new_df,update_y)
        xgb_weight = list(xgb.feature_importances_)  
        mi_weight=list(self.mutual_information(new_df, update_y))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit_transform(new_df)
        pca_weight = pca.components_[0]
        # pca_weight=list(self.pca_feature_selection(new_df,update_y))
        lasso_weight=list(self.lasso_feature_selection(new_df,update_y))

        lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = self.composite_weight(new_df,lasso_weight,avg_base_dxi) #save weight 3 - ML1  
        # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi) 
        mi_dxi,avg_mi_dxi,w_mi,weight_mi = self.composite_weight(new_df,mi_weight,avg_base_dxi) #save weight 4 - ML2
        pca_dxi,avg_pca_dxi,w_pca,weight_pca = self.composite_weight(new_df,pca_weight,avg_base_dxi)  #save weight 5 - ML3
        nb_dxi,avg_nb_dxi,w_nb,weight_nb = self.composite_weight(new_df,nb_weight,avg_base_dxi)  #save weight 6 - ML4
        xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = self.composite_weight(new_df,xgb_weight,avg_base_dxi)  #save weight 7 - ML5

        top_names =[]
        for weight in [weight_lasso,weight_nb,weight_mi, weight_pca,weight_xgb]:   
            idx = (-(np.array(weight))).argsort()[:len(weight)+1]
            names = x.columns[idx]
            top_names.append(names)

        tp_params = {}
        tp_params["Algorithm1"]=top_names[0][0:5] 
        tp_params["Algorithm2"]=top_names[1][0:5]
        tp_params["Algorithm3"]=top_names[2][0:5]
        tp_params["Algorithm4"]=top_names[3][0:5]
        tp_params["Algorithm5"]=top_names[4][0:5]

        sample_composite_dxi1=self.sample_composite_dxi(new_df,weight_lasso, weight_xgb, weight_mi, weight_pca,weight_nb)
        composite_dxi1=sample_composite_dxi1[0]
        avg_composite_dxi=np.mean(composite_dxi1)
        update_y_com=self.create_label_forward(composite_dxi1,avg_composite_dxi)
        initsxiwgts = sample_composite_dxi1[2]
        topparms = pd.DataFrame(tp_params)
        value_counts = topparms.melt(value_name='value').value.value_counts()
        repeated_values = value_counts[value_counts > 1].index
        vallist = topparms.values.flatten().tolist()
        freqvals = [vallist.count(i) for i in repeated_values]
        most_common_features_df = pd.DataFrame(repeated_values, columns=['Most Common Features'])
        most_common_features_df['No.of Times'] = freqvals
        print(most_common_features_df)
        hcols = list(x.columns)
        impind = []
        for i in repeated_values:
            impind.append(hcols.index(i))
        featimp = [1] * len(hcols) 
        for i in range(len(impind)):
            featimp[impind[i]] = 1 + (freqvals[i]/len(impind))

        sxi = avg_composite_dxi
        # fdf = pd.DataFrame()
        # finalsxiweights = []
        # selcls = '' 
        # twds = ''

        # return fdf,sxi,finalsxiweights,initsxiwgts,selcls,twds,prelasso_weight,w,update_w,weight_lasso,weight_mi,weight_pca,weight_nb,weight_xgb
        
        # acctensor,rfagenttensor = self.tensorflownn(x,pd.Series(y),featimp)
        rfagenttorch,acctorch = self.pytorchnn(new_df,pd.Series(update_y),featimp) #Final Weights
        acctensor = 0
        if acctensor > acctorch:
            rfagents = rfagenttensor
        else:
            rfagents = rfagenttorch
            rfagents = [tensor.tolist() for tensor in rfagents]
            rfagents = rfagents[:3] ### Change weights accordingly large data (6) smaller (3)
        # list(rfagents)
        df_buynobuy=df_buynobuy.dropna(how='any')
        df_buynobuy['composite_dxi_label'] = update_y_com
        df_buynobuy['composite_dxi'] = composite_dxi1
        df_buynobuy.to_csv('IOT_initial_sxi.csv',index=False)
        print('File Saved')
        

        initialsxidata = df_buynobuy
        dropfeatsre = df_buynobuy[dropcols]
        df_buynobuy = df_buynobuy.drop(dropcols,axis=1)
        clas1 = (df_buynobuy[tv].value_counts()[1]/len(df_buynobuy))*100  
        clas2 = (df_buynobuy[tv].value_counts()[0]/len(df_buynobuy))*100

        f1init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] >= df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 1)]
        f2init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] >= df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 0)]
        f3init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] < df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 1)]
        f4init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] < df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 0)]

        if clas1 > clas2:
            jabv = (len(f2init)/(len(f1init) +len(f2init)))*100
            jbel = (len(f4init)/(len(f3init) +len(f4init)))*100
            jlst = [jabv,jbel]
            j = max(jlst)
            if jlst.index(j) == 0:
                twds = 'abv' 
            else:
                twds = 'bel'
            minwhcls = clas2 
            selcls = 'cls2' 
        else:
            jabv = (len(f1init)/(len(f1init) +len(f2init)))*100
            jbel = (len(f3init)/(len(f3init) +len(f4init)))*100
            jlst = [jabv,jbel]
            j = max(jlst)
            if jlst.index(j) == 0:
                twds = 'abv'
            else:
                twds = 'bel' 
            minwhcls = clas1 
            selcls = 'cls1'  

        classifydt = []
        if selcls == 'cls1' and twds == 'abv':
            print(selcls+twds)
            classifydt.append(f1init)
            classifydt.append(f4init)
            misclassif1=f3init
            misclassif2=f2init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
            
        elif selcls == 'cls1' and twds == 'bel':
            print(selcls+twds)
            classifydt.append(f3init)
            classifydt.append(f2init)
            misclassif1=f1init
            misclassif2=f4init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        elif selcls == 'cls2' and twds == 'abv':
            print(selcls+twds)
            classifydt.append(f2init)
            classifydt.append(f3init)
            misclassif1=f1init
            misclassif2=f4init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        else:
            print(selcls+twds)
            classifydt.append(f1init)
            classifydt.append(f4init)
            misclassif1=f3init
            misclassif2=f2init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        orgmsclasif = len(misclassif1) + len(misclassif2)
        print(f'Original Misclassifed: {orgmsclasif}')
        clsfdfnl2 = [] 
        fnlmisclasifff2 = []
        clsfdfnl1 = []
        fnlmisclasifff1 = []
        lenmisclassif = []
        sxiplusmjwgts1, sxiplusmjwgts2= [],[]
        if len(misclassif2.columns) > 0:
            setcols = misclassif2.columns
        else:
            setcols = misclassif1.columns
        if orgmsclasif != 0:
        
            for wgs in rfagents:
                if len(misclassif2) > 0:
                    clsfdfnl_2,fnlmisclasif_2,fnlsxplswgt1 = self.wrecalibrate(sxi,'class2',misclassif2,tv,wgs,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                        avg_base_dxi,selcls,twds,classes,primkey)
                    clsfdfnl2.append(clsfdfnl_2)
                    print(f'Type of clsfdfnl2 in wgs: {type(clsfdfnl_2)}')
                    print(f'Type of fnlmisclasif_2 in wgs: {type(fnlmisclasif_2)}')
                    fnlmisclasifff2.append(fnlmisclasif_2)
                    sxiplusmjwgts1.append(fnlsxplswgt1)
                else:
                    print('Entering Else Misclassif2')
                    fnlmisclasif_2 = []
                    # clsfdfnl2.append([pd.DataFrame(columns=setcols)])
                    clsfdfnl2.append([])
                    fnlmisclasifff2.append(pd.DataFrame(columns=setcols))
                    sxiplusmjwgts1.append([])

                if len(misclassif1) > 0:  
                    clsfdfnl_1,fnlmisclasif_1,fnlsxplswgt2 = self.wrecalibrate(sxi,'class1',misclassif1,tv,wgs,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                    avg_base_dxi,selcls,twds,classes,primkey)
                    clsfdfnl1.append(clsfdfnl_1)
                    fnlmisclasifff1.append(fnlmisclasif_1)
                    sxiplusmjwgts2.append(fnlsxplswgt2)
                else:
                    print('Entering Else Misclassif1')     
                    fnlmisclasif_1 = []
                    fnlmisclasifff1.append(pd.DataFrame(columns=setcols))
                    # clsfdfnl1.append([pd.DataFrame(columns=setcols)])
                    clsfdfnl1.append([])
                    sxiplusmjwgts2.append([])
                msclass = len(fnlmisclasif_2) + len(fnlmisclasif_1)
                lenmisclassif.append(msclass)
                if msclass == 0:
                    break
                else:
                    pass
            # print(f'lenmisclassif: {lenmisclassif}')
            # print(f'fnlmisclasifff1: {len(fnlmisclasifff1)}')
            # print(f'fnlmisclasifff2: {len(fnlmisclasifff2)}')
            # print(fnlmisclasifff2)
            # print(fnlmisclasifff1)
            fnlmdlind = lenmisclassif.index(min(lenmisclassif))
            print(f'fnlmdlind: {fnlmdlind}')
            lists,list2,finldata=[],[],[]
            print(fnlmisclasifff1[fnlmdlind]['index'])
            finalsxiweights = sxiplusmjwgts1[fnlmdlind] + sxiplusmjwgts2[fnlmdlind]
            
            for i in fnlmisclasifff1[fnlmdlind]['index']:
                lists.append(i)
            result_misclassif1 = misclassif1[misclassif1['index'].isin(lists)]
            for i in fnlmisclasifff2[fnlmdlind]['index']:
                list2.append(i)
            
            result_misclassif2 = misclassif2[misclassif2['index'].isin(list2)]
            
            print(f'clsfdfnl1: {clsfdfnl1[fnlmdlind]}')
            print(f'clsfdfnl1 Type: {type(clsfdfnl1[fnlmdlind])}')
            for i in clsfdfnl1[fnlmdlind]:
                for j in i:
                    finldata.append(j)
            
            print('\n')
            print('\n')
            # print(f'clsfdfnl2: {clsfdfnl2[fnlmdlind]}')
            # print(f'clsfdfnl2 Type: {type(clsfdfnl2[fnlmdlind])}')


            for k in clsfdfnl2[fnlmdlind]:
                for o in k:
                    finldata.append(o)
            
            print('\n')
            print('\n')
            print(f'finldata: {finldata}')
            fnlmsclasif = len(result_misclassif1) + len(result_misclassif2)
            if fnlmsclasif == orgmsclasif:
                fdf = pd.DataFrame(columns=result_misclassif1.columns)
            else:
                dt1 = pd.concat(finldata)
                dt2 = pd.concat(classifydt)
                fdf = pd.concat([dt1,dt2,result_misclassif1,result_misclassif2])
        else:
            fdf = df_buynobuy
            finalsxiweights = initsxiwgts
    
   
        return fdf,sxi,finalsxiweights,initsxiwgts,selcls,twds,prelasso_weight,w,update_w,weight_lasso,weight_mi,weight_pca,weight_nb,weight_xgb,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,rfagents,weightcols
        




    def generate_sxirlblind(self,target,buynobuy,sxi_dataframe,classes,primkey,prelasso_weight,w,update_w,weight_lasso,weight_mi,weight_pca,weight_nb,weight_xgb,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,rfagents):
        dropcols = []
        tv = target
        df_buynobuy = buynobuy ### Read BUYNOBUY DATA
        df = sxi_dataframe ### Read DXI DATA
        df = df.iloc[:, 1:]
        minmax=list(df.iloc[1])
        
        print(f'Minmax1: {minmax}')
        print(f'Len Minmax1: {len(minmax)}')
        column=df.columns

        for i in range(len(minmax)):
            if float(minmax[i]) == 0:
                df.at[1,column[i]] = 1
                # dropcols.append(column[i])
        df=df.dropna(how='any')
        new_df=self.normalize(df)
        
        # b=self.bivarient_correlation(new_df) #not required
        # w=self.weight(new_df,b)  #Weight 1 to save
        before_base_dxi=self.dxi(new_df,w)
        avg_dxi=np.mean(np.array(before_base_dxi))
        x=new_df
        print(f'Average BASE SXI1: {avg_dxi}')
        y=self.create_label_forward(before_base_dxi,avg_dxi)
        # clf = linear_model.Lasso(alpha=0.2)
        # clf.fit(x,y)
        # prelasso_weight= clf.coef_ # save lasso weight 1
        
        min_max= (df.iloc[2])

        print(f'Minmax2: {minmax}')
        print(f'Len Minmax2: {len(minmax)}')

        for i in range(len(min_max)):
            if prelasso_weight[i] < 0 :
                if min_max[i] != 'MIN' :
                    (df.loc[2])[i]='MIN'
        new_df=self.normalize(df)

        # update_b=self.bivarient_correlation(new_df)  
        parameter=df.iloc[3]
        # update_w=self.weight(new_df,update_b) #weight 2 to save
        base_dxi=self.dxi(new_df,update_w)
        behaviour_dxi,transactional_dxi,kpi_dxi,visual_dxi=self.catogorical_dxi(new_df,update_w,parameter)
        avg_behaviour=np.mean(np.array(behaviour_dxi))
        avg_kpi=np.mean(np.array(kpi_dxi))
        avg_transactional=np.mean(np.array(transactional_dxi))
        avg_visual=np.mean(np.array(visual_dxi))
        avg_base_dxi=np.mean(np.array(base_dxi))
        update_y = self.create_label_forward(base_dxi,avg_base_dxi)
        print(f'Average BASE SXI2: {avg_base_dxi}')
        data=df.iloc[4:]
        data['Base_dxi_label']=update_y
        data['Base_dxi']=base_dxi
        data[tv]=df_buynobuy[tv]
        op = new_df
        print(f'Len of Cols: {len(new_df.columns)}')
        
        # scaler = MinMaxScaler()
        # x_train = scaler.fit_transform(op)
        # if (x.values < 0).any() == True:
        #     a = x_train
        # else:
        #     a = x
        # cnb = ComplementNB().fit(a, update_y)
        # logprobs = cnb.feature_log_prob_
        # avgprob =[]
        # for i in range(len(logprobs[0])):
        #     avgprob.append((logprobs[0][i]+logprobs[1][i])/2)
        # nb_weight = list(np.exp(avgprob) / (np.exp(avgprob)).sum())  

        # xgb = xg.XGBClassifier().fit(new_df,update_y)
        # xgb_weight = list(xgb.feature_importances_)  
        # mi_weight=list(self.mutual_information(new_df, update_y))
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=3)
        # pca.fit_transform(new_df)
        # pca_weight = pca.components_[0]
        # # pca_weight=list(self.pca_feature_selection(new_df,update_y))
        # lasso_weight=list(self.lasso_feature_selection(new_df,update_y))

        # lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = self.composite_weight(new_df,lasso_weight,avg_base_dxi) #save weight 3 - ML1  
        # # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi) 
        # mi_dxi,avg_mi_dxi,w_mi,weight_mi = self.composite_weight(new_df,mi_weight,avg_base_dxi) #save weight 4 - ML2
        # pca_dxi,avg_pca_dxi,w_pca,weight_pca = self.composite_weight(new_df,pca_weight,avg_base_dxi)  #save weight 5 - ML3
        # nb_dxi,avg_nb_dxi,w_nb,weight_nb = self.composite_weight(new_df,nb_weight,avg_base_dxi)  #save weight 6 - ML4
        # xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = self.composite_weight(new_df,xgb_weight,avg_base_dxi)  #save weight 7 - ML5

        top_names =[]
        for weight in [weight_lasso,weight_nb,weight_mi, weight_pca,weight_xgb]:   
            idx = (-(np.array(weight))).argsort()[:len(weight)+1]
            names = x.columns[idx]
            top_names.append(names)

        tp_params = {}
        tp_params["Algorithm1"]=top_names[0][0:5] 
        tp_params["Algorithm2"]=top_names[1][0:5]
        tp_params["Algorithm3"]=top_names[2][0:5]
        tp_params["Algorithm4"]=top_names[3][0:5]
        tp_params["Algorithm5"]=top_names[4][0:5]

        sample_composite_dxi1=self.sample_composite_dxi(new_df,weight_lasso, weight_xgb, weight_mi, weight_pca,weight_nb)
        composite_dxi1=sample_composite_dxi1[0]
        avg_composite_dxi=np.mean(composite_dxi1)
        update_y_com=self.create_label_forward(composite_dxi1,avg_composite_dxi)
        initsxiwgts = sample_composite_dxi1[2]
        topparms = pd.DataFrame(tp_params)
        value_counts = topparms.melt(value_name='value').value.value_counts()
        repeated_values = value_counts[value_counts > 1].index
        vallist = topparms.values.flatten().tolist()
        freqvals = [vallist.count(i) for i in repeated_values]
        most_common_features_df = pd.DataFrame(repeated_values, columns=['Most Common Features'])
        most_common_features_df['No.of Times'] = freqvals
        print(most_common_features_df)
        hcols = list(x.columns)
        impind = []
        for i in repeated_values:
            impind.append(hcols.index(i))
        featimp = [1] * len(hcols) 
        for i in range(len(impind)):
            featimp[impind[i]] = 1 + (freqvals[i]/len(impind))

        sxi = avg_composite_dxi
        # fdf = pd.DataFrame()
        # finalsxiweights = []
        # selcls = '' 
        # twds = ''

        # return fdf,sxi,finalsxiweights,initsxiwgts,selcls,twds,prelasso_weight,w,update_w,weight_lasso,weight_mi,weight_pca,weight_nb,weight_xgb
        
        # acctensor,rfagenttensor = self.tensorflownn(x,pd.Series(y),featimp)
        # rfagenttorch,acctorch = self.pytorchnn(new_df,pd.Series(update_y),featimp) #Final Weights
        # acctensor = 0
        # if acctensor > acctorch:
        #     rfagents = rfagenttensor
        # else:
        #     rfagents = rfagenttorch
        #     rfagents = [tensor.tolist() for tensor in rfagents]
        #     rfagents = rfagents[:3] ### Change weights accordingly large data (6) smaller (3)
        # list(rfagents)
        df_buynobuy=df_buynobuy.dropna(how='any')
        df_buynobuy['composite_dxi_label'] = update_y_com
        df_buynobuy['composite_dxi'] = composite_dxi1
        df_buynobuy.to_csv('IOT.csv',index=False)
        print('File Saved')
        initialsxidata = df_buynobuy
        dropfeatsre = df_buynobuy[dropcols]
        # df_buynobuy = df_buynobuy.drop(dropcols,axis=1)
        clas1 = (df_buynobuy[tv].value_counts()[1]/len(df_buynobuy))*100  
        clas2 = (df_buynobuy[tv].value_counts()[0]/len(df_buynobuy))*100

        f1init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] >= df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 1)]
        f2init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] >= df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 0)]
        f3init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] < df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 1)]
        f4init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] < df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 0)]

        if clas1 > clas2:
            jabv = (len(f2init)/(len(f1init) +len(f2init)))*100
            jbel = (len(f4init)/(len(f3init) +len(f4init)))*100
            jlst = [jabv,jbel]
            j = max(jlst)
            if jlst.index(j) == 0:
                twds = 'abv' 
            else:
                twds = 'bel'
            minwhcls = clas2 
            selcls = 'cls2' 
        else:
            jabv = (len(f1init)/(len(f1init) +len(f2init)))*100
            jbel = (len(f3init)/(len(f3init) +len(f4init)))*100
            jlst = [jabv,jbel]
            j = max(jlst)
            if jlst.index(j) == 0:
                twds = 'abv'
            else:
                twds = 'bel' 
            minwhcls = clas1 
            selcls = 'cls1'  

        classifydt = []
        if selcls == 'cls1' and twds == 'abv':
            print(selcls+twds)
            classifydt.append(f1init)
            classifydt.append(f4init)
            misclassif1=f3init
            misclassif2=f2init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
            
        elif selcls == 'cls1' and twds == 'bel':
            print(selcls+twds)
            classifydt.append(f3init)
            classifydt.append(f2init)
            misclassif1=f1init
            misclassif2=f4init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        elif selcls == 'cls2' and twds == 'abv':
            print(selcls+twds)
            classifydt.append(f2init)
            classifydt.append(f3init)
            misclassif1=f1init
            misclassif2=f4init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        else:
            print(selcls+twds)
            classifydt.append(f1init)
            classifydt.append(f4init)
            misclassif1=f3init
            misclassif2=f2init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        orgmsclasif = len(misclassif1) + len(misclassif2)
        print(f'Original Misclassifed: {orgmsclasif}')
        clsfdfnl2 = [] 
        fnlmisclasifff2 = []
        clsfdfnl1 = []
        fnlmisclasifff1 = []
        lenmisclassif = []
        sxiplusmjwgts1, sxiplusmjwgts2= [],[]
        if len(misclassif2.columns) > 0:
            setcols = misclassif2.columns
        else:
            setcols = misclassif1.columns
        if orgmsclasif != 0:
        
            for wgs in rfagents:
                if len(misclassif2) > 0:
                    clsfdfnl_2,fnlmisclasif_2,fnlsxplswgt1 = self.wrecalibrate(sxi,'class2',misclassif2,tv,wgs,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                        avg_base_dxi,selcls,twds,classes,primkey)
                    clsfdfnl2.append(clsfdfnl_2)
                    print(f'Type of clsfdfnl2 in wgs: {type(clsfdfnl_2)}')
                    print(f'Type of fnlmisclasif_2 in wgs: {type(fnlmisclasif_2)}')
                    fnlmisclasifff2.append(fnlmisclasif_2)
                    sxiplusmjwgts1.append(fnlsxplswgt1)
                else:
                    print('Entering Else Misclassif2')
                    fnlmisclasif_2 = []
                    # clsfdfnl2.append([pd.DataFrame(columns=setcols)])
                    clsfdfnl2.append([])
                    fnlmisclasifff2.append(pd.DataFrame(columns=setcols))
                    sxiplusmjwgts1.append([])

                if len(misclassif1) > 0:  
                    clsfdfnl_1,fnlmisclasif_1,fnlsxplswgt2 = self.wrecalibrate(sxi,'class1',misclassif1,tv,wgs,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                    avg_base_dxi,selcls,twds,classes,primkey)
                    clsfdfnl1.append(clsfdfnl_1)
                    fnlmisclasifff1.append(fnlmisclasif_1)
                    sxiplusmjwgts2.append(fnlsxplswgt2)
                else:
                    print('Entering Else Misclassif1')     
                    fnlmisclasif_1 = []
                    fnlmisclasifff1.append(pd.DataFrame(columns=setcols))
                    # clsfdfnl1.append([pd.DataFrame(columns=setcols)])
                    clsfdfnl1.append([])
                    sxiplusmjwgts2.append([])
                msclass = len(fnlmisclasif_2) + len(fnlmisclasif_1)
                lenmisclassif.append(msclass)
                if msclass == 0:
                    break
                else:
                    pass
            # print(f'lenmisclassif: {lenmisclassif}')
            # print(f'fnlmisclasifff1: {len(fnlmisclasifff1)}')
            # print(f'fnlmisclasifff2: {len(fnlmisclasifff2)}')
            # print(fnlmisclasifff2)
            # print(fnlmisclasifff1)
            fnlmdlind = lenmisclassif.index(min(lenmisclassif))
            print(f'fnlmdlind: {fnlmdlind}')
            lists,list2,finldata=[],[],[]
            print(fnlmisclasifff1[fnlmdlind]['index'])
            finalsxiweights = sxiplusmjwgts1[fnlmdlind] + sxiplusmjwgts2[fnlmdlind]
            
            for i in fnlmisclasifff1[fnlmdlind]['index']:
                lists.append(i)
            result_misclassif1 = misclassif1[misclassif1['index'].isin(lists)]
            for i in fnlmisclasifff2[fnlmdlind]['index']:
                list2.append(i)
            
            result_misclassif2 = misclassif2[misclassif2['index'].isin(list2)]
            
            print(f'clsfdfnl1: {clsfdfnl1[fnlmdlind]}')
            print(f'clsfdfnl1 Type: {type(clsfdfnl1[fnlmdlind])}')
            for i in clsfdfnl1[fnlmdlind]:
                for j in i:
                    finldata.append(j)
            
            print('\n')
            print('\n')
            # print(f'clsfdfnl2: {clsfdfnl2[fnlmdlind]}')
            # print(f'clsfdfnl2 Type: {type(clsfdfnl2[fnlmdlind])}')


            for k in clsfdfnl2[fnlmdlind]:
                for o in k:
                    finldata.append(o)
            
            print('\n')
            print('\n')
            print(f'finldata: {finldata}')
            fnlmsclasif = len(result_misclassif1) + len(result_misclassif2)
            if fnlmsclasif == orgmsclasif:
                fdf = pd.DataFrame(columns=result_misclassif1.columns)
            else:
                dt1 = pd.concat(finldata)
                dt2 = pd.concat(classifydt)
                fdf = pd.concat([dt1,dt2,result_misclassif1,result_misclassif2])
        else:
            fdf = df_buynobuy
            finalsxiweights = initsxiwgts
    
        return fdf,sxi,finalsxiweights,initsxiwgts,selcls,twds
        




    
    def generate_sxiTS(self,target,buynobuy,sxi_dataframe,classes,primkey):
        dropcols = []
        tv = target
        df_buynobuy = buynobuy ### Read BUYNOBUY DATA
        df = sxi_dataframe ### Read DXI DATA
        df = df.iloc[:, 1:]
        minmax=list(df.iloc[1])
        column=df.columns

        for i in range(len(minmax)):
            if float(minmax[i]) == 0:
                df=df.drop([column[i]],axis=1)
                dropcols.append(column[i])
        df=df.dropna(how='any')
        new_df=self.normalize(df)
        b=self.bivarient_correlation(new_df)  
        w=self.weight(new_df,b)
        before_base_dxi=self.dxi(new_df,w)
        avg_dxi=np.mean(np.array(before_base_dxi))
        x=new_df
        print(f'Average Pre-Base DXI: {avg_dxi}')
        #############################################################
        y=self.create_label_forward_time_weighted(list(df_buynobuy[target]),0.95)
        ###############################################################
       
        clf = linear_model.Lasso(alpha=0.2)
        clf.fit(x,y)
        lasso_weight= clf.coef_
        min_max= (df.iloc[2])
        for i in range(len(min_max)):
            if lasso_weight[i] < 0 :
                if min_max[i] != 'MIN' :
                    (df.loc[2])[i]='MIN'
        new_df=self.normalize(df)

        update_b=self.bivarient_correlation(new_df)  
        parameter=df.iloc[3]
        update_w=self.weight(new_df,update_b)
        base_dxi=self.dxi(new_df,update_w)
        behaviour_dxi,transactional_dxi,kpi_dxi,visual_dxi=self.catogorical_dxi(new_df,update_w,parameter)
        avg_behaviour=np.mean(np.array(behaviour_dxi))
        avg_kpi=np.mean(np.array(kpi_dxi))
        avg_transactional=np.mean(np.array(transactional_dxi))
        avg_visual=np.mean(np.array(visual_dxi))
        avg_base_dxi=np.mean(np.array(base_dxi))
        print(f'Average Base DXI: {avg_base_dxi}')
        corrs = self.single_correlation(base_dxi,list(df_buynobuy[target]))
        print(f'Correlation w.r.t SXI: {corrs}')
        
        ############################################################
        update_y = self.create_label_forward_time_weighted(y,0.95)
        ############################################################
        
        print(f'MAPE: {mean_absolute_percentage_error(list(df_buynobuy[target]), update_y) * 100}')
        
       
        data=df.iloc[4:]
        data['Base_dxi_label']=update_y
        data['Base_dxi']=base_dxi
        data[tv]=df_buynobuy[tv]
        # op = new_df
        # scaler = MinMaxScaler()
        # x_train = scaler.fit_transform(op)
        # if (x.values < 0).any() == True:
        #     a = x_train
        # else:
        #     a = x
        # cnb = ComplementNB().fit(a, update_y)
        # logprobs = cnb.feature_log_prob_
        # avgprob =[]
        # for i in range(len(logprobs[0])):
        #     avgprob.append((logprobs[0][i]+logprobs[1][i])/2)
        # nb_weight = list(np.exp(avgprob) / (np.exp(avgprob)).sum())    

        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor

        bagging_model = BaggingRegressor(
                estimator=DecisionTreeRegressor(),
                n_estimators=100,  
                random_state=42
        )

        # Training the model
        bagging_model.fit(new_df,update_y)
        nb_weight = np.mean([
            tree.feature_importances_ for tree in bagging_model.estimators_], axis=0)
        xgb = xg.XGBRegressor().fit(new_df,update_y)
        xgb_weight = list(xgb.feature_importances_)

        mi_weight=list(self.mutual_informationreg(new_df, update_y))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit_transform(new_df)
        pca_weight = pca.components_[0]
        # pca_weight=list(self.pca_feature_selection(new_df,update_y))
        lasso_weight=list(self.lasso_feature_selection(new_df,update_y))

        print(f'Lasso Weight: {lasso_weight}')
        print(f'MI Weight: {mi_weight}')
        print(f'PCA Weight: {pca_weight}')
        print(f'Bagg Weight: {nb_weight}')
        print(f'XGB Weight: {xgb_weight}')
        # lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = self.composite_weight(new_df,lasso_weight,avg_base_dxi)
        # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi)
        mi_dxi,avg_mi_dxi,w_mi,weight_mi = self.composite_weight(new_df,mi_weight,avg_base_dxi)
        pca_dxi,avg_pca_dxi,w_pca,weight_pca = self.composite_weight(new_df,pca_weight,avg_base_dxi)
        nb_dxi,avg_nb_dxi,w_nb,weight_nb = self.composite_weight(new_df,nb_weight,avg_base_dxi)
        xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = self.composite_weight(new_df,xgb_weight,avg_base_dxi)
 
        # top_names =[]
        # for weight in [weight_nb,weight_mi, weight_pca,weight_xgb]:   
        #     idx = (-(np.array(weight))).argsort()[:len(weight)+1]
        #     names = x.columns[idx]
        #     top_names.append(names)

        # tp_params = {}
        # tp_params["Algorithm1"]=top_names[0][0:5] 
        # tp_params["Algorithm2"]=top_names[1][0:5]
        # tp_params["Algorithm3"]=top_names[2][0:5]
        # tp_params["Algorithm4"]=top_names[3][0:5]
        # tp_params["Algorithm5"]=top_names[4][0:5]

        sample_composite_dxi1=self.sample_composite_dxireg(new_df,weight_xgb, weight_mi, weight_pca,weight_nb)
        composite_dxi1=sample_composite_dxi1[0]
        avg_composite_dxi=np.mean(composite_dxi1)
        # print(f'SXI Scores: {composite_dxi1}')
        print(f'SXI: {avg_composite_dxi}')
        corrs1 = self.single_correlation(composite_dxi1,list(df_buynobuy[target]))
        print(f'Correlation w.r.t SXI: {corrs1}')
        df_buynobuy['composite_dxi'] = composite_dxi1
        df_buynobuy.to_csv('IOT_initial_sxi.csv',index=False)
        
        """
        update_y_com=self.create_label_forward(composite_dxi1,avg_composite_dxi)
        initsxiwgts = sample_composite_dxi1[2]
        topparms = pd.DataFrame(tp_params)
        value_counts = topparms.melt(value_name='value').value.value_counts()
        repeated_values = value_counts[value_counts > 1].index
        vallist = topparms.values.flatten().tolist()
        freqvals = [vallist.count(i) for i in repeated_values]
        most_common_features_df = pd.DataFrame(repeated_values, columns=['Most Common Features'])
        most_common_features_df['No.of Times'] = freqvals
        print(most_common_features_df)
        hcols = list(x.columns)
        impind = []
        for i in repeated_values:
            impind.append(hcols.index(i))
        featimp = [1] * len(hcols) 
        for i in range(len(impind)):
            featimp[impind[i]] = 1 + (freqvals[i]/len(impind))

        sxi = avg_composite_dxi
        df_buynobuy=df_buynobuy.dropna(how='any')
        df_buynobuy['composite_dxi_label'] = update_y_com
        df_buynobuy['composite_dxi'] = composite_dxi1
        # df_buynobuy.to_csv('electricconsumpinitsxi.csv',index=False)
        # print('File Saved')
        
        return fdf,sxi,finalsxiweights,initsxiwgts
        """

    
    
   