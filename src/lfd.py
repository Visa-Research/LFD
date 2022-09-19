# ©2022 Visa.
 
# Permission is hereby granted, free of charge, 
# to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), 
# to deal in the Software without restriction, 
# including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit 
# persons to whom the Software is furnished to do so, 
# subject to the following conditions:

# The above copyright notice and this permission 
# notice shall be included in all copies or substantial 
# portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY 
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
# OR OTHER DEALINGS IN THE SOFTWARE.

from sklearn.metrics import roc_auc_score, roc_curve, log_loss, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import shap
import csv
from sklearn.model_selection import train_test_split


class LFD():
    def __init__(self):
        self._data_dirty = True
        self._threshold_dirty = True
        
        self.A_score = pd.DataFrame()
        self.B_score = pd.DataFrame()
        self.pd_all = pd.DataFrame()
        self.X = pd.DataFrame()     # instances with meta-features 
        self.X_vis = pd.DataFrame() # instances used for visualization
        
        self.bst = None
        self.shap_values = None
        
        self.threshold = 0.15
        
        self.score_threshold_A = 0
        self.score_threshold_B = 0
        
        self.pd_record_A_plus = pd.DataFrame()
        self.pd_record_B_plus = pd.DataFrame()
        self.pd_record_A_plus_B_plus_tp = pd.DataFrame()
        self.pd_record_A_plus_B_plus_fp = pd.DataFrame()
        self.pd_record_A_plus_B_minus_tp = pd.DataFrame()
        self.pd_record_A_plus_B_minus_fp = pd.DataFrame()
        self.pd_record_A_minus_B_plus_tp = pd.DataFrame()
        self.pd_record_A_minus_B_plus_fp = pd.DataFrame()
        
    
    def set_A_score(self, file_A_score):
        self.A_score = pd.read_csv(file_A_score, sep=',', header=None, names=['id', 'true', 'A'], dtype={'id': 'string'})
        self._data_dirty = True
        
        if(not (self.A_score.empty or self.B_score.empty)):
            self._join_A_B_score()
    
    
    def set_B_score(self, file_B_score):
        self._data_dirty = True
        self.B_score = pd.read_csv(file_B_score, sep=',', header=None, names=['id', 'true_B', 'B'], dtype={'id': 'string'})
        
        if(not (self.A_score.empty or self.B_score.empty)):
            self._join_A_B_score()
        
        
    def _join_A_B_score(self):
        if self._data_dirty:
            self.pd_all = pd.merge(self.A_score, self.B_score, how="inner", on="id") # inner join, mismatch ones will be discarded
            self.pd_all = self.pd_all.drop(['true_B'], axis = 1)
            self._data_dirty = False

        pd_tp_all = self.pd_all[(self.pd_all['true']>0.5)]
        print("number of positive:", pd_tp_all.shape[0])
        print("number of records:", self.pd_all.shape[0])
        print("positive ratio: ", pd_tp_all.shape[0]/self.pd_all.shape[0])
        
    
    def _intersection(self, lst1, lst2): 
        return list(set(lst1) & set(lst2)) 


    def set_threshold(self, threshold):
        self.threshold = threshold
        
        record_count = int(self.threshold*self.pd_all.shape[0])

        pd_sorted_A = self.pd_all.sort_values(['A'], ascending=[False])
        #pd_record_A_plus = pd_sorted_A.iloc[0:record_count,:]
        #print("Data shape with count cutoff for A:", pd_record_A_plus.shape)
        self.score_threshold_A = pd_sorted_A['A'].iloc[record_count-1] # the score threshold 
        self.pd_record_A_plus = self.pd_all[self.pd_all['A']>=self.score_threshold_A]

        pd_sorted_B = self.pd_all.sort_values(['B'], ascending=[False])
        #pd_record_B_plus = pd_sorted_B.iloc[0:record_count,:]
        #print("Data shape with count cutoff for B:", pd_record_B_plus.shape)
        self.score_threshold_B = pd_sorted_B['B'].iloc[record_count-1]# the score threshold
        self.pd_record_B_plus = self.pd_all[self.pd_all['B']>=self.score_threshold_B]

        print("current threshold: ", self.threshold)
        print("A cutoff score: ", self.score_threshold_A)
        print("B cutoff score: ", self.score_threshold_B)
        print("A+:", self.pd_record_A_plus.shape[0])
        print("B+:", self.pd_record_B_plus.shape[0])
        
        self._threshold_dirty = True
        
        
    def compute_disagreement_matrix(self):
        if(self._threshold_dirty):
            # ===== A+ and B+ =====
            # join the two sets of data records (note that there are duplications!!)                        
            pd_record_six_cell = pd.concat([self.pd_record_A_plus, self.pd_record_B_plus])

            # the following two steps are to clean the duplications
            idx = np.unique(pd_record_six_cell.index.values, return_index=True)[1]
            pd_record_six_cell = pd_record_six_cell.iloc[idx]
            #print("number of total LFD records (all six cells): ", pd_record_six_cell.shape[0])
            #pd_record_six_cell.head()

            np_index_A_plus = self.pd_record_A_plus.index.values
            np_index_B_plus = self.pd_record_B_plus.index.values
        
            # ===== A+B+ =====
            list_index_A_plus_B_plus = self._intersection(np_index_A_plus, np_index_B_plus)
            #print("A+B+ size: ", len(list_index_A_plus_B_plus))
            #print("All six cell size:", len(np_index_A_plus)+len(np_index_B_plus)-len(list_index_A_plus_B_plus))
            
            self.pd_record_A_plus_B_plus = pd_record_six_cell.loc[list_index_A_plus_B_plus, :]
            self.pd_record_A_plus_B_plus_tp = self.pd_record_A_plus_B_plus[self.pd_record_A_plus_B_plus['true']>0.5]
            self.pd_record_A_plus_B_plus_fp = self.pd_record_A_plus_B_plus[self.pd_record_A_plus_B_plus['true']<0.5]
            
            pd_record_four_diff_cell = pd_record_six_cell.loc[~pd_record_six_cell.index.isin(list_index_A_plus_B_plus)]
            np_index_four_diff_cell = pd_record_four_diff_cell.index.values
            #print("the rest four cells in total: ", pd_record_four_diff_cell.shape)

            # ===== A+B- =====
            list_index_A_plus_B_minus = self._intersection(np_index_four_diff_cell, np_index_A_plus)
            self.pd_record_A_plus_B_minus = pd_record_four_diff_cell.loc[list_index_A_plus_B_minus, :]
            self.pd_record_A_plus_B_minus_tp = self.pd_record_A_plus_B_minus[(self.pd_record_A_plus_B_minus['true']>0.5)]
            self.pd_record_A_plus_B_minus_fp = self.pd_record_A_plus_B_minus[(self.pd_record_A_plus_B_minus['true']<0.5)]

            # ===== A-B+ =====
            list_index_A_minus_B_plus = self._intersection(np_index_four_diff_cell, np_index_B_plus)
            self.pd_record_A_minus_B_plus = pd_record_four_diff_cell.loc[list_index_A_minus_B_plus, :]
            #print("A-B+", pd_record_A_minus_B_plus.shape)

            # pd_record_A_minus_B_plus_tp
            self.pd_record_A_minus_B_plus_tp = self.pd_record_A_minus_B_plus[(self.pd_record_A_minus_B_plus['true']>0.5)]
            self.pd_record_A_minus_B_plus_fp = self.pd_record_A_minus_B_plus[(self.pd_record_A_minus_B_plus['true']<0.5)]
            
            print("A+B+(TP&FP): ", self.pd_record_A_plus_B_plus.shape[0])
            print("A+B-(TP&FP): ", self.pd_record_A_plus_B_minus.shape[0])
            print("A-B+(TP&FP): ", self.pd_record_A_minus_B_plus.shape[0])
            
            print("A+B+(TP): ", self.pd_record_A_plus_B_plus_tp.shape[0])
            print("A+B-(TP): ", self.pd_record_A_plus_B_minus_tp.shape[0])
            print("A-B+(TP): ", self.pd_record_A_minus_B_plus_tp.shape[0])
            
            print("A+B+(FP): ", self.pd_record_A_plus_B_plus_fp.shape[0])
            print("A+B-(FP): ", self.pd_record_A_plus_B_minus_fp.shape[0])
            print("A-B+(FP): ", self.pd_record_A_minus_B_plus_fp.shape[0])
            
            self._threshold_dirty = False
        
    
    def set_meta_features(self, data_dir):
        print("Loading meta-features...")
        types = {'id': 'string'} # the first varialble, id, should be in string format
        self.X = pd.read_csv(data_dir, sep=',', index_col=False, dtype=types)
        print("Number of meta-features:", self.X.shape[1]-1) # there is one extra id column
        
    
    def set_meta_features_large_file(self, data_dir):
        print("Loading meta-features...")
        col_names = self._get_file_header(data_dir)
        types = {}
        for col_name in col_names:
            if col_name=='id':
                types[col_name] = 'string'
            else:
                types[col_name] = 'float16' # use 2 bytes only for float
        self.X = pd.read_csv(data_dir, sep=',', index_col=False, dtype=types)
        print("The shape of the meta-feature:", self.X.shape)
        print("Number of meta-features:", self.X.shape[1]-1) # there is one extra id column
        
        
    def learn(self, cell='tp', learning_rate=0.2, tree_depth=6, nrounds=100, early_stop=1, early_stop_round=50, test_size = 0.2):
        # compose training data
        if cell=='tp': # labe 1 for A-B+(tp)
            print("train the TP discriminator...")
            label_0_data = pd.merge(self.pd_record_A_plus_B_minus_tp, self.X, how="inner", on="id") # inner join, mismatch ones will be discarded
            label_1_data = pd.merge(self.pd_record_A_minus_B_plus_tp, self.X, how="inner", on="id") # inner join, mismatch ones will be discarded
        elif cell=='fp': # label 1 for A-B+(fp)
            print("train the FP discriminator...")
            label_0_data = pd.merge(self.pd_record_A_plus_B_minus_fp, self.X, how="inner", on="id") # inner join, mismatch ones will be discarded
            label_1_data = pd.merge(self.pd_record_A_minus_B_plus_fp, self.X, how="inner", on="id") # inner join, mismatch ones will be discarded
        elif cell=='both':
            print("train the discriminator on both TP and FP data...")
            label_0_data = pd.merge(self.pd_record_A_plus_B_minus, self.X, how="inner", on="id") # inner join, mismatch ones will be discarded
            label_1_data = pd.merge(self.pd_record_A_minus_B_plus, self.X, how="inner", on="id") # inner join, mismatch ones will be discarded
        else:
            print("please specify the discriminator type!")
            return
        
        print("Number of Negative Instances:", label_0_data.shape)
        print("Number of Positive Instances:", label_1_data.shape)
        X_raw = pd.concat([label_0_data, label_1_data])
        Y_raw = [0]*label_0_data.shape[0] + [1]*label_1_data.shape[0]

        sample_pos_weight = label_0_data.shape[0]/label_1_data.shape[0]
        print("positive/negative ratio for the discriminator:", sample_pos_weight)
        
        # default parameters
        # learning_rate = 0.2
        # tree_depth = 6
        # nrounds = 100
        # early_stop = 1
        # early_stop_round = 50
        # test_size = 0.2
        
        split_seed = 1024
        
        tree_param = {
            'verbosity': 1,
            'max_depth': tree_depth, 
            'eta': learning_rate, 
            #'subsample': 0.1, 
            'tree_method': 'hist', # valid values are: {'approx', 'auto', 'exact', 'gpu_exact', 'gpu_hist', 'hist'}
            'objective': 'binary:logistic', 
            #'scale_pos_weight': sample_pos_weight,
            'eval_metric': 'auc',
            #'seed': xgb_seed,
        }
        
        # update parameters if given by users
        X_train, self.X_vis, Y_train, Y_vis = train_test_split(X_raw, Y_raw, test_size=test_size, random_state=split_seed)
         
        xgb.rabit.init()
        if(early_stop):
            # use the data itself as validation, as we just want to overfit the model
            dtrain = xgb.DMatrix(X_train.drop(list(self.pd_all.keys()), axis = 1), label=Y_train) # remove column, [id, true, A, B]
            dvalid = xgb.DMatrix(self.X_vis.drop(list(self.pd_all.keys()), axis = 1), label=Y_vis)
            self.bst = xgb.train(tree_param, dtrain, nrounds, evals=[(dtrain, 'train'), (dvalid, 'valid')], early_stopping_rounds=early_stop_round)
        else:
            dtrain = xgb.DMatrix(X_raw.drop(list(self.pd_all.keys()), axis = 1), label=Y_raw)
            dvalid = xgb.DMatrix(self.X_vis.drop(list(self.pd_all.keys()), axis = 1), label=Y_vis)
            self.bst = xgb.train(tree_param, dtrain, nrounds, evals=[(dtrain, 'train'), (dvalid, 'valid')])
        xgb.rabit.finalize()
        
    
    def compute_shap(self):        
        X_valid = self.X_vis.drop(list(self.pd_all.keys()), axis = 1) 
        
        explainer = shap.TreeExplainer(model=self.bst, feature_perturbation='interventional')
        self.shap_values = explainer.shap_values(X_valid)
        # print(self.shap_values.shape)
        
        
    def vis(self, num_feat=20):
        shap.initjs() # load JS visualization code to notebook
        
        X_valid = self.X_vis.drop(list(self.pd_all.keys()), axis = 1) 
        
        plt.figure(figsize=(100, 150))
        shap.summary_plot(self.shap_values, X_valid, show=False, max_display=num_feat)
        
        
    def save_samples(self, outdir):
        self.X_vis.to_csv(out_dir, index=False)
    
    
    def save_samples_shap(self, out_dir):
        X_shap = pd.DataFrame(self.shap_values, columns=X_valid.keys())
        X_shap.to_csv(out_dir, index=False)
        
        
    def _get_file_header(self, file_name):
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            list_of_column_names = []
            for row in csv_reader:
                list_of_column_names.append(row)
                break
        print("column names: ", list_of_column_names[0])
        return list_of_column_names[0]
