######################################################
# NEXT PURCHASE FORECAST
# File name: next_purchasing_forecast.py
# Author: Nhan Thanh Ngo
######################################################

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
import re
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn import preprocessing
import random
import pickle
import yaml
import lightgbm as lgb

####################################################
# General

# MODEL PARAMETER
PREDICT_LEN    = 50    # period of time need to create clasified cluster
CLUSTER_PERIOD = 7     # time range of each cluster
PRERROR        = 2     # variant from mean (central next purchase day)
SPLIT_BACKWARD = 60    # how many day before last day for split time

SLIDE_BACKWARD_STEP = 50 #Slide step for get more samples in TRAIN MODE

# DATA PRE-PROCESSING
PUR_TIME_PER_DAY_UNNORMAL_THRESHOLD = 8 #
TOTAL_PUR_DAY_UNNORMAL_THRESHOLD = 200  # will be change as below
PUR_PERIOD = 1 # avg 4 days/purchase

# Model 1
### Choose number of cluster for dimensions [R, F, M] 
R_kcluster = int((PREDICT_LEN-1)/CLUSTER_PERIOD+1)
F_kcluster = int((PREDICT_LEN-1)/CLUSTER_PERIOD+1)
M_kcluster = int((PREDICT_LEN-1)/CLUSTER_PERIOD+1)

# Model 2
SLIDE_STEP = 14
R_KCLUSTER_ML2 = int((PREDICT_LEN-1)/CLUSTER_PERIOD+1)
M_KCLUSTER_ML2 = int((PREDICT_LEN-1)/CLUSTER_PERIOD+1)
CATE_KCLUSTER_ML2 = int((PREDICT_LEN-1)/CLUSTER_PERIOD+1)

#Statistics Predict (Frequent customer)
BASE_TIME_BLOCK = 7 # Month

####################################################
version = 'demo' 
versionx=re.sub("_",".",version)

####################################################
# Step 0: GetOptions
import sys, getopt
import glob
import time

#default
train_test_split_mode = True
train_mode = False
test_mode = False
predict_stats_mode = False
predict_ml_mode = False
predict_merge = False
predict_mode = True
pseudo_labeling = False
global debug
debug = False

model_path = ''

input = "need_to_specified_table"
mode = 'PREDICT' # it could be 'TRAIN_TEST_SPLIT' 'TRAIN' 'TEST'
query = False
limit = ''
commit = False
skfold = False #skfold False, the test_test_split() will be used to divide Train and Valid data
commitonly = False
stats_update = False
global g_model
global models
models = 'xgb'
global model_dict 
model_dict = {'xgb':'XGBoost Classifier', 'dct': 'Decision Tree Classifier', 'svc':'Support Vector Classifier', 'rf':'Random Forest Classifier', 'xgbr':'XGBoost Regressor', 'lgb':'LightGBM'}
backsplit = 0
yaml_file='default.yaml'
NFOLD = 5

from optparse import OptionParser

usage = "usage: %prog [options] arg1 arg2 ...\n\n"\
        "Example:"\
        "\n\tpython %prog -m TRAIN   --input=DW.tablename [--config=custom_config_a] [--query]    (query + TRAIN)"\
        "\n\tpython %prog -m TRAIN   --input=DW.tablename [--model=rf] (TRAIN with model random forest)"\
        "\n\tpython %prog -m TEST    --input=DW.tablename [--query]"\
        "\n\tpython %prog -m PREDICT --input=DW.tablename [--query] [--commit]"\
        "\n\tpython %prog -m PREDICT --commit   (PREDICT + COMMIT)"\
        "\n\tpython %prog --commitonly"\
        "\nFeatures:\n\tAllow user in choosing mode to run: 'TRAIN_TEST_SPLIT', 'TRAIN', 'TEST', 'PREDICT', 'PREDICT_ML', 'PREDICT_STATS' mode"\
        "\n\t\t'TRAIN_TEST_SPLIT': Split data into 2 files, one file run TRAIN, one file run TEST"\
        "\n\t\t'TRAIN': Run TRAIN+VALIDATION, TEST, then run TRAIN with all data, save model."\
        "\n\t\t'TEST': Run TEST ONLY with the indicated data."\
        "\n\t\t'PREDICT': Run PREDICT Only for the indicated data, data out is csv file. Use --commit to submit"\
        "\n\tAllow query data, or using offline input file (saved csv on PC)"\
        "\n\tAllow select model build (xgboost[xgb], random-forest[rf], decision-tree [dct], support-vector[svc])"

parser = OptionParser(usage=usage)

parser.add_option("-i", "--input",
                  default="`Aldo.nodefaultyet`",
                  metavar="SANDBOX", help="Sandbox dataname"
                                     "[default: %default]")        
       
parser.add_option("-q", "--query",
                  default="False",
                  help="Query mode: query data from Big Query"
                       "[default: %default]")
                       
parser.add_option("-m", "--mode",
                  default="PREDICT",
                  help="Choose mode for run ['TRAIN_TEST_SPLIT', 'TRAIN', 'TEST', 'PREDICT_STATS', 'PREDICT_ML', 'PREDICT']"
                       "[default: %default]")

parser.add_option("-t", "--model",
                  default="XGBoost",
                  help="Choose with model will be run ['rf', 'dct', 'svc']"
                       "[default: %default]")

parser.add_option("-g", "--config",
                  default="config_default",
                  help="Choose particular PARAMETER CONFIG for particular scenario (eg. custom_config_a, custom_config_b, custom_default, etc."
                       "[default: %default]")

parser.add_option("-p", "--pseudo",
                  default="False",
                  help="Enable pseudo labeling when training model. Good option when base model > 0.8 accuracy"
                       "[default: %default]")

parser.add_option("-l", "--limit",
                  default="LIMIT",
                  help="Limit number of line in SQL command"
                       "[default: %default]")
                  
parser.add_option("-c", "--commit",
                  default="False",
                  help="Commit result to sandbox"
                       "[default: %default]")

parser.add_option("-u", "--commitonly",
                  default="False",
                  help="Only run commit file (Not cluster)"
                       "[default: %default]")

parser.add_option("-a", "--modelpath",
                  default="model path is formed from input table name",
                  help="modelpath, used when run predict with differ input name"
                       "[default: %default]")
                       
parser.add_option("-b", "--backsplit",
                  default="0",
                  help="PREDICT mode only: number of day backward of current_day will be used as SPLIT DATE"
                       "[default: %default]")

parser.add_option("-d", "--debug",
                  default="False",
                  help="Enable debug mode"
                       "[default: %default]")

parser.add_option("-k", "--skfold",
                  default="False",
                  help="StratifiedKFold enable"
                       "[default: %default]")
                       
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hq:i:m:t:l:b:g:a:p:c:u:d:k:s', ['help','query','input=','mode=','model=','limit=','backsplit=','config=','modelpath=','pseudo','commit','commitonly','debug','skfold','stats_update'])
  
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-m', '--mode'):
    mode = str(arg)
  if opt in ('-t', '--model'):
    models = str(arg)
  if opt in ('-l', '--limit'):
    limit = " LIMIT "+str(arg)
  if opt in ('-g', '--config'):
    yaml_file = str(arg)+'.yaml'
  if opt in ('-a', '--modelpath'):
    model_path = "./"+str(arg)+"_next_purchase_forecast_"+version+"_log"
  if opt in ('-p', '--pseudo'):
    pseudo_labeling = True
  if opt in ('-c', '--commit'):
    commit = True
  if opt in ('-u', '--commitonly'):
    commitonly = True
  if opt in ('-b', '--backsplit'):
    backsplit = int(arg)
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-k', '--skfold'):
    skfold = True
  if opt in ('-k', '--stats_update'):
    stats_update = True	
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)

if commitonly:
  mode = 'COMMIT' 
  query = False
  limit = ''
  commit = True

# Classification
if models == 'xgb':
   g_model = xgb.XGBClassifier()
elif models == 'rf':
   g_model = RandomForestClassifier(max_depth=8, random_state=0)
elif models == 'dct':
   g_model = DecisionTreeClassifier(max_depth=8, random_state=0)
elif models == 'svc':
   g_model = SVC(gamma='auto')

# Regressor
elif models == 'xgbr':
   g_model = xgb.XGBRegressor()

elif models == 'lgb':
   g_model = "LightGBM"  
   
else:
   print("[Error] Please check --model=... to be sure that you enter the right value. If not, please leave this option.")

model_name = model_dict.get(models)

print("##################################")
print("ARGUMENTS:")
print("Configure   : {}".format(yaml_file))
print("Model Path  : {}/model_{}".format(model_path,models))
print("Debug mode  : {}".format(debug))
print("Query mode  : {}".format(query))
print("Running Mode: {}".format(mode))
print("pseudo      : {}".format(pseudo_labeling))
print("skfold      : {}".format(skfold))
print("Limit       : {}".format(limit))
print("Commit      : {}".format(commit))
print("CommitOnly  : {}".format(commitonly))
print("Input Name  : {}".format(input))
print("Model Name  : {}".format(model_name))
print("##################################")

train_test_split_mode = True if mode=='TRAIN_TEST_SPLIT' else False
train_mode              = True if mode=='TRAIN'           else False
test_mode               = True if mode=='TEST'            else False
predict_mode            = True if mode=='PREDICT'         else False
predict_ml_mode         = True if mode=='PREDICT_ML'      else False
predict_stats_mode      = True if mode=='PREDICT_STATS'   else False

#skfold string
if skfold:
  skf_en = 'skf'
else:
  skf_en = 'tts'
  
###########################################
#create DEBUG directory
#

import os
cwd = os.getcwd()
print (cwd)
log_path = "./"+input.split('.')[-1]+"_next_purchase_forecast_"+version+"_log"

#### create log folder path
if os.path.exists(log_path):
  print ("\'{}\' is already EXISTED!\n".format(log_path))
  #shutil.rmtree(log_path)
  #os.mkdir(log_path)
else:
  os.mkdir(log_path)
  print ("\'{}\' is CREATED!\n".format(log_path))

#### ---------------- debug_log file path
debug_log = 'data_processing_debug_'+skf_en+'_'+mode+'_'+model_name+'.log'
if os.path.isfile(log_path+"/"+debug_log):
  os.remove(log_path+"/"+debug_log)


#### --------------- report_log file path
report_log = 'data_processing_report_'+skf_en+'_'+mode+'_'+model_name+'.log'
if os.path.isfile(log_path+"/"+report_log):
  os.remove(log_path+"/"+report_log)
  
#### -------------- where the model is save
if not model_path:
  model_path = log_path

savemodel_path = model_path+"/model_"+models
if os.path.exists(savemodel_path):
  print ("\'{}\' is already EXISTED!\n".format(savemodel_path))
  #shutil.rmtree(savemodel_path)
  #os.mkdir(savemodel_path)
else:
  os.mkdir(savemodel_path)
  print ("\'{}\' is CREATED!\n".format(savemodel_path))

#store all statistics file from stats model
stats_model_path = model_path+"/stats_model_files"
if os.path.exists(stats_model_path):
  print ("\'{}\' is already EXISTED!\n".format(stats_model_path))
  #shutil.rmtree(savemodel_path)
  #os.mkdir(savemodel_path)
else:
  os.mkdir(stats_model_path)
  print ("\'{}\' is CREATED!\n".format(stats_model_path))

# store all predict output data
predict_output = log_path+"/predict_output"
if os.path.exists(predict_output):
  print ("\'{}\' is already EXISTED!\n".format(predict_output))
  #shutil.rmtree(savemodel_path)
  #os.mkdir(savemodel_path)
else:
  os.mkdir(predict_output)
  print ("\'{}\' is CREATED!\n".format(predict_output))

# store all accuracy file from ML model
ml_model_path = model_path+"/ml_model_files"
if os.path.exists(ml_model_path):
  print ("\'{}\' is already EXISTED!\n".format(ml_model_path))
  #shutil.rmtree(savemodel_path)
  #os.mkdir(savemodel_path)
else:
  os.mkdir(ml_model_path)
  print ("\'{}\' is CREATED!\n".format(ml_model_path))
###################################################
# GENERAL FUNCTION
#

def print_debug (strcont):
  debugn = debug
  if debugn:
    print(strcont)  
    with open(log_path+"/"+debug_log, "a") as logfile:
      logfile.write("\n"+strcont)

def print_report (strcont):
  debugn = debug
  if debugn:
    print(strcont)  
  with open(log_path+"/"+report_log, "a") as logfile:
    logfile.write("\n"+strcont)

###################################################
# READ PARAMETER CONFIG
#

config_dict = {}
with open(r'./config/'+yaml_file) as file:
  # The FullLoader parameter handles the conversion from YAML
  # scalar values to Python the dictionary format
  config_dict = yaml.load(file, Loader=yaml.FullLoader)
  print('[Main] config_dict = {}'.format(config_dict))

for key in config_dict:
  if key == 'PREDICT_LEN':
    PREDICT_LEN = config_dict[key]
  elif key == 'CLUSTER_PERIOD':
    CLUSTER_PERIOD = config_dict[key]    
  elif key == 'PRERROR':
    PRERROR = config_dict[key]
  elif key == 'SPLIT_BACKWARD':
    SPLIT_BACKWARD = config_dict[key]
  elif key == 'SLIDE_BACKWARD_STEP':
    SLIDE_BACKWARD_STEP = config_dict[key]
  elif key == 'PUR_TIME_PER_DAY_UNNORMAL_THRESHOLD':
    PUR_TIME_PER_DAY_UNNORMAL_THRESHOLD = config_dict[key]
  elif key == 'PUR_PERIOD':
    PUR_PERIOD = config_dict[key]
  elif key == 'SLIDE_STEP':
    SLIDE_STEP = config_dict[key]


print_debug("##############################################")
print_debug("Scenarios:")
print_debug("YAML file                           : {}".format(yaml_file))
print_debug("PREDICT_LEN                         : {}".format(PREDICT_LEN))
print_debug("CLUSTER_PERIOD                      : {}".format(CLUSTER_PERIOD))
print_debug("PRERROR                             : {}".format(PRERROR))
print_debug("SPLIT_BACKWARD                      : {}".format(SPLIT_BACKWARD))
print_debug("SLIDE_BACKWARD_STEP                 : {}".format(SLIDE_BACKWARD_STEP))
print_debug("PUR_TIME_PER_DAY_UNNORMAL_THRESHOLD : {}".format(PUR_TIME_PER_DAY_UNNORMAL_THRESHOLD))
print_debug("PUR_PERIOD                          : {}".format(PUR_PERIOD))
print_debug("SLIDE_STEP                          : {}".format(SLIDE_STEP))
print_debug("##############################################")


###################################################
# Step 1: check `query`

df = pd.DataFrame()
if not commitonly:
  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  sql = "SELECT * FROM "+input+limit
  bq_cus_purchase = input.split('.')[-1]+".csv"

  if query:
    # Run a Standard SQL query with the project set explicitly
    print("Querying data from sandbox...\nit will take a few minutes...")
    df = client.query(sql, project=project_id).to_dataframe()
    
    if not glob.glob(bq_cus_purchase):
      print ("[INFO] No BigQuery datafile available")
    else:
      print("[INFO] Remove exist bq datafile")
      os.remove(bq_cus_purchase)
    
    print("[INFO] Store query data from Big Query to file")
    df.to_csv(bq_cus_purchase)
  else:
    print("[INFO] Read input data from offline file, need update please run again with -q to query new data from Big Query")
    print("Read offline input data file...\nit will take a few minutes...")
    df = pd.read_csv(bq_cus_purchase, index_col=None)
    
  df = df[['partner_id','order_id','channel','date','category_id','qty','rev','outlet_id']]
  df.columns = ['customer_id','order_id','channel','date_order','category','qty','amount','store_id']
  #df = df[:20000]
  #information
  print_debug ("[INFO] customer purchase history: {}\n{}".format(df.shape,df[:5]))
  
  # add category to df
  df['date_order'] = pd.to_datetime(df.date_order.values).strftime("%Y-%m-%d")
  df['date_order'] = pd.to_datetime(df.date_order)
  df['category'] = df['category'].astype(str)

  print_debug ("df.date_order.max()={}".format(df.date_order.max()))
  #df=df[df.date_order<=datetime.datetime(2019,12,29)]
  #sys.exit(1)
  print_debug ("[INFO] df input final: {}\n{}".format(df.shape,df.head()))


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
# General Function 
#
if not debug:
  print("-----------------------------------------------------------------")
  print("DEBUG MODE IS DISABLE --> no more info will be showed following...")

##################################################################################
# Function: kmeans_cluster(df,kcluster,asc,model_savepath)
#    - Using for running kmeans for R, F, M
#    - Sort sum value of each cluster is applied (ascending/descending mode)
#    - saving model
# Usage:
#    Input: - df: dataframe [customer_id, r|f|m]
#           - kcluster: number of cluster used for kmeans
#           - asc: True (ascending), False (descending)
#           - model_savepath: where to save model
#    Output: return dataframe of cluster result [customer_id, r_clus|f_clus|m_clus]
#

def kmeans_cluster(df,kcluster,asc,model_savepath):

  col_name = df.columns.values[-1]# r, f, m
  
  print_debug("####################################################")
  print_debug("# KMeans CLUSTER FOR DIMENSION: {}".format(col_name))
  print_debug("####################################################")
  cus1 = df.customer_id.nunique()
  col_clus = col_name+"_clus"
  kmeans = KMeans(n_clusters=kcluster, random_state=10).fit(df[[col_name]])
    
  df.loc[:,col_clus]= kmeans.labels_
  
  #save Recency Model  
  pickle.dump(kmeans,open(model_savepath,'wb'))
  print_debug("[kmeans_cluster()][SAVE MODEL] saved model at {}".format(model_savepath))

  x = df.groupby([col_clus])[col_name].mean().reset_index()
  x.columns = [col_clus,'mean']
  y = df.groupby([col_clus])[col_name].std().reset_index()
  y.columns = [col_clus,'std']
  x = pd.merge(x,y,on=col_clus,how='left')
  x['spm'] = x['std']/x['mean']
  
  x.sort_values(['mean'],ascending=asc,inplace=True)
  x.reset_index(inplace=True)
  x.reset_index(inplace=True)
  df = pd.merge(df,x[['level_0',col_clus]],on=col_clus,how='left')
  df['cluster'] = df['level_0']
  df = df[['customer_id',col_clus]]

  print_debug ("[kmeans_cluster()][OUTPUT] cluster result of kmeans clustering: {}\n{}".format(df.shape,df[:5]))
  
  cus2 = df.customer_id.nunique()
  print_debug('number customer remove: {}-{}={}'.format(cus1, cus2, cus1-cus2))

  return df


##################################################################################
# Function: predict_nextpurchase_thro_freq_pur_stats_method(df_pivot)
# Description:
#    - data_pre_processing() --> predict_customer_frequent_purchase_active() --> predict_nextpurchase_thro_freq_pur_stats_method()
#    - sweep all data of df_pivot table, count number of times happen N continuous purchase
#    - calculate accuracy
# Usage:
#    Input: -df_pivot: dataframe [customer_id, 01-2018, 02-2018, ...,12-2019,...]
#    Output: df_cus_groupday['customer_id','group_period','num_period_cont']
# 
  
def predict_nextpurchase_thro_freq_pur_stats_method(df_pivot,NUM_PERIOD):
 
  # loop with multiple diff NUM_PERIOD periods for check accuracy of prediction method
  stop_point = 2

  #add new columns NUM_PERIOD
  column = df_pivot.columns
  df_pivot.loc[:,'num_period_cont'] = np.zeros(len(df_pivot),dtype=int)
  
  #rearrange columns
  df_pivot = df_pivot[['num_period_cont'] + list(column)]  
  df_pivot = df_pivot.reset_index()  
  print("df_pivot customer_id={}".format(df_pivot.columns.values))

  df_cus_groupday = pd.DataFrame(columns=['customer_id','num_period_cont'])  
  print("len df_pivot = {}".format(len(df_pivot)))

  # sweep column of df_pivot from lastest to earlier   
  for col in reversed(df_pivot.columns.values):

    df_pivot['num_period_cont'] = df_pivot.apply(lambda x: x['num_period_cont'] if x[col] == 0 else x['num_period_cont']+1, axis=1)
    df_cus_groupday = pd.concat([df_cus_groupday,df_pivot[df_pivot[col]==0][['customer_id','num_period_cont']]])    
    #df_cus_groupday.loc[:,['customer_id','num_period_cont']] = df_pivot[df_pivot[col]==0].loc[:,['customer_id','num_period_cont']]
    df_pivot = df_pivot[~(df_pivot[col]==0)]
    
    if (col == stop_point) | (len(df_pivot)==0):
      break
    
  df_cus_groupday.loc[:,'group_period'] = NUM_PERIOD
  
  print_debug("[freq_purchase_accuracy()] dict_cont_purchase = \n{}\n".format(df_cus_groupday))

  return df_cus_groupday
  
##################################################################################
# Function: freq_purchase_accuracy(df_pivot)
# Description:
#    - data_pre_processing() --> predict_customer_frequent_purchase_active() --> freq_purchase_accuracy()
#    - sweep all data of df_pivot table, count number of times happen N continuous purchase
#    - calculate accuracy
# Usage:
#    Input: -df_pivot: dataframe [customer_id, 01-2018, 02-2018, ...,12-2019,...]
#    Output: accuracy dictionary
#

def freq_purchase_accuracy(df_pivot):

  #stop_point = np.max(df_pivot.columns.values)
  #print(stop_point)
  df_pivot['acc'] = np.zeros(len(df_pivot))
  dict_cont_purchase = {}
  
  if len(df_pivot) > 5000:
    DIV = int(len(df_pivot)/5000)+1
    DF_SPLIT = int(len(df_pivot)/DIV)+1
  else:
    DIV = 1
    DF_SPLIT = len(df_pivot)
  
  for i in np.arange(0,DIV):
    end = 0
    if i == DIV-1:
      end = None
    else:
      end = (i+1)*DF_SPLIT
    
    df_pivot_i = df_pivot[DF_SPLIT*i:end]
    #print("*************   i={}\n".format(i))
    ###############
    #stop_point = np.max(df_pivot_i.columns.values)
    for col in df_pivot_i.columns.values:
  
      # get customer just ended continous purchasing  
      harvest_arr = df_pivot_i[df_pivot_i[col] == 0].acc.values
      harvest_arr_stat = np.unique(harvest_arr,return_counts=True)    
    
      # record accumulatively of continuous purchasing periods
      for ind, key in enumerate(np.array(harvest_arr_stat[0])):
        if int(key) in dict_cont_purchase:
          dict_cont_purchase[int(key)] += harvest_arr_stat[1][ind]
        else:
          dict_cont_purchase[int(key)] = harvest_arr_stat[1][ind]
        
      #print_debug ("[freq_purchase_accuracy()] dict_cont_purchase = \n{}\n".format(dict_cont_purchase))
    
      # remove accumulation value that ended continuation
      df_pivot_i.loc[:,'acc'] = df_pivot_i.apply(lambda x: 0 if x[col] == 0 else x['acc'], axis=1)

      #continue accumulation
      df_pivot_i.loc[:,'acc'] = df_pivot_i.loc[:,'acc'] + df_pivot_i.loc[:,col]
  
      #if col == stop_point:
        #break
    ###############
    #print("--> df_pivot_i = {}".format(df_pivot_i['acc']))
    
  # accuracy of freq customer in continuous purchase
  dict_accuracy = {}
  dict_cont_purchase_key = [ key for key in dict_cont_purchase.keys()]
  for ind, key in enumerate(dict_cont_purchase_key):
    if ind < len(dict_cont_purchase_key):
      numerator = sum(dict_cont_purchase[keyx] for keyx in dict_cont_purchase_key[ind+1:])      
      denominator = sum(dict_cont_purchase[keyx] for keyx in dict_cont_purchase_key[ind:])
      dict_accuracy[key] = numerator/denominator
    else:
      dict_accuracy[key] = 0

  return dict_accuracy


##################################################################################
# Function: create_cus_purchase_mark_by_any_n_period_ofbasetime(df_pivot,NUM_PERIOD)
# Description:
#    - data_pre_processing() --> predict_customer_frequent_purchase_active() --> create_cus_purchase_mark_by_any_n_period_ofbasetime()
#    - base on input df_pivot, generate another df_pivot_month from input by group N month together (each month is each column)
#    - return df_pivot_month following NUM_PERIOD, for eg. 2 months, 3 months, or N months
# Usage:
#    Input: -df_pivot: dataframe [customer_id, 01-2018, 02-2018, ...,12-2019,...], value is having purchase or not (1: purchased, 0: none)
#    Output: dataframe
#

def create_cus_purchase_mark_by_any_n_period_ofbasetime(df_pivot,NUM_PERIOD):

  #NUM_PERIOD = 2
  df_any_group_period = pd.DataFrame()
  df_any_group_period['customer_id'] = df_pivot.index.values
  TOTAL_TIME_COL = len(df_pivot.columns.values)

  for i in np.arange(0,TOTAL_TIME_COL,NUM_PERIOD):
    if not i:
      end=None
    else:
      end=-i
    
    df_any_group_period[str(int((TOTAL_TIME_COL-i)/NUM_PERIOD))] = df_pivot.iloc[:,-NUM_PERIOD-i:end].sum(axis=1).values

  df_any_group_period.set_index(['customer_id'],drop=True,inplace=True)
  df_any_group_period = df_any_group_period[df_any_group_period.columns.values[::-1]]
  
  # adjust value of dataframe, adjust all value greater than 1 to 1
  df_any_group_period=df_any_group_period.clip(0,1)
  
  return df_any_group_period
  
##################################################################################
# Function: process_df_pivot(df_order)
# Description:
#    - predict_customer_frequent_purchase_active() --> process_df_pivot(df_order)
#    - PROBLEM:  when run direct pivot command for customer and pivot for all dates, the output date will be large and can not work
#    - SOLUTION: process pivot for small group of customers and then pd.concat()
# Usage:
#    Input: - df_order_line
#    Output: df_pivot 
#
 

# when split all customer_id into smaller ones
# the date_order does not cover full range, and the same for all small group
# this cause WRONG when create df_pivot by pd.concat()
# SOLUTION: Add ONE DUMMY CUSTOMER with All day having order into all smaller groups

def process_df_pivot(df_order):
  
  #first=True
  df_pivot = pd.DataFrame()
   
  # list of all customer_id
  customer_arr = df_order.customer_id.unique() 
 
  print ("Total Number of customer: {}".format(len(customer_arr)))

  ######################################
  LAST_DAY = df_order.year_month_day.max()
  FIRST_DAY = df_order.year_month_day.min()
  TOTAL_DAY = int((pd.to_datetime(LAST_DAY) - pd.to_datetime(FIRST_DAY)).days) + 1
  print("LAST_DAY={} FIRST_DAY={} TOTAL_DAY={}".format(LAST_DAY, FIRST_DAY, TOTAL_DAY))
  
  dummy_date_order = pd.date_range(start=pd.to_datetime(FIRST_DAY), end=pd.to_datetime(LAST_DAY))
  
  df_dummy_cus = pd.DataFrame({'year_month_day': dummy_date_order})
  df_dummy_cus['customer_id'] = 'dummy'
  df_dummy_cus['have_order'] = int(1)  
  df_dummy_cus['year_month_day'] = pd.to_datetime(df_dummy_cus['year_month_day']).dt.strftime("%Y-%m-%d")  
  df_dummy_cus = df_dummy_cus[['customer_id','year_month_day','have_order']]

  ######################################
  
  # DIV is number of split blocks
  # NUM_SPLIT is quantity of customer_id in each DIV
  if len(customer_arr) > 5000:
    DIV = int(len(customer_arr)/5000)+1
    NUM_SPLIT = int(len(customer_arr)/DIV)+1
  else:
    DIV = 1
    NUM_SPLIT = len(customer_arr)

  # sweep from DIV_0 to DIV_x
  for i in np.arange(0,DIV):
    end = 0
    if i == DIV-1:
      end = None
    else:
      end = (i+1)*NUM_SPLIT
    
    df_order_i = df_order[df_order.customer_id.isin(customer_arr[NUM_SPLIT*i:end])]
    df_order_i = pd.concat([df_order_i,df_dummy_cus]).reset_index(drop=True)
    print ("Total customer of DIV = {} is {}".format(i,len(df_order_i.customer_id.unique())))
    print ("df_order_i DIV = {} is {}".format(i,df_order_i))

    # df_pivot by day
    df_pivot_i = pd.pivot_table(df_order_i, values='have_order', index=['customer_id'], columns=['year_month_day'], aggfunc=np.sum, fill_value=0)
    print ("[1]df_pivot_i DIV = {} is {}".format(i,df_pivot_i))  
	
    # create new df_pivot following BASE_TIME_BLOCK
    df_pivot_i = create_cus_purchase_mark_by_any_n_period_ofbasetime(df_pivot_i,BASE_TIME_BLOCK)    
    df_pivot = pd.concat([df_pivot, df_pivot_i])

  df_pivot = df_pivot[df_pivot.index != 'dummy']
  if len(df_pivot) == len(customer_arr):
    print("[process_df_pivot] Number of customer have been processed: {}".format(len(df_pivot)))
    return df_pivot
  else:
    print("[process_df_pivot] ERROR: Something is wrong with this function, please check!!!")
    sys.exit(2)

##################################################################################
# Function: predict_customer_frequent_purchase_active(df)
# Description:
#    - data_pre_processing() --> predict_customer_frequent_purchase_active()
#    - [1] statistics GROUP_PERIOD, CONTINUOUS_PURCHASE_PERIOD, compute predict proportion --> save file (accuracy_df)
#    - [2] compute last active continuous purchasing of customer, this will be used to predict for next purchase in next period.
#    - [3] ONLY Predict for customer having Frequent Purchase by 2 condition as below, refer accuracy_df for proportion of returning.
#
# Usage:
#    Input: -df: dataframe [customer_id, date_order, order_id, ..any..]
#    Output: result of stats predict [customer_id, pred_standing_date, nextpurchase_day, from_day, to_day, from_date, to_date, accuracy_stats, 'cus_type']
#

def predict_customer_frequent_purchase_active(df):

  print_debug("####################################################")
  print_debug("# GET CUSTOMERS WITH FREQUENT PURCHASE")
  print_debug("####################################################")
  '''
  df['year_month'] = df['date_order'].dt.strftime("%Y-%m")
  
  # CREATE PIVOT TABLE BY MONTH ALL READY
  df_order = df.groupby(['customer_id','year_month']).order_id.count().reset_index()
  df_order['have_order'] = df_order['order_id'].apply(lambda x: 1 if x>=1 else 0)
  
  # day
  df_pivot = pd.pivot_table(df_order, values='have_order', index=['customer_id'], columns=['year_month'], aggfunc=np.sum, fill_value=0)
  #accuracy_dict = freq_purchase_accuracy(df_pivot)
  #print_debug ("[predict_customer_frequent_purchase_active()] NUM_PERIOD = {} -- accuracy_dict =\n{}".format(1,accuracy_dict))
  '''  
  
  df['year_month_day'] = df['date_order'].dt.strftime("%Y-%m-%d")  
  df_order = df.groupby(['customer_id','year_month_day']).order_id.count().reset_index()
  df_order['have_order'] = df_order['order_id'].apply(lambda x: 1 if x>=1 else 0)  
  df_order = df_order[['customer_id','year_month_day','have_order']]
  print ("df_order = {}".format(df_order))
  
  #################################################
  #  Pivot by Day and Group BASE_TIME_BLOCK
  #################################################
  df_pivot = process_df_pivot(df_order)
  print (df_pivot)
  # this is the based pivot, below continue group of BASE TIME BLOCK
  ##############
  
  df_group_acc_stats = pd.DataFrame(columns=['group_period','num_period_cont','accuracy_stats'])
  df_cus_group_freq_pred = pd.DataFrame(columns=['customer_id','group_period','num_period_cont'])
  
  # loop with multiple diff NUM_PERIOD periods for check accuracy of prediction method
  for NUM_PERIOD in np.arange(1,int(TOTAL_DAY_OF_DATA/4),1):
  
    ######################################################
    # IN CASE of THIS NUM_PERIOD: STATISTIC ACCURACY
    ######################################################
    
    # create dataframe for different PERIOD OF TIME UNIT (eg.DAY)
    print_debug ("\n\n*****************************************\n*** NUM_PERIOD = {} ***".format(NUM_PERIOD))
    df_pivot_n_period = create_cus_purchase_mark_by_any_n_period_ofbasetime(df_pivot,NUM_PERIOD)
    df_pivot_n_period.to_csv(stats_model_path+"/df_pivot_group_"+str(NUM_PERIOD)+".csv")
    
	#df_pivot_n_period_c = df_pivot_n_period.copy()
    print_debug ("[predict_customer_frequent_purchase_active()] df_pivot_n_period =\n{}".format(df_pivot_n_period))
	
    if (not os.path.isfile(stats_model_path+"/df_group_acc_stats.csv")) or (stats_update):
	  
      # input dataframe marked customer purchase status in TIME_PERIOD to statistics function
      accuracy_dict = freq_purchase_accuracy(df_pivot_n_period)
      print_debug ("[predict_customer_frequent_purchase_active()] accuracy_dict =\n{}".format(accuracy_dict))
      # convert dict containing accuracy statistics by num continuous purchase into dataframe
      df_g_acc_i = pd.DataFrame({'group_period': np.array([NUM_PERIOD]*len(accuracy_dict)), 'num_period_cont': list(accuracy_dict.keys()), 'accuracy_stats': list(accuracy_dict.values())})
      # concat new result of NUM_PERIOD i to general df
      df_group_acc_stats = pd.concat([df_group_acc_stats,df_g_acc_i])

  if (os.path.isfile(stats_model_path+"/df_group_acc_stats.csv")) and (not stats_update):
    df_group_acc_stats = pd.read_csv(stats_model_path+"/df_group_acc_stats.csv",index_col=None)
  else:
    #####################################################################
    # SAVE FILE: STATISTICS RESULT: GROUP_PERIOD and NUM_CONT_PURCHASES
    ######################################################################
    #df_group_acc_stats: keeps all information about continuous purchasing of customer and accuracy statistics of diff. period group
    print_debug ("[predict_customer_frequent_purchase_active()] df_group_acc_stats = \n{}".format(df_group_acc_stats))
    df_group_acc_stats.to_csv(stats_model_path+"/df_group_acc_stats.csv",index=False)  
	  
  #####################################
  # Stats Prediction
  #####################################
  for NUM_PERIOD in np.arange(1,int(TOTAL_DAY_OF_DATA/4),1):
  
    #######################################################
    # IN CASE of THIS NUM_PERIOD: PREDICT for CUSTOMER
    #######################################################
    df_pivot_n_period_c = pd.read_csv(stats_model_path+"/df_pivot_group_"+str(NUM_PERIOD)+".csv",index_col=None)
    # predict by statistic method
    df_cus_group_freq_pred_i = predict_nextpurchase_thro_freq_pur_stats_method(df_pivot_n_period_c,NUM_PERIOD)
    # update to general dataframe
    df_cus_group_freq_pred = pd.concat([df_cus_group_freq_pred,df_cus_group_freq_pred_i]) 
  
  ######################################################################
  # SAVE FILE: CUSTOMER_ID, LAST CONT_PURCHASE AND PREDICT RESULT
  ######################################################################
  print_debug ("[predict_customer_frequent_purchase_active()] df_cus_group_freq_pred = \n{}".format(df_cus_group_freq_pred))
  df_cus_group_freq_pred = pd.merge(df_cus_group_freq_pred, df_group_acc_stats,on=['group_period','num_period_cont'],how='left')
  
  print_debug ("[predict_customer_frequent_purchase_active()] df_cus_group_freq_pred = \n{}".format(df_cus_group_freq_pred))

  # save to .csv file all customers with different GROUP_PREDIODs and accuracy estimation respectively
  df_cus_group_freq_pred.to_csv(stats_model_path+"/freq_customer_last_active_cases.csv",index=False)
  
  ######################################################################
  # FROM FORMAT OF NEXT PURCHASE PREDICT RESULT
  ######################################################################
  # Condition for choosing Frequent Purchase Customer 
  #   [1] Choose GROUP_PERIOD persuading condition: < 2*PRERROR+1 (to persuade the condition predict output in range of [-PERROR,PERROR])
  #   [2] Number of continuous purchasing periods should be from 2.
  #   [3] Choose GROUP_PERIOD having max accuracy
  
  df_cus_filter = df_cus_group_freq_pred[(df_cus_group_freq_pred.group_period <= (2*PRERROR/BASE_TIME_BLOCK+1)) & (df_cus_group_freq_pred.num_period_cont >=2)]
  df_cus_stats_pred_consider = df_cus_filter.groupby(['customer_id']).accuracy_stats.max().reset_index()
  
  df_cus_stats_pred_consider = pd.merge(df_cus_stats_pred_consider,df_cus_filter, on=['customer_id','accuracy_stats'],how='left')
     
  ######################################################
  # Create Stats Prediction Output dataframe
  
  # compute 'nextpurchase_day_pred', 'from_day', 'to_day'
  df_cus_stats_pred_consider.loc[:,'nextpurchase_day_pred'] = ((df_cus_stats_pred_consider.loc[:,'group_period'].astype(int))*BASE_TIME_BLOCK/2).astype(int)
  df_cus_stats_pred_consider.loc[:,'from_day'] = 0
  df_cus_stats_pred_consider.loc[:,'to_day'] = df_cus_stats_pred_consider.loc[:,'group_period']*BASE_TIME_BLOCK
  
  # add 'pred_standing_date'
  df_cus_stats_pred_consider.loc[:,'pred_standing_date'] = CURRENT_DATE
  df_cus_stats_pred_consider.reset_index(drop=True, inplace=True)
  
  # compute 'from_date', 'to_date'
  df_cus_stats_pred_consider.loc[:,'from_date'] = pd.to_datetime(df_cus_stats_pred_consider.loc[:,'pred_standing_date']) + pd.to_timedelta(df_cus_stats_pred_consider.loc[:,'from_day'], unit='days')
  df_cus_stats_pred_consider.loc[:,'to_date'] = pd.to_datetime(df_cus_stats_pred_consider.loc[:,'pred_standing_date']) + pd.to_timedelta(df_cus_stats_pred_consider.loc[:,'to_day'], unit='days')
  
  # add 'cus_type' for this group is 3
  df_cus_stats_pred_consider.loc[:,'cus_type'] = int(3)
  
  column = ['customer_id','pred_standing_date','nextpurchase_day_pred','from_day','to_day','from_date','to_date','accuracy_stats','cus_type']
  df_cus_stats_pred_consider = df_cus_stats_pred_consider[column]
  ######################################################
  
  df_cus_stats_pred_consider.to_csv(predict_output+"/stats_freq_customer_PRED_nextpurchase.csv",index=False)
  
  return True
  
##################################################################################
# Function: data_pre_processing(df)
# Description:
#    - preprocesing input data, remove particular cases which may be not the normal customer
#    - things need removing: refer as below
#          -- order lines have amount less than 0 (it is lines of return, refund, remove them but still keep the root order)
#          -- remove customer having more than 2 orders per day (it is not real customer)
#          -- remove customer having frequency of purchasing greater greater than 200 (not real customer or so usuall purchase, not need to forecast)
# Usage:
#    Input: -df: dataframe [customer_id, date_order, order_id, ..any..]
#    Output: return filtered dataframe  [customer_id, date_order, order_id, ..any..]
#

def data_pre_processing(df):

  print_debug("####################################################")
  print_debug("# FILTER UNNORMAL CUSTOMERS")
  print_debug("####################################################")
  
  # REMOVE customer having category isnull
  df_cate = df[df.category.isnull()]
  df = df[~df.customer_id.isin(df_cate.customer_id)]
  
  #df['date_order']=pd.to_datetime(df.date_order.values).strftime("%Y-%m-%d")

  # REMOVE amount < 0 ; amount < 0 is refund bill. Ignore
  df = df[df.amount>=0]
  #df['date_order'] = pd.to_datetime(df.date_order)
  print_debug ("[INFO: data_processing()] Get df.amount>0 only: df = {}\n{}".format(df.shape,df[:5]))

  #[1.3] remove customer order_id/day >=3
  df_cus_order_id = df.groupby(['customer_id','date_order','order_id']).amount.sum().reset_index()
  df_cus_order_id_cnt = df_cus_order_id.groupby(['customer_id','date_order']).order_id.count().reset_index()
  df_cus_order_id_cnt_gte3 = df_cus_order_id_cnt[df_cus_order_id_cnt.order_id >= PUR_TIME_PER_DAY_UNNORMAL_THRESHOLD]
  df = df[~df.customer_id.isin(df_cus_order_id_cnt_gte3.customer_id)]
  print_debug ("[INFO: data_processing()] Remove customer order_id/day >= 3: df = {}\n{}".format(df.shape,df[:5]))

 
  #[1.4] remove freq > 200
  df_cus_f = df.groupby(['customer_id','date_order']).amount.sum().reset_index()
  df_cus_f_cnt = df_cus_f.groupby(['customer_id']).date_order.count().reset_index()
  df_cus_f_cnt.columns = ['customer_id','f']
  
  df_cus_f_gt_1 = df_cus_f_cnt[df_cus_f_cnt.f<TOTAL_PUR_DAY_UNNORMAL_THRESHOLD] 
  df = df[df.customer_id.isin(df_cus_f_gt_1.customer_id)]
  
  print_debug ("[INFO: data_processing()] Remove customer having freq > 200: df = {}\n{}".format(df.shape,df[:5]))
  print_report ("[data_pre_processing()] [INFO] df out of this function have the shape is {}".format(df[:5]))
  print_report ("[data_pre_processing()] [INFO] Number of customer at this step: {}".format(df.customer_id.nunique()))  
  
  return df

##################################################################################
# Function: data_pre_processing_ml1(dfa, SPLIT_TIME, PREDICT_MODE)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1()
#    - pre-processing data to get file of data pursuading condition below
#           -- customer having at least 02 purchases before split date, 01 purchase after split date
#      those customers will be used to train the model
# Usage:
#    Input:  - dfa: dataframe [customer_id, date_order, order_id, ..any..]
#            - SPLIT_TIME: time where data are splited
#            - PREDICT_MODE: True/False
#    Output: - return two dataframes:
#                 - df[customer_id, date_order, order_id, ..any..]
#                 - df[customer_id, f_bf, f_af]
#

def data_pre_processing_ml1(dfa, SPLIT_TIME, PREDICT_MODE=False):
  
  print_debug("####################################################")
  print_debug("# PREPARING DATA FOR MODEL 1")
  print_debug("####################################################")
    
  # group 'amount' following 'customer_id' by 'day_order'
  dfa_cus = dfa.groupby(['customer_id','date_order']).amount.sum().reset_index()

  # split data before split date and after split date
  dfa_cus_bf = dfa_cus[dfa_cus.date_order <= SPLIT_TIME]
  dfa_cus_af = dfa_cus[dfa_cus.date_order > SPLIT_TIME]

  # create new table holding information by customer_id, to store feature into...
  dfa_cus_f_bfaf = pd.DataFrame(columns = ['customer_id'])
  dfa_cus_f_bfaf['customer_id'] = np.asarray(dfa_cus.customer_id.unique())

  # Get F (before split date): count number of different purchase_date before split date
  dfa_cus_bf_cnt = dfa_cus_bf.groupby(['customer_id']).date_order.count().reset_index()
  dfa_cus_bf_cnt.columns = ['customer_id','f_bf']

  # Get F (after split date): count number of different purchase_date after split date
  dfa_cus_af_cnt = dfa_cus_af.groupby(['customer_id']).date_order.count().reset_index()
  dfa_cus_af_cnt.columns = ['customer_id','f_af']

  # Form dataframe [customer_id, f_bf, f_af]
  dfa_cus_f_bfaf = pd.merge(dfa_cus_f_bfaf,dfa_cus_bf_cnt,on='customer_id',how='left')
  dfa_cus_f_bfaf = pd.merge(dfa_cus_f_bfaf,dfa_cus_af_cnt,on='customer_id',how='left')

  if not PREDICT_MODE:
    ### [1.6] get customer having at least 02 purchases before split date, 01 purchase after split date
    # remove customer_id having null in f_bf and f_af (because those customer do not adapt [1.6]
    dfa_temp1 = dfa_cus_f_bfaf.loc[~dfa_cus_f_bfaf.f_bf.isnull(),:]
    dfa_temp2 = dfa_temp1.loc[~dfa_cus_f_bfaf.f_af.isnull(),:]

    # get file of customer_id which adapt [1.6]
    dfa_cus_bf_gte2 = dfa_temp2[dfa_temp2.f_bf>=2]
    print_debug ("[data_pre_processing_ml1()][INFO] number of customer having 2 purchases before split date, 01 purchase after split date is {}\n".format(dfa_cus_bf_gte2.shape[0]))
  
    #----------------------------------------------------------------
    #dfa_cus_bf_gte2: final customer file persuading condition [1.6]
    #----------------------------------------------------------------
    # Get original data of customer purchasing  
    dfa_train_ml1 = dfa[dfa.customer_id.isin(dfa_cus_bf_gte2.customer_id)]
    dfa_train_ml1.loc[:,'date_order'] = pd.to_datetime(dfa_train_ml1['date_order'])

    print_debug ("[data_pre_processing_ml1()][OUTPUT] customer purchase data file for train model 1: dfa_train_ml1 {}\n{}".format(dfa_train_ml1.shape,dfa_train_ml1[:5]))
    print_debug ("[data_pre_processing_ml1()][OUTPUT] customer and frequency of purchase >= 2: dfa_cus_bf_gte2 {}\n{}".format(dfa_cus_bf_gte2.shape,dfa_cus_bf_gte2[:5]))

    return dfa_train_ml1, dfa_cus_bf_gte2
  
  # PREDICT_MODE
  else:
    dfa_temp = dfa_cus_f_bfaf[~dfa_cus_f_bfaf.f_bf.isnull()]
    dfa_cus_bf_gte2 = dfa_temp[dfa_temp.f_bf>=2]
    print_debug ("[data_pre_processing_ml1()][INFO] number of customer having 2 purchases before split date, 01 purchase after split date is {}\n".format(dfa_cus_bf_gte2.shape[0]))
    
    # Get original data of customer purchasing  
    dfa_predict_ml1 = dfa[dfa.customer_id.isin(dfa_cus_bf_gte2.customer_id)]
    dfa_predict_ml1.loc[:,'date_order'] = pd.to_datetime(dfa_predict_ml1['date_order'])
    
    print_report ("[data_pre_processing_ml1()] [INFO] df out of this function have the shape is {}".format(dfa_predict_ml1.shape))
    print_report ("[data_pre_processing_ml1()] [INFO] Number of customer after this step: {}".format(dfa_predict_ml1.customer_id.nunique()))  
    
    return dfa_predict_ml1, dfa_cus_bf_gte2
        
##################################################################################
# Function: create_features_ml1(dfa_train_ml1, dfa_cus_bf_gte2, SPLIT_TIME, mode)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1()
#    - [1] Split data into [data before split date, data after split date]
#    - [2] Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchaseday(y), y_label ]             
#          (Note: Number of features inputing to model for training model can be adjusted at the training period)
#          [2.1] R, F, M Computation
#          [2.2] R cluster, F cluster, M cluster
#          [2.3] daydiff_mean, daydiff_std 
#          [2.4] nextpurchaseday(y), y_label
# Usage:
#    Input: - dfa_train_ml1:   df[customer_id, date_order, order_id, ..any..]
#           - dfa_cus_bf_gte2: df[customer_id, f_bf, f_af]
#           - SPLIT_TIME: is the date which split data for train and test if not predict mode
#           - mode: ['TRAIN', 'TEST', 'PREDICT']
# 
#    Output:   mode = 'TRAIN'  : dataframe df_features [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day, y_label]
#              mode = 'TEST'   : dataframe df_features [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day, y_label]
#    (default) mode = 'PREDICT':  dataframe df_features [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std]

def create_features_ml1(dfa_train_ml1, dfa_cus_bf_gte2, SPLIT_TIME, mode='PREDICT'):

  print_debug("####################################################")
  print_debug("# GENERATE FEATURES FOR MODEL")
  print_debug("####################################################")
  
  # [1] Split data into [data before split date, data after split date]
  dfa_train_ml1_bf = dfa_train_ml1[dfa_train_ml1.date_order <= SPLIT_TIME]
  dfa_train_ml1_af = dfa_train_ml1[dfa_train_ml1.date_order > SPLIT_TIME]

  # [2] Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchaseday(y), y_label ]  
  ###################################
  # [2.1] R, F, M Computation  
  # df_features: [customer_id, r, f, m]
  
  # Column F
  # get customer_id and frequency of purchase [before sppit date]
  df_features = dfa_cus_bf_gte2[['customer_id','f_bf']]
  df_features.columns = ['customer_id','f']
  
  # Column M 
  # sum amount in date order
  dfa_train_ml1_bf_cus = dfa_train_ml1_bf.groupby(['customer_id','date_order']).amount.sum().reset_index()
  # sum total amount of a customer
  df_train_final_bf_cus_m = dfa_train_ml1_bf_cus.groupby(['customer_id']).amount.sum().reset_index()
  
  df_features = pd.merge(df_features,df_train_final_bf_cus_m,on='customer_id',how='left')
  df_features.columns = ['customer_id','f','m']
  #df_features['m'] = round(df_features['m'],0).astype(int)
    
  # Column R 
  dfa_train_ml1_bf_cus_r = dfa_train_ml1_bf.groupby(['customer_id']).date_order.max().reset_index()
  dfa_train_ml1_bf_cus_r.columns = ['customer_id','last_buy']
  
  dfa_train_ml1_bf_cus_r.loc[:,'split_date'] = [SPLIT_TIME] * len(df_features)
  dfa_train_ml1_bf_cus_r.loc[:,'split_date'] = pd.to_datetime(dfa_train_ml1_bf_cus_r['split_date'])
  dfa_train_ml1_bf_cus_r.loc[:,'r'] = (dfa_train_ml1_bf_cus_r.split_date - dfa_train_ml1_bf_cus_r.last_buy).dt.days
  
  df_features = pd.merge(df_features,dfa_train_ml1_bf_cus_r[['customer_id','r']],on='customer_id',how='left')
  
  # Column Gender
  #df_gender = dfa_train_ml1_bf.groupby(['customer_id']).gender.first().reset_index()
  #df_features = pd.merge(df_features,df_gender[['customer_id','gender']],on='customer_id',how='left')
 
  # Column VIP
  #df_vip = dfa_train_ml1_bf.groupby(['customer_id']).vip.first().reset_index()
  #df_features = pd.merge(df_features,df_vip[['customer_id','vip']],on='customer_id',how='left') 
  
  ###################################
  # [2.2] R cluster, F cluster, M cluster  
  if mode=='TRAIN':
    df_r_clus = kmeans_cluster(df_features[['customer_id','r']], R_kcluster, False, savemodel_path+'/kmeans_r_model.sav')
    df_f_clus = kmeans_cluster(df_features[['customer_id','f']], F_kcluster, True, savemodel_path+'/kmeans_f_model.sav')
    df_m_clus = kmeans_cluster(df_features[['customer_id','m']], M_kcluster, True, savemodel_path+'/kmeans_m_model.sav')
    df_features = pd.merge(df_features,df_r_clus[['customer_id','r_clus']],on='customer_id',how='left')
    df_features = pd.merge(df_features,df_f_clus[['customer_id','f_clus']],on='customer_id',how='left')
    df_features = pd.merge(df_features,df_m_clus[['customer_id','m_clus']],on='customer_id',how='left')
  else:
    kmeans=pickle.load(open(savemodel_path+'/kmeans_r_model.sav','rb'))
    df_features['r_clus'] = kmeans.predict(np.array(df_features.r.values).reshape(-1,1))
    kmeans=pickle.load(open(savemodel_path+'/kmeans_f_model.sav','rb'))
    df_features['f_clus'] = kmeans.predict(np.array(df_features.f.values).reshape(-1,1))
    kmeans=pickle.load(open(savemodel_path+'/kmeans_m_model.sav','rb'))
    df_features['m_clus'] = kmeans.predict(np.array(df_features.m.values).reshape(-1,1))
  
  
  ###################################
  # [2.3] daydiff_mean, daydiff_std
  dfa_diffbuy = dfa_train_ml1_bf_cus.sort_values(['customer_id','date_order'])
  dfa_diffbuy['prev_order'] = dfa_diffbuy.groupby('customer_id')['date_order'].shift(1)
  dfa_diffbuy['daydiff'] = (dfa_diffbuy['date_order'] - dfa_diffbuy['prev_order']).dt.days

  dfa_mean = dfa_diffbuy[~dfa_diffbuy.daydiff.isnull()].groupby('customer_id').daydiff.mean().reset_index()
  dfa_mean.columns = ['customer_id','daydiff_mean']
  
  dfa_std = dfa_diffbuy[~dfa_diffbuy.daydiff.isnull()].groupby('customer_id').daydiff.std().reset_index()
  dfa_std.fillna(0,inplace=True)
  dfa_std.columns = ['customer_id','daydiff_std']  

  df_features = pd.merge(df_features,dfa_mean,on='customer_id',how='left')
  df_features = pd.merge(df_features,dfa_std,on='customer_id',how='left')

  df_features['daydiff_mean'] = round(df_features['daydiff_mean'],0).astype(int)
  df_features['daydiff_std']  = round(df_features['daydiff_std'],0).astype(int)  
  
  if not (mode=='PREDICT'):
    ###################################  
    # [2.4] nextpurchaseday(y), y_label 
    # last purchase day before split date
    df_bf_max = dfa_train_ml1_bf_cus.groupby(['customer_id']).date_order.max().reset_index()
    df_bf_max.columns = ['customer_id','bf_max']  

    # first purchase day after split date
    dfa_train_ml1_af_cus = dfa_train_ml1_af.groupby(['customer_id','date_order']).amount.sum().reset_index()
    df_af_min = dfa_train_ml1_af_cus.groupby(['customer_id']).date_order.min().reset_index()
    df_af_min.columns = ['customer_id','af_min']

    # combine last before and first after split date
    df_bfaf_max_min = pd.merge(df_bf_max,df_af_min,on='customer_id',how='left')
    # Compute NextPurchaseDay 
    df_bfaf_max_min['nextpurchase_day'] = (df_bfaf_max_min.af_min - df_bfaf_max_min.bf_max).dt.days
  
    # Column: nextpurchase_day 
    df_features = pd.merge(df_features,df_bfaf_max_min[['customer_id','nextpurchase_day']],on='customer_id',how='left')
    
    # Column: next_purchase_label or y_label
    df_features['y_label'] = df_features.nextpurchase_day.values
    for ind, val in enumerate(np.arange(0,PREDICT_LEN,CLUSTER_PERIOD)):
      #df_features['y_label'] = 1
      df_features.loc[df_features.nextpurchase_day>val,'y_label'] = ind+1

  #corr = df_features[df_features.columns].corr()
  #plt.figure(figsize = (12,10))
  #sns_plot = sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f",cmap='coolwarm_r')
  #sns_plot.savefig("./ml1_features_correlation_heatmap.png")
  
  print_report ("[create_features_ml1()] [INFO] df_features_ml1 has the shape: {}".format(df_features.shape))
  print_report ("[create_features_ml1()] [INFO] Number of customer in df_features: {}".format(df_features.customer_id.nunique()))  
  print_report ("[create_features_ml1()] [INFO] df_features review:\n{}".format(df_features.head()))  
  
  return df_features
  
##################################################################################
# Function: run_model_test (model, X_test, y_test, filepath)
# Description:
#    - called by train_test_ml1()
#    - compute accuracy and plot chart
# Usage:
#    Input: - model: the model have been compiled
#           - X_test[series], y_test[array]: data in series to input to model
#           - filepath: the path where we can get the saved model
#    Output: no return
#  

def plot_pred_test_result (y_pred, y_test, filepath):
  
  #print_debug("[plot_pred_test_result()] Predict for X_test...")
  #print_debug ("[plot_pred_test_result()] y_pred = {} \n type = {}".format(y_pred, type(y_pred)))
  #print_debug ("[plot_pred_test_result()] y_test = {} \n type = {}".format(y_test, type(y_test)))    
    
  #y_pred_arr = np.reshape(y_pred,(1,len(y_pred)))[0]
  #y_test_arr = np.reshape(y_test,(1,len(y_test)))[0]
  y_pred_arr = np.array(y_pred)
  y_test_arr = np.array(y_test)
  #print_debug("[plot_pred_test_result()] [Process] y_pred_arr = {} \n type = {}".format(y_pred_arr, type(y_pred_arr)))
  #print_debug("[plot_pred_test_result()] [Process] y_test_arr = {} \n type = {}".format(y_test_arr, type(y_test_arr)))
       
  y_mae = y_pred_arr - y_test_arr
  error = np.absolute(y_mae)/y_test_arr
  
  #print_debug ("[plot_pred_test_result()] [INFO] error = y_pred_arr - y_test_arr = {}".format(error))
  #print_debug ("--> Average of error: {}, whereas Average of Actual value (y_test): {}".format(np.mean(error),np.mean(y_test_arr)))
  
  OUT_STD = PRERROR
  # Compute accuracy
  y_diff = []
  y_diff_bool = []
  y_correct_color = []
  for i in range(len(y_test_arr)):
    y_diff.append(y_test_arr[i] - y_pred_arr[i])
    if((y_diff[-1]<OUT_STD)&(y_diff[-1]>-OUT_STD)):
      y_diff_bool.append(True)
      y_correct_color.append('purple')
    else:
      y_diff_bool.append(False)
      y_correct_color.append('orange')

  result_check_hist = np.unique(y_diff_bool,return_counts=True)
  accuracy = result_check_hist[1][1]/len(y_diff_bool)

  print_debug("[plot_pred_test_result()][RESULT] MODEL EVALUATION WITH X_test data")
  print_debug("[plot_pred_test_result()][RESULT] Predicting for Total number of customer: {}".format(len(y_diff_bool)))
  print_debug("[plot_pred_test_result()][RESULT] Final Result of model after adjust: accuracy = {}".format(accuracy)) 
  #######################################
  # Plot Actual and Predict result chart
  
  x = np.linspace(0,len(y_pred_arr)-1,len(y_pred_arr)).astype(int)
  data_x = {'y_test': y_test_arr, 'y_pred': y_pred_arr, 'y_above': y_test_arr+OUT_STD, 'y_below':y_test_arr-OUT_STD,'y_correct_color':y_correct_color} 

  #df_rs = pd.DataFrame(data=data_x, columns = ['y_test','y_pred'], index=x)
  df_rs = pd.DataFrame(data_x, index=x)
  df_rs.sort_values(['y_test'],ascending=True,inplace=True)    
    
  fig = plt.figure(figsize=(20,4))
  ax = fig.add_subplot(111)
  ax.plot(x, df_rs.y_test, color='blue', linewidth=2)
  ax.scatter(x,df_rs.y_pred , color=df_rs.y_correct_color.values, marker='^')
  ax.scatter(x,df_rs.y_above , color='green', marker='.',linewidth=0.5)
  ax.scatter(x,df_rs.y_below , color='green', marker='.',linewidth=0.5)
  plt.savefig(filepath)


##################################################################################
# Function: train_by_lgb (X_train, y_train, X_test, y_test)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1_skfold() --> train_by_lgb (X_train, y_train)
#    Train model with LightGBM
#
# Usage:
#    Input: - X_train, y_train, , X_test, y_test
#    Output: trained model
#
 
  
def train_by_lgb(X_train, y_train, X_test, y_test): 

  print_debug("***********************************************")
  print_debug("BEGIN TO TRAIN MODEL WITH LIGHTGBM ...")  
  # money
  params = {
          "objective" : "multiclass",
          "num_class" : np.unique(y_train).size+1,
          "num_leaves" : 20,
          "max_depth": 10,
          "boosting_type" : 'gbdt',
          "learning_rate" : 0.005,
          #"bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.8,  # colsample_bytree
          #"bagging_freq" : 5,        # subsample_freq
          #"bagging_seed" : 2018,
          "verbosity" : 0 }
          
  lgtrain, lgval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)
  lgbmodel = lgb.train(params, lgtrain, num_boost_round=4000, valid_sets=[lgtrain, lgval], early_stopping_rounds=30)
  
  print_debug("LightGBM model is trained completely!")
  print_debug("***********************************************")
  return lgbmodel


##################################################################################
# Function: lightgbm_train_transform_rs (X_train, y_train, X_test, y_test, X, skfold)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1_skfold() --> lightgbm_train_transform_rs()
#    - train model by lightGBM
#    - comput accuracy score
#    - transform lightGBM output to y_label_pred
#
# Usage:
#    Input: - X_train, y_train, X_test, y_test, X, skfold
#    Output: model, acc_score, y_label_pred
#

def lightgbm_train_transform_rs(X_train, y_train, X_test, y_test, X, skfold):

    ##############################################################
    # TRAIN LightGBM
    #scores = []
    model = train_by_lgb(X_train, y_train, X_test, y_test)               

    ##############################################################
    # This for test model with test data to get accuracy score        
    # Prediction
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_softmax = np.zeros_like(y_pred)
    y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    y_label_pred = np.argmax(y_pred_softmax, axis=1)  
	
    ###############################################
    # COMPUTE 'acc_score'
    acc_score = accuracy_score(y_test, y_label_pred)

    #print ("y_test={}, y_pred={}".format(y_test.values.flatten(),y_label_pred))
    #a= f1_score(y_test.values.flatten(), y_label_pred, average=None)
    #print(a)
    #print ("y_test={}, y_pred={}".format(y_test,y_label_pred))
    #a= f1_score(y_test, y_label_pred, average=None)
    #print(a)
    

	###############################################
    # COMPUTE 'f1_score_dict'
	
    # replace by this for accuracy computation
    #valid_count = 0
    #print ("len(y_test)={}, len(y_pred)={}".format(len(y_test),len(y_label_pred)))
    #for  in zip(np.unique(y_test_pred,return_counts=True)[0], np.unique(y_test_pred,return_counts=True)[1]):
    ##  str(e) for e in np.unique(y_test_pred,return_counts=True)[0]
    #  if j==k:
    #    valid_count+=1       		
    #acc_score = valid_count/len(y_test)
    #print_debug ("TRAIN --> accuracy_score = {}".format(acc_score))
	
    ######################################################################     
    # COMPUTE 'y_label_pred' of X_test (for calculate and store f1_score)

    data = X if skfold else X_test

    # Predict full data, would like to check accuracy after adjustment in full data
    y_pred = model.predict(data, num_iteration=model.best_iteration) #return proportion of each label
    y_pred_softmax = np.zeros_like(y_pred) 
    y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1 #choose the highest proportion
    y_label_pred = np.argmax(y_pred_softmax, axis=1)  
    
    return model, acc_score, y_label_pred

    
##################################################################################
# Function: train_test_ml1 (df, mode, model_savepath)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1()
#    - Split data into [data before split date, data after split date]
#    - Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
# Usage:
#    Input: - df:   df_features [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day, y_label]
#           - mode: ['TRAIN','TEST','TRAIN_PSEUDO']. TRAIN_PSEUDO: add sample of Pseudo Labeling
#           - model_savepath: path to save model
#    Output: dataframe of cluster result [customer_id, r_clus|f_clus|m_clus]
#

def train_test_ml1 (df, mode, model_savepath): 
 
  modeln = models
  model_savepath = model_savepath+".sav"
  #model_name = model_dict.get(model)
  print_debug("####################################################")
  print_debug("# MODEL 01: TRAIN TEST MODEL ({} MODE)".format(mode))  
  print_debug("####################################################")

  model_temp = g_model
  model = ''
  acc_score = ''
  y_label_pred=''
  PSEUDO = False
  
  #################################
  # Plot corr()
  df_plot = df[['r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std','nextpurchase_day','y_label']]  
  corr = df_plot[df_plot.columns].corr()
  plt.figure(figsize = (12,8))
  sns_plot = sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f",cmap='coolwarm_r')
  sns_fig = sns_plot.get_figure()
  sns_fig.savefig(log_path+"/ml1_features_label_correlation_map.png")
  #################################
  
  X = df[['r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std']]
  y = ''  

  if modeln =='xgbr':
    y = df[['nextpurchase_day']]
  else:
    y = df[['y_label']] 
 
  if mode == 'TRAIN_PSEUDO':
    mode = 'TRAIN'
    PSEUDO = True
  print_report('[train_test_ml1()] [INPUT] Input feature for train model 1\n X = \n {} \n y= \n{} \n X.shape {}, y.shape {}\n'.format(X.head(),y.head(), X.shape, y.shape))
  
  ############################################
  # TRAIN MODE  
  ############################################
  if mode=='TRAIN':
    if modeln =='xgbr':
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
    else:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, stratify=y)
      
    if PSEUDO:
      model_temp = pickle.load(open(model_savepath,'rb'))
    else:
      model_temp = g_model
 
    # LightGBM
    if modeln == 'lgb':
      model, acc_score, y_label_pred = lightgbm_train_transform_rs(X_train, y_train, X_test, y_test, X, skfold)
    # XGboost, DCT, RF	
    else:
      model = model_temp.fit(X_train, y_train)
      acc_score = model.score(X_test,y_test)
      y_label_pred = model.predict(X_test)          
        
    # SAVE MODEL 
    pickle.dump(model,open(model_savepath,'wb'))
    print_debug("[train_test_ml1()] [LightGBM] saved model at {}".format(model_savepath))	
	  
    print_debug("[TRAIN MODE] Model 01 Result ")
    print_debug("[train_test_ml1()] Accuracy Score: {}".format(acc_score))
    print_debug("[train_test_ml1()] Classication Report \n{}".format(classification_report(y_test, y_label_pred)))
    print_debug("[train_test_ml1()] Confusion Matrix \n{}".format(confusion_matrix(y_test, y_label_pred)))
    
    print_report("[TRAIN MODE] Model 01 Result ")
    print_report("[train_test_ml1()] [y_clus] Accuracy Score: {}".format(acc_score))
    print_report("[train_test_ml1()] [y_clus] Classication Report \n{}".format(classification_report(y_test, y_label_pred)))
    print_report("[train_test_ml1()] [y_clus] Confusion Matrix \n{}".format(confusion_matrix(y_test, y_label_pred)))
      
	##############################################
	# SAVE ACCURACY SCORE of EACH LABEL GROUP
     
    # save f1_score of all cluster of current model
    with open(model_path+'/model01_cluster_accuracy_'+modeln+'.txt', 'w') as fw:
      fw.write('#'.join(str(e) for e in np.unique(y_label_pred,return_counts=True)[0])+'\n')
      fw.write('#'.join(str(e) for e in f1_score(y_test, y_label_pred, average=None)))    
  
  ############################################
  # TEST MODE  
  ############################################  
  if (mode=='TEST'):
    #load model
    model=pickle.load(open(model_savepath,'rb'))
	
    if modeln=='lgb':
      y_pred = model.predict(X, num_iteration=model.best_iteration)
      y_pred_softmax = np.zeros_like(y_pred)
      y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
      y_label_pred = np.argmax(y_pred_softmax, axis=1)  
      #print ("y={}, y_pred={}, y.value={},y.flatten={}".format(y,y_label_pred,y.values,y.values.flatten()))
      acc_score = accuracy_score(y, y_label_pred)
    else:
      y_label_pred = model.predict(X)
      acc_score = model.score(X,y)
    
    print_debug("[TEST MODE ] MODEL 1: \n")
    print_debug("[test_ml1()] Accuracy Score: {}".format(acc_score))
    print_debug("[test_ml1()] Classication Report \n{}".format(classification_report(y, y_label_pred)))
    print_debug("[test_ml1()] Confusion Matrix \n{}".format(confusion_matrix(y, y_label_pred)))
    
  
  #######################################################################
  # Adjust output data to compute the next purchase day after split date
  #######################################################################
  
  if modeln=='lgb':
      y_pred = model.predict(X, num_iteration=model.best_iteration)
      y_pred_softmax = np.zeros_like(y_pred)
      y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
      y_label_pred = np.argmax(y_pred_softmax, axis=1)  
      #print ("y={}, y_pred={}, y.value={},y.flatten={}".format(y,y_label_pred,y.values,y.values.flatten()))
      acc_score = accuracy_score(y, y_label_pred)
  else:
      y_label_pred = model.predict(X)
      acc_score = model.score(X,y)

  df['y_label_pred'] = y_label_pred
  print_debug("[FULL DATA] MODEL 1 PREDICT RESULT FOR FULL DATA: \n")
  print_debug("Accuracy Score: {}".format(acc_score))
  print_debug("Classication Report \n{}".format(classification_report(y, y_label_pred)))
  print_debug("Confusion Matrix \n{}".format(confusion_matrix(y, y_label_pred)))
  
  #create num of day after split date the customer could purchase
  if not (modeln =='xgbr'):
    df['nextpurchase_fromsplit_pred'] = (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_fromsplit_pred - (df.nextpurchase_day - df.r)
  else:
    df['nextpurchase_pred'] = df['y_label_pred']
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_pred - df.nextpurchase_day

  ######################
  # set Error Wing is 45
  OUT_STD = PRERROR
  check_arr = []
  cus_std = OUT_STD
  for i in df.index.values:
    #if not (df.loc[i,'daydiff_std']==0):
    #cus_std = OUT_STD
    #else:
    #  cus_std = min(2.5*df.loc[i,'daydiff_std'],OUT_STD)
    
    if ((df.loc[i,'nextpurchase_diff_act_pred']<=cus_std)&(df.loc[i,'nextpurchase_diff_act_pred']>=-cus_std)):
      check_arr.append(True)
    else:
      check_arr.append(False)      
      
  df['result_check'] = check_arr

  result_check_hist = np.unique(df.result_check,return_counts=True)
  accuracy = result_check_hist[1][1]/len(df)
  
  print_debug("[FULL DATA][ADJUST] ADJUST RESULT FOR FINAL")
  print_debug("[ADJUST] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_debug("[ADJUST] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  print_report("[FULL DATA][ADJUST] ADJUST RESULT FOR FINAL")
  print_report("[ADJUST] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_report("[ADJUST] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  
  #######################################
  # Plot Actual and Predict result chart

  # plot for predict next purchase from SPLIT_DATE
  plot_filepath = log_path+"/"+mode+"_"+modeln+"_ACTUAL_and_PREDICT_result_NP_FROM_SPLIT.png"
  plot_pred_test_result (df.nextpurchase_fromsplit_pred.values, df.nextpurchase_day.values - df.r.values, plot_filepath)
  
  # plot for predict next purchase from LAST PURCHASE DAY
  plot_filepath = log_path+"/"+mode+"_"+modeln+"_ACTUAL_and_PREDICT_result_NP_FROM_LAST.png"    
  plot_pred_test_result (df.nextpurchase_fromsplit_pred.values+df.r.values, df.nextpurchase_day.values, plot_filepath)
  
  # plot for predict next purchase from LAST PURCHASED DAY
  #if modeln =='xgbr':
  #  plot_pred_test_result(df.nextpurchase_pred.values, df.nextpurchase_day.values, plot_filepath)
  #else:
  #  plot_pred_test_result(df.nextpurchase_fromsplit_pred.values + df.r.values, df.nextpurchase_day.values, plot_filepath)
  #######################################  
  
  
  if mode=='TRAIN':
    # save to file the accuracy of each cluster
    df_clus_cus_count = df.groupby(['y_label_pred']).customer_id.count().reset_index()
    df_clus_cus_count.columns = ['y_label_pred','customer_qty']
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_cus_count =\n{}".format(df_clus_cus_count))
  
    df_clus_cus_right = df.groupby(['y_label_pred']).result_check.sum().reset_index()
    df_clus_cus_right.columns = ['y_label_pred','right_pred']
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_cus_right =\n{}".format(df_clus_cus_right))

    df_clus_acc = pd.merge(df_clus_cus_count,df_clus_cus_right,on='y_label_pred',how='left')
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc['accuracy'] = df_clus_acc.right_pred/df_clus_acc.customer_qty
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc.to_csv(ml_model_path+"/ml1_y_predict_cluster_accuracy_"+modeln+".csv",index=False)

  ############################################
  # RETRAIN MODEL USING FULL DATA
  ############################################  
  # After train and test, this step train again with full data
  if mode=='TRAIN':
    #if PSEUDO:
    #  modelx = pickle.load(open(model_savepath,'rb'))
    #else:
    #  modelx = xgb.XGBClassifier()
    if modeln=='lgb':
      print_debug("[RETRAIN FULL DATA] LightGBM --> skip this step")
    else:
      model = model_temp.fit(X, y)
      pickle.dump(model,open(model_savepath,'wb'))
	
 
##################################################################################
# Function: train_test_ml1_skfold (df, mode, model_savepath)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1_skfold()
#    - Split data into [data before split date, data after split date]
#    - Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
#      USING Stratified Kfold to split Train and Valid data, run with NFOLD=5, save 5 models for 5 Folds
#      Predict result is Average of 5 model result
#      This helps increasing Model accuracy to 5%
#
# Usage:
#    Input: - df:   df_features [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day, y_label]
#           - mode: ['TRAIN','TEST','TRAIN_PSEUDO']. TRAIN_PSEUDO: add sample of Pseudo Labeling
#           - model_savepath: path to save model
#    Output: dataframe of cluster result [customer_id, r_clus|f_clus|m_clus]
#

def train_test_ml1_skfold (df, mode, model_savepath): 
   
  modeln = models
  #NFOLD = 5
  #model_name = model_dict.get(model)
  print_debug("####################################################")
  print_debug("# MODEL 01: TRAIN TEST MODEL ({} MODE)".format(mode))  
  print_debug("####################################################")

  model_temp = g_model
  model = ''
  accuracy_score = ''
  PSEUDO = False

  #################################
  # Plot corr()
  df_plot = df[['r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std','nextpurchase_day','y_label']]  
  corr = df_plot[df_plot.columns].corr()
  plt.figure(figsize = (12,8))
  sns_plot = sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f",cmap='coolwarm_r')
  sns_fig = sns_plot.get_figure()
  sns_fig.savefig(log_path+"/ml1_features_label_correlation_map.png")
  #################################
  
  X = df[['r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std']]
  y = ''
  if modeln =='xgbr':
    y = df[['nextpurchase_day']]
  else:
    y = df[['y_label']]
    
  if mode == 'TRAIN_PSEUDO':
    mode = 'TRAIN'
    PSEUDO = True

  #set new model or load model
  if PSEUDO:
    model_temp = pickle.load(open(model_savepath,'rb'))
  else:
    model_temp = g_model    
  
  print_report('[train_test_ml1()] [INPUT] Input feature for train model 1\n X = \n {} \n y= \n{} \n X.shape {}, y.shape {}\n'.format(X.head(),y.head(), X.shape, y.shape))
  
  ########################################
  # TRANSFROM STEP FOR SKFOLD
  X = X.to_numpy()
  y = y.to_numpy()
  y = y.reshape((len(y),))  
  print_report('[train_test_ml1()] [LightGBM] Transformed input\n X.shape {}, y.shape {}\n'.format(X.shape, y.shape))
 
  scores = []
  y_label_pred=[]
  ################################################
  #Stratified Kfold
  if mode=='TRAIN':  
    skf = StratifiedKFold(n_splits=NFOLD, random_state=10, shuffle=True)
    
    i = 0
    df['nextpurchase_fromsplit_pred'] = 0
    
    # loop for train
    for train_index, test_index in skf.split(X, y):
      i+=1
      model_savepath_skf = model_savepath+"_skf_"+str(i)+".sav"
      print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      
      if modeln == 'lgb':
        model, acc_score, y_label_pred = lightgbm_train_transform_rs(X_train, y_train, X_test, y_test, X, skfold)
        
      else:
        model = model_temp.fit(X_train, y_train)
        acc_score = model.score(X_test,y_test)
        scores.append(acc_score)
        y_label_pred = model.predict(X)          
        
      # SAVE MODEL and TEST
      pickle.dump(model,open(model_savepath_skf,'wb'))
      print_debug("[train_test_ml1()][SKFold] [{}] saved model at {}".format(i,model_savepath_skf))

      # Append score result of each SKfold     
      print_debug("[ml1] skfold {}: acc_score = {}".format(i,acc_score))	  
      scores.append(acc_score)
      
      ##################################
      # create 'y_label_pred' for accuracy report when predicting, using last skfold for y_label_predict
      df['y_label_pred'] = y_label_pred
      df['nextpurchase_fromsplit_pred'] = df.nextpurchase_fromsplit_pred + (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)

    ##<-> 
    df['nextpurchase_fromsplit_pred'] = round(df.nextpurchase_fromsplit_pred/NFOLD,0).astype(int) 
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_fromsplit_pred - (df.nextpurchase_day - df.r)
    accuracy_score = np.mean(scores)
    
    print_debug('[train_test_ml1()] [INFO] Accuracy of stratifiedKfold: {}, \n --> Average Accuracy: {} '.format(scores, accuracy_score))
    print_report('[train_test_ml1()] [INFO] Accuracy of stratifiedKfold: {}, \n --> Average Accuracy: {} '.format(scores, accuracy_score))
                
      
  ################################################
  #Stratified Kfold
  if mode=='TEST':
    df['nextpurchase_fromsplit_pred'] = 0
    for i in np.arange(1,NFOLD+1):
      model_savepath_skf = model_savepath+"_skf_"+str(i)+".sav"
      model=pickle.load(open(model_savepath_skf,'rb'))
      
      # LightGBM
      if modeln == 'lgb':
        ##############################################################
        # This for test model with test data to get accuracy score        
        # Prediction
        y_pred = model.predict(X, num_iteration=model.best_iteration)

        y_pred_softmax = np.zeros_like(y_pred)
        y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
        y_label_pred = np.argmax(y_pred_softmax, axis=1)  
    
        ###############################################
        # replace by this for accuracy computation
        valid_count = 0
        #print ("len(y_test)={}, len(y_pred)={}".format(len(y_test),len(y_label_pred)))
        for j,k in zip(y, y_label_pred):
          if j==k:
            valid_count+=1        
        acc_score = valid_count/len(y)
        ###############################################
        #scores.append(acc_score)        
        print_debug ("TEST skfold {}--> accuracy_score = {}".format(i, acc_score))
                    
      # XGBoost, DCT, RF, SVC            
      else:
        y_label_pred = model.predict(X)
        acc_score = model.score(X,y)

      scores.append(acc_score)  
      df['y_label_pred'] = y_label_pred
      df['nextpurchase_fromsplit_pred'] = df.nextpurchase_fromsplit_pred + (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)
      
    ##<->  
    df['nextpurchase_fromsplit_pred'] = round(df.nextpurchase_fromsplit_pred/NFOLD,0).astype(int) 
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_fromsplit_pred - (df.nextpurchase_day - df.r)
    accuracy_score = np.mean(scores)
    
    print_debug("#################################################################")    
    print_debug("[test_ml1()] MODEL 1 RESULT FOR dftest:")
    print_debug("[test_ml1()] Classication Report \n{}".format(classification_report(y, y_label_pred)))
    print_debug("[test_ml1()] Confusion Matrix \n{}".format(confusion_matrix(y, y_label_pred)))
    print_debug("[test_ml1()] Accuracy Score: {}".format(accuracy_score))
    print_debug("#################################################################")    
    
  ######################
  # set Error Wing is PRERROR
  OUT_STD = PRERROR
  check_arr = []
  cus_std = OUT_STD
  for i in df.index.values:
    #if not (df.loc[i,'daydiff_std']==0):
    #cus_std = OUT_STD
    #else:
    #  cus_std = min(2.5*df.loc[i,'daydiff_std'],OUT_STD)
    
    if ((df.loc[i,'nextpurchase_diff_act_pred']<=cus_std)&(df.loc[i,'nextpurchase_diff_act_pred']>=-cus_std)):
      check_arr.append(True)
    else:
      check_arr.append(False)      
    
  df['result_check'] = check_arr

  result_check_hist = np.unique(df.result_check,return_counts=True)
  accuracy = result_check_hist[1][1]/len(df)
  
  print_debug("[train_test_ml1()][RESULT] ADJUST RESULT FOR FINAL")
  print_debug("[train_test_ml1()][RESULT] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_debug("[train_test_ml1()][RESULT] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  print_report("[train_test_ml1()][Adjust] ADJUST RESULT FOR FINAL")
  print_report("[train_test_ml1()][Adjust] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_report("[train_test_ml1()][Adjust] Final Result of model after adjust: accuracy = {}".format(accuracy))
    
  #######################################
  # Plot Actual and Predict result chart

  # plot for predict next purchase from SPLIT_DATE
  plot_filepath = log_path+"/"+mode+"_"+modeln+"_ACTUAL_and_PREDICT_result_NP_FROM_SPLIT.png"
  plot_pred_test_result (df.nextpurchase_fromsplit_pred.values, df.nextpurchase_day.values - df.r.values, plot_filepath)
  
  # plot for predict next purchase from LAST PURCHASE DAY
  plot_filepath = log_path+"/"+mode+"_"+modeln+"_ACTUAL_and_PREDICT_result_NP_FROM_LAST.png"    
  plot_pred_test_result (df.nextpurchase_fromsplit_pred.values+df.r.values, df.nextpurchase_day.values, plot_filepath)
  
  # plot for predict next purchase from LAST PURCHASED DAY
  #if modeln =='xgbr':
  #  plot_pred_test_result(df.nextpurchase_pred.values, df.nextpurchase_day.values, plot_filepath)
  #else:
  #  plot_pred_test_result(df.nextpurchase_fromsplit_pred.values + df.r.values, df.nextpurchase_day.values, plot_filepath)
  #######################################  
    
  if mode=='TRAIN':
    # save to file the accuracy of each cluster
    df_clus_cus_count = df.groupby(['y_label_pred']).customer_id.count().reset_index()
    df_clus_cus_count.columns = ['y_label_pred','customer_qty']
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_cus_count =\n{}".format(df_clus_cus_count))
  
    df_clus_cus_right = df.groupby(['y_label_pred']).result_check.sum().reset_index()
    df_clus_cus_right.columns = ['y_label_pred','right_pred']
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_cus_right =\n{}".format(df_clus_cus_right))

    df_clus_acc = pd.merge(df_clus_cus_count,df_clus_cus_right,on='y_label_pred',how='left')
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc['accuracy'] = df_clus_acc.right_pred/df_clus_acc.customer_qty
    print_debug("[train_test_ml1()][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc.to_csv(ml_model_path+"/ml1_y_predict_cluster_accuracy_"+modeln+".csv",index=False)

  # RETRAIN MODEL WITH FULL DATA
  # HOW TO TRAIN ALL DATA BUT STILL KEEP KFOLD SPEC? 
  # --> this version, still train with 80% and valid 20%, can not train all 100% data
  #

##################################################################################
# Function: predict_ml1 (df, model_savepath):
# Description:
#    - data_pre_processing() --> create_features_ml1() --> predict_ml1()
#    - Split data into [data before split date, data after split date]
#    - Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
# Usage:
#    Input: - df:   df_features [customer_id, r, f, m, 'r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std']
#           - model_savepath: path to save model
#    Output: dataframe ['customer_id','y_label_pred','nextpurchase_day_pred']
#

def predict_ml1 (df, model_savepath):

  modeln = models 
  print_debug("####################################################") 
  print_debug("# [Model 1] MODEL 01 PREDICT ")
  print_debug("####################################################")
  print_debug("PREDICT RUN: Model: {}, skfold: {}...".format(modeln, skfold))
  
  X = df[['r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std']]

  #######################################################################
  # Predict and Adjust output data
  if skfold:  

    #transform X
    X = X.to_numpy()
    df['nextpurchase_fromsplit_pred'] = 0
	
    for i in np.arange(1,NFOLD+1):
      
      model_savepath_skf = model_savepath+"_skf_"+str(i)+".sav"      
      model=pickle.load(open(model_savepath_skf,'rb'))
      print_debug("Load model: {}".format(model_savepath_skf))
      #y_label_pred = model.predict(X)
      	  
      # LightGBM
      if modeln == 'lgb':
        ##############################################################
        # This for test model with test data to get accuracy score        
        # Prediction
        y_pred = model.predict(X, num_iteration=model.best_iteration)

        y_pred_softmax = np.zeros_like(y_pred)
        y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
        y_label_pred = np.argmax(y_pred_softmax, axis=1)          
                    
      # XGBoost, DCT, RF, SVC            
      else:
        y_label_pred = model.predict(X)		
	 
      df['y_label_pred'] = y_label_pred
      df['nextpurchase_fromsplit_pred'] = df.nextpurchase_fromsplit_pred + (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)
      print_debug("Complete predict and output process for skfold {}".format(i))  
	  
    df['nextpurchase_day_pred'] = round(df.nextpurchase_fromsplit_pred/NFOLD,0).astype(int) 

  else:
    model_savepath=model_savepath+".sav"
    model=pickle.load(open(model_savepath,'rb'))
	
    # LightGBM
    if modeln == 'lgb':
      ##############################################################
      # This for test model with test data to get accuracy score        
      # Prediction
      y_pred = model.predict(X, num_iteration=model.best_iteration)
      y_pred_softmax = np.zeros_like(y_pred)
      y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
      y_label_pred = np.argmax(y_pred_softmax, axis=1)          
  
    # XGBoost, DCT, RF, SVC            
    else:
      y_label_pred = model.predict(X)

    df['y_label_pred'] = y_label_pred
  
    #create num of day after split date the customer could purchase
    df['nextpurchase_day_pred'] = (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)

  return df[['customer_id','y_label_pred','nextpurchase_day_pred']]

##################################################################################
# Function: slide_split_time_compute_recency (df, INI_SPLIT_TIME, END_DATE, slide_step)
# Description:
#    - called by create_features_ml2() or independent use
#    - slide split time to get df [customer_id, recency]
# Usage:
#    Input : - df:   df[customer_id, date_order, order_id, ..any..]
#            - INI_SPLIT_TIME: initial split_date
#            - END_DATE: last split_date limit
#            - slide_step: moving step of split_time
#    Output:   dataframe[customer_id,'r','nextpurchase_day']
#

def slide_split_time_compute_recency (df, INI_SPLIT_TIME, END_DATE, slide_step):

  SPLIT_TIME = INI_SPLIT_TIME
  
  # Define df_cus_r is an dataframe include ['customer_id','r'] for output
  #df_cus_result = pd.DataFrame(columns = ['customer_id'])
  #df_cus_result['customer_id'] = np.asarray(df.customer_id.unique())
  df_cus_result = pd.DataFrame()
  first = True
  
  while (SPLIT_TIME <= END_DATE):
      
    # group 'amount' following 'customer_id' by 'day_order'
    df_cus = df.groupby(['customer_id','date_order']).amount.sum().reset_index()

    # split data before split date and after split date
    df_cus_bf = df_cus[df_cus.date_order <= SPLIT_TIME]
    df_cus_af = df_cus[df_cus.date_order > SPLIT_TIME]

    # create new table holding information by customer_id, to store feature into...
    df_cus_f_bfaf = pd.DataFrame(columns = ['customer_id'])
    df_cus_f_bfaf['customer_id'] = np.asarray(df_cus.customer_id.unique())

    # date_order before split_time and after split_time
    df_cus_bf_last_order = df_cus_bf.groupby(['customer_id']).date_order.max().reset_index()
    df_cus_bf_last_order.columns = ['customer_id','date_order_bf_max']

    df_cus_af_first_order = df_cus_af.groupby(['customer_id']).date_order.min().reset_index()
    df_cus_af_first_order.columns = ['customer_id','date_order_af_min']
    
    # Get F (before split date): count number of different purchase_date before split date
    df_cus_bf_cnt = df_cus_bf.groupby(['customer_id']).date_order.count().reset_index()
    df_cus_bf_cnt.columns = ['customer_id','f_bf']

    # Get F (after split date): count number of different purchase_date after split date
    df_cus_af_cnt = df_cus_af.groupby(['customer_id']).date_order.count().reset_index()
    df_cus_af_cnt.columns = ['customer_id','f_af']

    # Form dataframe [customer_id, f_bf, f_af]
    df_cus_f_bfaf = pd.merge(df_cus_f_bfaf, df_cus_bf_cnt, on='customer_id',how='left')
    df_cus_f_bfaf = pd.merge(df_cus_f_bfaf, df_cus_af_cnt, on='customer_id',how='left')
    df_cus_f_bfaf = pd.merge(df_cus_f_bfaf, df_cus_bf_last_order, on='customer_id',how='left')
    df_cus_f_bfaf = pd.merge(df_cus_f_bfaf, df_cus_af_first_order, on='customer_id',how='left')
    
    # Compute recency of first purchase of each customer. Find customer having 1 purchase before split_time, more than one after split_time
    df_cus_firstbuy = df_cus_f_bfaf[(df_cus_f_bfaf.f_bf==1)&(df_cus_f_bfaf.f_af>=1)]
    
    # Compute R and Nextpurchase_day
    df_cus_firstbuy.loc[:,'split_date'] = [SPLIT_TIME] * len(df_cus_firstbuy)
    df_cus_firstbuy.loc[:,'split_date'] = pd.to_datetime(df_cus_firstbuy['split_date'])
    df_cus_firstbuy.loc[:,'r'] = (df_cus_firstbuy.split_date - df_cus_firstbuy.date_order_bf_max).dt.days
    df_cus_firstbuy.loc[:,'nextpurchase_day'] = (df_cus_firstbuy.date_order_af_min - df_cus_firstbuy.date_order_bf_max).dt.days

    # update data to df_cus_result
    if first:
      df_cus_result = df_cus_firstbuy.copy()
      first = False
    else:
      df_cus_result = pd.concat([df_cus_result, df_cus_firstbuy])
    
    print_debug("[slide_split_time_compute_recency] {}: Number of customer are processed in this slide_step: {}, total have {} customers".format(SPLIT_TIME,df_cus_firstbuy.customer_id.nunique(),df_cus_result.customer_id.nunique()))
    # upgrade df for next run of new split_time
    df = df[~df.customer_id.isin(df_cus_firstbuy.customer_id)]    

    SPLIT_TIME += datetime.timedelta(days=slide_step)     
    #SPLIT_TIME = pd.to_datetime(SPLIT_TIME).strftime('%Y-%m-%d')
    #print_debug('[debug] {}  {}'.format(type(SPLIT_TIME),type(END_DATE)))

  return df_cus_result[['customer_id','r','nextpurchase_day']]


def create_features_of_one_purchase_customer_for_predict (df, SPLIT_TIME):
  
  # get customer having 1 purchase
  df_cus = df.groupby(['customer_id','date_order']).amount.sum().reset_index()
  df_cus_cnt = df_cus.groupby(['customer_id']).date_order.count().reset_index()
  df_cus_cnt_1p = df_cus_cnt[df_cus_cnt.date_order==1]
  df = df[df.customer_id.isin(df_cus_cnt_1p.customer_id)]
  
  #
  df_feature_ml2 = df.groupby(['customer_id','date_order']).amount.sum().reset_index()
  df_feature_ml2.columns =['customer_id','date_order','m']

  # check input data if customer having one purchase or not
  if df_feature_ml2.shape[0] == df.customer_id.nunique():
    print_debug("[INFO] Input data is right, all customer have only one purchase. Number of customer is: {}".format(df_feature_ml2.shape[0]))
  else:
    print_debug("[WARNING] STILL HAVING CUSTOMER WITH MORE THAN 1 PURCHASE DAY")
    sys.exit(5)

  #compute R
  df_feature_ml2['split_time'] = [SPLIT_TIME] * len(df_feature_ml2)
  df_feature_ml2['split_time'] = pd.to_datetime(df_feature_ml2['split_time'])
  df_feature_ml2['date_order'] = pd.to_datetime(df_feature_ml2['date_order'])
  df_feature_ml2['r'] = (df_feature_ml2.split_time - df_feature_ml2.date_order).dt.days

  #store id, and channel
  df_cus = df.groupby(['customer_id','date_order'])['store_id','channel'].first().reset_index() 
  df_cus = df_cus.groupby(['customer_id'])['store_id','channel'].first().reset_index()
  
  df_feature_ml2 = pd.merge(df_feature_ml2, df_cus, on = 'customer_id', how = 'left')
  
  print_debug("[create_features_of_one_purchase_customer_for_predict] df_feature_ml2.shape {}\ndf_feature_ml2 = \n{}".format(df_feature_ml2.shape,df_feature_ml2))
  
  return df_feature_ml2[['customer_id','m','r','store_id','channel']]
  
##################################################################################
# Function: create_features_ml2(df, mode = 'PREDICT', cate_column, INI_SPLIT_TIME, END_DATE, slide_step):
# Description:
#    - data_pre_processing() --> data_pre_processing_ml2()
#    - Get customer having more than two purchases
# Usage:
#    Input : df:   df[customer_id, date_order, order_id, ..any..] 
#            mode = [TRAIN, TEST, PREDICT]
#                     TRAIN  : create features for train, build kmeans model
#                     TEST   : create features, using saved kmeans model
#                     PREDICT: create feature, using saved kmeans model, dont create label
#    Output: dataframe[customer_id,'date_order','m','daydiff','store_id','channel','gender']]
#

#catekmc_column is array of category when does kmeans cluster

def create_features_ml2(df, mode, SPLIT_TIME, cate_column, INI_SPLIT_TIME, END_DATE, slide_step):

  #R_KCLUSTER_ML2 = 13
  #M_KCLUSTER_ML2 = 13
  #CATE_KCLUSTER_ML2 = 13
  
  print_debug("#############################################################")
  print_debug("# CREATE FEATURE FOR MODEL 2: 01 purchase customer forecast")
  print_debug("# create_features_ml2(\n# df.colums = {},\n# mode = {},\n# SPLIT_TIME = {},\n# cate_column = {},\n# INI_SPLIT_TIME = {},\n# END_DATE = {},\n# slide_step = {})".format(df.columns.values, mode, SPLIT_TIME, cate_column, INI_SPLIT_TIME, END_DATE, slide_step))
  print_debug("#############################################################")
  
  catekmc_column = ''
  df_feature_ml2 = pd.DataFrame()
  
  #for TRAIN, TEST
  if not (mode == 'PREDICT'):
    df_cus = df.groupby(['customer_id','date_order']).amount.sum().reset_index()
    df_cus = df_cus.sort_values(['customer_id','date_order'])
    df_cus['next_order'] = df_cus.groupby('customer_id')['date_order'].shift(-1)
    df_cus['daydiff'] = (df_cus['next_order'] - df_cus['date_order']).dt.days

    # daydiff in not null --> number of purchase gte 2
    df_cus_freq_gte2 = df_cus.loc[~df_cus.daydiff.isnull(),:]
    df = df[df.customer_id.isin(df_cus_freq_gte2.customer_id)]
    print_debug("[create_features_ml2()][INFO] Number of customer having freq = 2 or more: {}".format(df_cus_freq_gte2.customer_id.nunique()))  
  
    # first day purchase, amount of first purchase day, daydiff
    df_cus_1 = df_cus_freq_gte2.groupby(['customer_id'])[['date_order','amount','daydiff']].first().reset_index()
    df_cus_1.columns = ['customer_id','date_order','m','daydiff']

    #store id, and channel
    df_cus_2 = df.groupby(['customer_id','date_order'])[['store_id','channel']].first().reset_index() 
    df_cus_2 = df_cus_2.groupby(['customer_id'])[['store_id','channel']].first().reset_index()
  
    ##########################################
    # Column Recency (slide split_time)
    ##########################################
  
    # df at this time only contains customer_id who have 2 purchase and above
    df_feature_r = slide_split_time_compute_recency (df, INI_SPLIT_TIME, END_DATE, slide_step) 
    
    df_feature_ml2 = pd.merge(df_cus_1, df_cus_2, on = 'customer_id', how = 'left')
    print_debug("[create_features_ml2()][INFO] Number of customers having two purchases: {}".format(df_feature_ml2.shape[0]))

    df_feature_ml2 = pd.merge(df_feature_ml2, df_feature_r, on = 'customer_id', how = 'right')
    print_debug("[create_features_ml2()][INFO] Number of customers for Train: {}".format(df_feature_ml2.shape[0]))

  else:
    
    df_feature_ml2 = create_features_of_one_purchase_customer_for_predict (df, SPLIT_TIME)
  
  ##############################
  # KMeans for Recency (R), Monetary (M)
  ##############################

  if mode=='TRAIN':
    df_r_clus = kmeans_cluster(df_feature_ml2[['customer_id','r']], R_KCLUSTER_ML2, False, savemodel_path+'/kmeans_r_model_ml2.sav')
    df_m_clus = kmeans_cluster(df_feature_ml2[['customer_id','m']], M_KCLUSTER_ML2, True, savemodel_path+'/kmeans_m_model_ml2.sav')
    df_feature_ml2 = pd.merge(df_feature_ml2, df_r_clus[['customer_id','r_clus']],on='customer_id',how='left')
    df_feature_ml2 = pd.merge(df_feature_ml2, df_m_clus[['customer_id','m_clus']],on='customer_id',how='left')
  else: #TEST and PREDICT
    print_debug("[create_features_ml2()][INFO] df_feature_ml2.r =  {}".format(np.array(df_feature_ml2.r.values).reshape(-1,1)))
    kmeans = pickle.load(open(savemodel_path+'/kmeans_r_model_ml2.sav','rb'))
    df_feature_ml2['r_clus'] = kmeans.predict(np.array(df_feature_ml2.r.values).reshape(-1,1))
    kmeans = pickle.load(open(savemodel_path+'/kmeans_m_model_ml2.sav','rb'))
    df_feature_ml2['m_clus'] = kmeans.predict(np.array(df_feature_ml2.m.values).reshape(-1,1))
    
  df = df[df.customer_id.isin(df_feature_ml2.customer_id)]
  ##############################
  # KMeans for Categories
  ##############################
  df_pivot = pd.pivot_table(df, values='qty', index=['customer_id'], columns=['category'], aggfunc=np.sum, fill_value=0)
  df_pivot.reset_index(inplace=True)
    
  columns = df_pivot.columns.values
  columns = np.array(columns).astype('str')
  columns = columns[columns!='customer_id']
  if (mode=='TRAIN'):    
    catekmc_column = columns #np.array(columns).astype('str')
    print_debug("TRAIN: Number categories of dftrain is {}, columns = {}".format(len(columns),columns))
  else:
    catekmc_column = cate_column
    # having at catekmc, but not in dftest --> add missing cols to df_pivot
    miss_cols = np.setdiff1d(catekmc_column, columns)    
    if len(miss_cols)>0:
      print_debug("miss_cols = {},  catekmc_column = {}, columns = {}".format(miss_cols, catekmc_column, columns))
      print_debug("[#1] df_pivot = {}".format(df_pivot.columns.values))
      for col in miss_cols:
        df_pivot[col] = 0
        print_debug("[#1.1] df_pivot = {}".format(df_pivot))

    columns = df_pivot.columns.values
    columns = np.array(columns).astype('str')
    columns = columns[columns!='customer_id']
    
    #miss_cols = np.setdiff1d(catekmc_column, columns)
    #if len(miss_cols) == 0:
    #  print_debug("[INFO] Number categories of dftest is {}, fit for running kmeans model".format(len(columns)))
    #else:
    #  print_debug("[ERROR] Number categories of dftest is {}, whereas of catekmc_column is {}, diff_col = {}".format(len(columns),len(catekmc_column),miss_cols))    
    print_debug("[#1.2] df_pivot columns = {}".format(df_pivot.columns.values))
    # having at dftest, but not in catekmc  --> remove 
    abundant_cols = np.setdiff1d(columns, catekmc_column)
    print_debug("abundant_cols = {},  catekmc_column = {}, columns = {}".format(abundant_cols, catekmc_column, columns))
    
    
    if len(abundant_cols)>0:
      df_pivot.drop(np.asarray(abundant_cols), axis = 1, inplace=True)

    print_debug("[#3] df_pivot = {}".format(df_pivot.columns.values))
    columns = df_pivot.columns.values
    columns = np.array(columns).astype('str')
    columns = columns[columns!='customer_id']
    
    if len(columns) == len(catekmc_column):
      print_debug("[INFO] Number categories of dftest and catekmc_column is the same: {}, fit for running kmeans model".format(len(columns)))
    else:
      print_debug("[ERROR] Number categories of dftest is {}, whereas of catekmc_column is {}".format(len(columns),len(catekmc_column)))    
      sys.exit(2)

  df_cus_kmc = df_pivot.groupby(['customer_id'])[columns].sum().reset_index()
  df_cus_kmc_drop =  df_cus_kmc.drop(['customer_id'],axis=1)

  # max-min normalization (Library)
  #min_max_scaler = preprocessing.MinMaxScaler()
  #x_scaled = min_max_scaler.fit_transform(df_cus_kmc_drop.values)
  #df_norm = pd.DataFrame(data = x_scaled, columns = df_cus_kmc_drop.columns.values)
  
  # max-min normalization (manual) 
  df_norm = (df_cus_kmc_drop - df_cus_kmc_drop.min(axis=0)) / (df_cus_kmc_drop.max(axis=0) - df_cus_kmc_drop.min(axis=0))
  # NA by mau = 0 
  df_norm.fillna(0,inplace=True)
  
  if mode=='TRAIN':
    kmeans = KMeans(n_clusters=CATE_KCLUSTER_ML2, max_iter=1000).fit(df_norm)
    df_norm['cate_clus'] = kmeans.labels_
    pickle.dump(kmeans,open(savemodel_path+'/kmeans_cate_model_ml2.sav','wb'))
    print_debug("[create_features_ml2()][SAVE MODEL] saved model of Categories KMC at {}".format(savemodel_path+'/kmeans_cate_model_ml2.sav'))
    
    # create df with category cluster result
    df_cus_category_cluster = pd.merge(df_cus_kmc[['customer_id']], df_norm[['cate_clus']],left_index=True, right_index=True)
  
    # Column: cate_clus
    df_feature_ml2 = pd.merge(df_feature_ml2,df_cus_category_cluster,on='customer_id', how='left')
  else:
    kmeans = pickle.load(open(savemodel_path+'/kmeans_cate_model_ml2.sav','rb'))
    df_feature_ml2['cate_clus'] = kmeans.predict(np.array(df_norm.values).reshape(-1,len(columns)))
    
  ##############################
  # Numerize for Channel
  ##############################

  # process channel: string --> numeric category
  channel_dict = {'POS':1,'Robins':2,'Lotte':3,'Lazada':4}

  for key, value in channel_dict.items():
    #print ("Channel: Key = {}, value = {}".format(key,value))
    df_feature_ml2.loc[df_feature_ml2.channel==key,'channel'] = channel_dict[key]

  if not (mode=='PREDICT'):
  
    # define y_label
    # Column: next_purchase_label or y_label
    df_feature_ml2['y_label'] = df_feature_ml2.nextpurchase_day.values
    for ind, val in enumerate(np.arange(0,PREDICT_LEN,CLUSTER_PERIOD)):
      df_feature_ml2.loc[df_feature_ml2.nextpurchase_day>val,'y_label'] = ind+1

  print_debug("[create_features_ml2()] df_feature_ml2.shape = {}\n df_feature_ml2 = {}".format(df_feature_ml2.shape, df_feature_ml2.head(10)))

  ##############################
  # Column 'gender' as feature
  ##############################
  #df_gender = df.groupby(['customer_id']).gender.first().reset_index()
  #df_feature_ml2 = pd.merge(df_feature_ml2,df_gender[['customer_id','gender']],on='customer_id',how='left')

  ##############################
  # Column 'vip' as feature
  ##############################
  #df_vip = df.groupby(['customer_id']).vip.first().reset_index()
  #df_feature_ml2 = pd.merge(df_feature_ml2,df_vip[['customer_id','vip']],on='customer_id',how='left')
  
  print_debug("catekmc_column = {}".format(catekmc_column))
  if mode=='TRAIN':
    return df_feature_ml2, catekmc_column
  else:
    return df_feature_ml2


##################################################################################
# Function: train_test_ml2_skfold (df, mode, model_savepath)
# Description:
#    - ... --> create_features_ml1() --> train_test_ml2_skfold
#    - Split data into [data before split date, data after split date]
#    - Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
# Usage:
#    Input: - df:   df_features [customer_id, r, m, r_clus, m_clus, nextpurchase_day, y_label]
#           - mode: ['TRAIN','TEST','TRAINZ']
#           - model_savepath: path to save model
#    Output: dataframe of cluster result [customer_id, r_clus|f_clus|m_clus]
#

def train_test_ml2_skfold (df, mode, model_savepath):
  
  print_debug("####################################################") 
  print_debug("# [Model 2] TRAINING_TEST_PREDICT : mode = {}".format(mode))
  print_debug("####################################################")
  
  #################################
  # Plot corr()
  df_plot = df[['r_clus', 'cate_clus', 'm_clus','nextpurchase_day','y_label']]  
  corr = df_plot[df_plot.columns].corr()
  plt.figure(figsize = (12,8))
  sns_plot = sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f",cmap='coolwarm_r')
  sns_fig = sns_plot.get_figure()
  sns_fig.savefig(log_path+"/ml2_features_label_correlation_map.png")
  #################################
  
  model_temp = g_model
  modeln = models
  if modeln=='xgbr':
    modeln='xgb'
  #  if not (mode=='PREDICT'):
  
  X = df[['r_clus', 'cate_clus', 'm_clus']]
  y = df[['y_label']]
  
  print_report("[train_test_ml2_skfold()] Data input for TRAIN OR TEST: \n X = {}\n y = {}".format(X.head(), y.head()))

  ##########################
  #transform X,y
  X = X.to_numpy()
  y = y.to_numpy()
  y = y.reshape((len(y),))  
  print_report('[train_test_ml2_skfold()] [LightGBM] Transformed input\n X.shape {}, y.shape {}\n'.format(X.shape, y.shape))
 
  scores = []
  y_label_pred=[]
   
  ################################
  # TRAIN
  ################################  
  
  if (mode=='TRAIN'):
    skf = StratifiedKFold(n_splits=NFOLD, random_state=10, shuffle=True)
    
    i = 0
    df['nextpurchase_fromsplit_pred'] = 0
    
    ################################################
    # loop Stratified-Kfold Data Split and Train
    ################################################
    
    for train_index, test_index in skf.split(X, y):
      i+=1
      model_savepath_skf = model_savepath+"_skf_"+str(i)+".sav"
      print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      
      if modeln == 'lgb':
        model, acc_score, y_label_pred = lightgbm_train_transform_rs(X_train, y_train, X_test, y_test, X, skfold)
        
      else:
        model = model_temp.fit(X_train, y_train)
        acc_score = model.score(X_test,y_test)
        scores.append(acc_score)
        y_label_pred = model.predict(X)          
        
      # SAVE MODEL and TEST
      pickle.dump(model,open(model_savepath_skf,'wb'))
      print_debug("[train_test_ml2_skfold()][SKFold] [{}] saved model at {}".format(i,model_savepath_skf))

      # Append score result of each SKfold      
      scores.append(acc_score)
      
      ##################################
      # create 'y_label_pred' for accuracy report when predicting, using last skfold for y_label_predict
      df['y_label_pred'] = y_label_pred
      df['nextpurchase_fromsplit_pred'] = df.nextpurchase_fromsplit_pred + (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)

    ##<-> 
    df['nextpurchase_fromsplit_pred'] = round(df.nextpurchase_fromsplit_pred/NFOLD,0).astype(int) 
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_fromsplit_pred - (df.nextpurchase_day - df.r)
    accuracy_score = np.mean(scores)
    
    print_debug('[train_test_ml2_skfold()] [INFO] Accuracy of stratifiedKfold: {}, \n --> Average Accuracy: {} '.format(scores, accuracy_score))
    print_report('[train_test_ml2_skfold()] [INFO] Accuracy of stratifiedKfold: {}, \n --> Average Accuracy: {} '.format(scores, accuracy_score))
                
  ################################
  # TEST
  ################################     

  if mode=='TEST':
    df['nextpurchase_fromsplit_pred'] = 0
    for i in np.arange(1,NFOLD+1):
      model_savepath_skf = model_savepath+"_skf_"+str(i)+".sav"
      model=pickle.load(open(model_savepath_skf,'rb'))
      
      # LightGBM
      if modeln == 'lgb':
        ##############################################################
        # This for test model with test data to get accuracy score        
        # Prediction
        y_pred = model.predict(X, num_iteration=model.best_iteration)

        y_pred_softmax = np.zeros_like(y_pred)
        y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
        y_label_pred = np.argmax(y_pred_softmax, axis=1)  
    
        ###############################################
        # replace by this for accuracy computation
        valid_count = 0
        #print ("len(y_test)={}, len(y_pred)={}".format(len(y_test),len(y_label_pred)))
        for j,k in zip(y, y_label_pred):
          if j==k:
            valid_count+=1        
        acc_score = valid_count/len(y)
        ###############################################
        #scores.append(acc_score)        
        print_debug ("TEST skfold {}--> accuracy_score = {}".format(i, acc_score))
                    
      # XGBoost, DCT, RF, SVC            
      else:
        y_label_pred = model.predict(X)
        acc_score = model.score(X,y)

      scores.append(acc_score)  
      df['y_label_pred'] = y_label_pred
      df['nextpurchase_fromsplit_pred'] = df.nextpurchase_fromsplit_pred + (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)
      
    ##<->  
    df['nextpurchase_fromsplit_pred'] = round(df.nextpurchase_fromsplit_pred/NFOLD,0).astype(int) 
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_fromsplit_pred - (df.nextpurchase_day - df.r)
    accuracy_score = np.mean(scores)
    
    print_debug("#################################################################")    
    print_debug("[train_test_ml2_skfold()] MODEL 2 RESULT FOR dftest:")
    print_debug("[train_test_ml2_skfold()] Classication Report \n{}".format(classification_report(y, y_label_pred)))
    print_debug("[train_test_ml2_skfold()] Confusion Matrix \n{}".format(confusion_matrix(y, y_label_pred)))
    print_debug("[train_test_ml2_skfold()] Accuracy Score: {}".format(accuracy_score))
    print_debug("#################################################################")    
    
 
  #######################
  # set Error Wing is 45
  OUT_STD = PRERROR
  check_arr = []
  for i in range(len(df)):
    cus_std = OUT_STD
    if ((df.loc[i,'nextpurchase_diff_act_pred']<=cus_std)&(df.loc[i,'nextpurchase_diff_act_pred']>=-cus_std)):
      check_arr.append(True)
    else:
      check_arr.append(False)
  df['result_check'] = check_arr

  result_check_hist = np.unique(df.result_check,return_counts=True)
  accuracy = result_check_hist[1][1]/len(df)
  print_debug("[train_test_ml2_skfold()][RESULT] ADJUST RESULT FOR FINAL")
  print_debug("[train_test_ml2_skfold()][RESULT] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_debug("[train_test_ml2_skfold()][RESULT] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  print_report("[train_test_ml2_skfold()] [Adjust] ADJUST RESULT FOR FINAL")
  print_report("[train_test_ml2_skfold()] [Adjust] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_report("[train_test_ml2_skfold()] [Adjust] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  if mode=='TRAIN':
    # save to file the accuracy of each cluster
    df_clus_cus_count = df.groupby(['y_label_pred']).customer_id.count().reset_index()
    df_clus_cus_count.columns = ['y_label_pred','customer_qty']
    print_debug("[train_test_ml2_skfold()][compute cluster acc] df_clus_cus_count =\n{}".format(df_clus_cus_count))
  
    df_clus_cus_right = df.groupby(['y_label_pred']).result_check.sum().reset_index()
    df_clus_cus_right.columns = ['y_label_pred','right_pred']
    print_debug("[train_test_ml2_skfold()][compute cluster acc] df_clus_cus_right =\n{}".format(df_clus_cus_right))

    df_clus_acc = pd.merge(df_clus_cus_count,df_clus_cus_right,on='y_label_pred',how='left')
    print_debug("[train_test_ml2_skfold()][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc['accuracy'] = df_clus_acc.right_pred/df_clus_acc.customer_qty
    print_debug("[train_test_ml2_skfold()][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc.to_csv(ml_model_path+"/ml2_y_predict_cluster_accuracy_"+modeln+".csv",index=False)
  
  #if mode=='TRAIN':
  #  model = model_temp.fit(X, y)
  #  # SAVE MODEL
  #  pickle.dump(model,open(model_savepath,'wb'))
  #  print_debug("[train_test_ml2()][TRAIN mode] Full data model train is saved model at {}".format(model_savepath))
    
    

##################################################################################
# Function: train_test_ml2 (df, mode, model_savepath)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_ml2
#    - Split data into [data before split date, data after split date]
#    - Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
# Usage:
#    Input: - df:   df_features [customer_id, r, m, r_clus, m_clus, nextpurchase_day, y_label]
#           - mode: ['TRAIN','TEST','TRAINZ']
#           - model_savepath: path to save model
#    Output: dataframe of cluster result [customer_id, r_clus|f_clus|m_clus]
#

def train_test_ml2 (df, mode, model_savepath):
  
  print_debug("####################################################") 
  print_debug("# [Model 2] TRAINING_TEST_PREDICT : mode = {}".format(mode))
  print_debug("####################################################")
  model_savepath = model_savepath+".sav"
  #################################
  # Plot corr()
  df_plot = df[['r_clus', 'cate_clus', 'm_clus','nextpurchase_day','y_label']]  
  corr = df_plot[df_plot.columns].corr()
  plt.figure(figsize = (12,8))
  sns_plot = sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f",cmap='coolwarm_r')
  sns_fig = sns_plot.get_figure()
  sns_fig.savefig(log_path+"/ml2_features_label_correlation_map.png")
  #################################
  
  #X=''#pd.DataFrame()
  #y=''#pd.DataFrame()
  model_temp = g_model
  modeln = models
  acc_score = ''
  y_label_pred=''
  
  if modeln=='xgbr':
    modeln='xgb'
  #  if not (mode=='PREDICT'):
  
  X = df[['r_clus', 'cate_clus', 'm_clus']]
  y = df[['y_label']]
  
  print_report("[train_test_ml2] Data input for TRAIN OR TEST: \n X = {}\n y = {}".format(X.head(), y.head()))

  if (mode=='TRAIN'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, stratify=y)
    
    # LightGBM
    if modeln == 'lgb':
      model, acc_score, y_label_pred = lightgbm_train_transform_rs(X_train, y_train, X_test, y_test, X, skfold)
    # XGboost, DCT, RF	
    else:
      model = model_temp.fit(X_train, y_train)
      acc_score = model.score(X_test,y_test)
      y_label_pred = model.predict(X_test)          
        
    # SAVE MODEL 
    pickle.dump(model,open(model_savepath,'wb'))
    print_debug("[train_test_ml2] [LightGBM] saved model at {}".format(model_savepath))	
	  
    print_debug("[TRAIN MODE] Model 02 Result ")
    print_debug("Accuracy Score: {}".format(acc_score))
    print_debug("Classication Report \n{}".format(classification_report(y_test, y_label_pred)))
    print_debug("Confusion Matrix \n{}".format(confusion_matrix(y_test, y_label_pred)))
    
    print_report("[TRAIN MODE] Model 02 Result ")
    print_report("[y_clus] Accuracy Score: {}".format(acc_score))
    print_report("[y_clus] Classication Report \n{}".format(classification_report(y_test, y_label_pred)))
    print_report("[y_clus] Confusion Matrix \n{}".format(confusion_matrix(y_test, y_label_pred)))
      
	##############################################
	# SAVE ACCURACY SCORE of EACH LABEL GROUP
     
    # save f1_score of all cluster of current model
    #with open(model_path+'/model01_cluster_accuracy_'+modeln+'.txt', 'w') as fw:
    #  fw.write('#'.join(str(e) for e in np.unique(y_label_pred,return_counts=True)[0])+'\n')
    #  fw.write('#'.join(str(e) for e in f1_score(y_test, y_label_pred, average=None)))    
        
			
  ############################################
  # TEST MODE  
  ############################################  
  if (mode=='TEST'):
    #load model
    model=pickle.load(open(model_savepath,'rb'))
	
    if modeln=='lgb':
      y_pred = model.predict(X, num_iteration=model.best_iteration)
      y_pred_softmax = np.zeros_like(y_pred)
      y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
      y_label_pred = np.argmax(y_pred_softmax, axis=1)  
      #print ("y={}, y_pred={}, y.value={},y.flatten={}".format(y,y_label_pred,y.values,y.values.flatten()))
      acc_score = accuracy_score(y, y_label_pred)
    else:
      y_label_pred = model.predict(X)
      acc_score = model.score(X,y)
    
    print_debug("[TEST MODE] MODEL 2: \n")
    print_debug("Accuracy Score: {}".format(acc_score))
    print_debug("Classication Report \n{}".format(classification_report(y, y_label_pred)))
    print_debug("Confusion Matrix \n{}".format(confusion_matrix(y, y_label_pred)))
    
  #######################################################################
  # Adjust output data to compute the next purchase day after split date
  #######################################################################
  
  if modeln=='lgb':
      y_pred = model.predict(X, num_iteration=model.best_iteration)
      y_pred_softmax = np.zeros_like(y_pred)
      y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
      y_label_pred = np.argmax(y_pred_softmax, axis=1)  
      #print ("y={}, y_pred={}, y.value={},y.flatten={}".format(y,y_label_pred,y.values,y.values.flatten()))
      acc_score = accuracy_score(y, y_label_pred)
  else:
      y_label_pred = model.predict(X)
      acc_score = model.score(X,y)

  df['y_label_pred'] = y_label_pred
  print_debug("[FULL DATA] MODEL 2 PREDICT RESULT FOR FULL DATA: \n")
  print_debug("Accuracy Score: {}".format(acc_score))
  print_debug("Classication Report \n{}".format(classification_report(y, y_label_pred)))
  print_debug("Confusion Matrix \n{}".format(confusion_matrix(y, y_label_pred)))
  
  #create num of day after split date the customer could purchase
  if not (modeln =='xgbr'):
    df['nextpurchase_fromsplit_pred'] = (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_fromsplit_pred - (df.nextpurchase_day - df.r)
  else:
    df['nextpurchase_pred'] = df['y_label_pred']
    df['nextpurchase_diff_act_pred'] = df.nextpurchase_pred - df.nextpurchase_day
 
 
  #######################
  # set Error Wing is 45
  OUT_STD = PRERROR
  check_arr = []
  for i in range(len(df)):
    cus_std = OUT_STD
    if ((df.loc[i,'nextpurchase_diff_act_pred']<=cus_std)&(df.loc[i,'nextpurchase_diff_act_pred']>=-cus_std)):
      check_arr.append(True)
    else:
      check_arr.append(False)
  df['result_check'] = check_arr

  result_check_hist = np.unique(df.result_check,return_counts=True)
  accuracy = result_check_hist[1][1]/len(df)
  print_debug("[FULL DATA][MODEL 2] ADJUST RESULT FOR FINAL")
  print_debug("[ADJUST] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_debug("[ADJUST] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  print_report("[FULL DATA][MODEL 2] ADJUST RESULT FOR FINAL")
  print_report("[ADJUST] Predicting for Total number of customer: {}".format(df.shape[0]))
  print_report("[ADJUST] Final Result of model after adjust: accuracy = {}".format(accuracy))
  
  if mode=='TRAIN':
    # save to file the accuracy of each cluster
    df_clus_cus_count = df.groupby(['y_label_pred']).customer_id.count().reset_index()
    df_clus_cus_count.columns = ['y_label_pred','customer_qty']
    print_debug("[train_test_ml2][compute cluster acc] df_clus_cus_count =\n{}".format(df_clus_cus_count))
  
    df_clus_cus_right = df.groupby(['y_label_pred']).result_check.sum().reset_index()
    df_clus_cus_right.columns = ['y_label_pred','right_pred']
    print_debug("[train_test_ml2][compute cluster acc] df_clus_cus_right =\n{}".format(df_clus_cus_right))

    df_clus_acc = pd.merge(df_clus_cus_count,df_clus_cus_right,on='y_label_pred',how='left')
    print_debug("[train_test_ml2][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc['accuracy'] = df_clus_acc.right_pred/df_clus_acc.customer_qty
    print_debug("[train_test_ml2][compute cluster acc] df_clus_acc =\n{}".format(df_clus_acc))
  
    df_clus_acc.to_csv(ml_model_path+"/ml2_y_predict_cluster_accuracy_"+modeln+".csv",index=False)
  
  if mode=='TRAIN':
    if modeln=='lgb':
      print_debug("[RETRAIN FULL DATA] LightGBM --> skip this step")
    else:
      model = model_temp.fit(X, y)
      pickle.dump(model,open(model_savepath,'wb'))
    
##################################################################################
# Function: predict_ml2 (df, model_savepath):
# Description:
#    - data_pre_processing() --> create_features_ml2() --> predict_ml2()
#    - Split data into [data before split date, data after split date]
#    - Features+label dateframe for training [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
# Usage:
#    Input: - df:   df_features [customer_id, r, m, r_clus, m_clus, cate_clus]
#           - model_savepath: path to save model
#    Output: dataframe ['customer_id','y_label_pred','nextpurchase_day_pred']
#

def predict_ml2 (df, model_savepath):

  modeln = models 
  print_debug("####################################################") 
  print_debug("# [Model 2] MODEL 02 PREDICT ")
  print_debug("####################################################")
  print_debug("PREDICT RUN: Model: {}, skfold: {}...".format(modeln, skfold))
    
  X = df[['r_clus', 'cate_clus', 'm_clus']]
  
  #######################################################################
  # Predict and Adjust output data
  if skfold:

    #transform X
    X = X.to_numpy()
    df['nextpurchase_fromsplit_pred'] = 0
    for i in np.arange(1,NFOLD+1):
	
      model_savepath_skf = model_savepath+"_skf_"+str(i)+".sav"      
      model=pickle.load(open(model_savepath_skf,'rb'))
      print_debug("Load model: {}".format(model_savepath_skf))
      #y_label_pred = model.predict(X)
      	  
	  # LightGBM
      if modeln == 'lgb':
        ##############################################################
        # This for test model with test data to get accuracy score        
        # Prediction
        y_pred = model.predict(X, num_iteration=model.best_iteration)

        y_pred_softmax = np.zeros_like(y_pred)
        y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
        y_label_pred = np.argmax(y_pred_softmax, axis=1)          
                    
      # XGBoost, DCT, RF, SVC            
      else:
        y_label_pred = model.predict(X)
	  
      df['y_label_pred'] = y_label_pred
      df['nextpurchase_fromsplit_pred'] = df.nextpurchase_fromsplit_pred + (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)
      print_debug("Complete predict and output process for skfold {}".format(i))         
	  
    df['nextpurchase_day_pred'] = round(df.nextpurchase_fromsplit_pred/NFOLD,0).astype(int) 

  else:
    model_savepath=model_savepath+".sav"
    model=pickle.load(open(model_savepath,'rb'))
	
    # LightGBM
    if modeln == 'lgb':
      ##############################################################
      # This for test model with test data to get accuracy score        
      # Prediction
      y_pred = model.predict(X, num_iteration=model.best_iteration)
      y_pred_softmax = np.zeros_like(y_pred)
      y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
      y_label_pred = np.argmax(y_pred_softmax, axis=1)          
                    
    # XGBoost, DCT, RF, SVC            
    else:
      y_label_pred = model.predict(X)

    df['y_label_pred'] = y_label_pred
  
  ####################################
  #create num of day after split date the customer could purchase
  df['nextpurchase_day_pred'] = (df.y_label_pred - 1)*CLUSTER_PERIOD - df.r + int(CLUSTER_PERIOD/2)

  return df[['customer_id','y_label_pred','nextpurchase_day_pred']]
 
 
##################################################################################
# Function: main_train_ml1  (df,SPLIT_DATE,savingmodel_path_ml1)
# Description:
#    main_train_ml1 is full flow of processing data, create feature, train and test model.
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1()
#
# Usage:
#    Input: - df:  customer purchase order_line 
#           - SPLIT_DATE: date for split
#           - savingmodel_path_ml1: path to save model
#
#    Output: model 01, save at savingmodel_path_ml1
#
 
  
def main_train_ml1_nostrip (df,SPLIT_DATE,savingmodel_path_ml1):
  
  ##################################################################################################
  #                            A. MODEL 01 : TRAIN + TEST                                         ##
  #                   Predict for customer with at least two purchased                            ##
  ##################################################################################################

  print_report("\n#########################################")
  print_report("###### MODEL 01: TRAIN \n")
  print_report("#########################################")

  # model 01 saved path
  #savingmodel_path_ml1 = savemodel_path+"/ml1_load_model_next_purchasing_forecast.sav"

  # filtering raw data
  df_pre_processing = data_pre_processing(df)

  # get customer have 2 before split, 01 after split to be used for Training model
  df_train_ml1, df_cus_bf_gte2 = data_pre_processing_ml1(df_pre_processing, SPLIT_DATE, False)

  # Featuring
  df_features = create_features_ml1(df_train_ml1, df_cus_bf_gte2, SPLIT_DATE, 'TRAIN')

  # Train model (XGBoosting)
  train_test_ml1(df_features,'TRAIN',savingmodel_path_ml1)
  
  return True

def main_train_ml1 (df,SPLIT_DATE,SLIDE_BACKWARD_STEP,savingmodel_path_ml1):
  
  # SLIDE_BACKWARD_STEP to get more samples
  
  ##################################################################################################
  #                            A. MODEL 01 : TRAIN + TEST                                         ##
  #                   Predict for customer with at least two purchased                            ##
  ##################################################################################################

  print_report("\n#########################################")
  print_report("###### MODEL 01: TRAIN \n")
  print_report("#########################################")

  # model 01 saved path
  #savingmodel_path_ml1 = savemodel_path+"/ml1_load_model_next_purchasing_forecast.sav"

  # filtering raw data
  df_pre_processing = data_pre_processing(df)
  #df_train_ml1 = pd.DataFrame(columns = df_pre_processing.columns.values)  
  
  # dummy run
  df_train_ml1, df_cus_bf_gte2 = data_pre_processing_ml1(df_pre_processing, SPLIT_DATE, False)
  #df_cus_bf_gte2 = pd.DataFrame(columns = df_cus_bf_gte2xx.columns.values)
  
  # Featuring
  df_features = create_features_ml1(df_train_ml1, df_cus_bf_gte2, SPLIT_DATE, 'TRAIN')
  print_debug("[main_train_ml1()] SPLIT_DATE = {} -- Number of customers (samples) is {}".format(SPLIT_DATE,df_features.customer_id.nunique())) 
  
  STOP_SLIDE_DATE   = pd.to_datetime(df_pre_processing.date_order.min()) + datetime.timedelta(days=60)  
  MOVING_SPLIT_DATE = SPLIT_DATE - datetime.timedelta(days=SLIDE_BACKWARD_STEP) 

  print_debug("[main_train_ml1()] Stripping SPLIT TIME to get more samples")  
  while MOVING_SPLIT_DATE > STOP_SLIDE_DATE:
    
    df_pre_processing = df_pre_processing[~df_pre_processing.customer_id.isin(df_train_ml1.customer_id)]  
    print_debug("[main_train_ml1()] SPLIT_DATE = {} -- df_pre_processing.customer_id.nunique() = {}".format(MOVING_SPLIT_DATE,df_pre_processing.customer_id.nunique()))
            
    if df_pre_processing.customer_id.nunique() == 0:
      break   
    # get customer have 2 before split, 01 after split to be used for Training model
    df_train_ml1, df_cus_bf_gte2 = data_pre_processing_ml1(df_pre_processing, MOVING_SPLIT_DATE, False)
    if not (len(df_train_ml1) == 0): 
      df_features_i = create_features_ml1(df_train_ml1, df_cus_bf_gte2, MOVING_SPLIT_DATE, 'TEST')    
      df_features = pd.concat([df_features,df_features_i])
      print_debug("[main_train_ml1()] Strip to SPLIT_DATE = {} -- Adding {} samples, total have {}".format(MOVING_SPLIT_DATE,df_features_i.customer_id.nunique(),df_features.customer_id.nunique())) 
         
    MOVING_SPLIT_DATE = MOVING_SPLIT_DATE - datetime.timedelta(days=SLIDE_BACKWARD_STEP) 
     
  #print_debug("feature = \n{}".format(df_features))
  df_features.reset_index(inplace = True, drop = True)

  # Train model (XGBoosting)
  if skfold:
    train_test_ml1_skfold (df_features,'TRAIN',savingmodel_path_ml1)
  else:
    train_test_ml1(df_features,'TRAIN',savingmodel_path_ml1)
  
  return True
  
  
##################################################################################
# Function: main_test_ml1 (df,TEST_SPLIT_DATE,loadmodel_path_ml1)
# Description:
#    main_train_ml1 is full flow of processing data, create feature, train and test model.
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1()
#
# Usage:
#    Input: - df:  customer purchase order_line 
#           - TEST_SPLIT_DATE: date for split
#           - loadmodel_path_ml1: path to save model
#
#    Output: result of model 01 test with input data
#
 
def main_test_ml1 (df,TEST_SPLIT_DATE,loadmodel_path_ml1):

  print_report("\n#########################################")
  print_report("###### MODEL 01: TEST \n")
  print_report("#########################################")

  # filtering raw data
  df_pre_processing = data_pre_processing(df)

  # get customer have 2 before split, 01 after split to be used for Training model
  df_train_ml1, df_cus_bf_gte2 = data_pre_processing_ml1(df_pre_processing, TEST_SPLIT_DATE,False)

  # Featuring
  df_features = create_features_ml1(df_train_ml1, df_cus_bf_gte2, TEST_SPLIT_DATE, 'TEST')

  # Test model (XGBoosting)
  if skfold:
    train_test_ml1_skfold(df_features,'TEST',loadmodel_path_ml1)
  else:
    train_test_ml1(df_features,'TEST',loadmodel_path_ml1)

  return True
    
##################################################################################
# Function: main_train_ml2 (df, INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2)
# Description:
#    main_train_ml2 is full flow of processing data, create feature, train and test model.
#    - data_pre_processing() --> create_features_ml2() --> train_test_ml2()
#
# Usage:
#    Input: - df:  customer purchase order_line 
#           - INI_SPLIT_TIME: initial date for split date slide
#           - END_DATE: the day split date slide to
#           - SLIDE_STEP: number of day for split time slide
#           - savingmodel_path_ml2: path to save model
#
#    Output: model 02, save at savingmodel_path_ml2
#

def main_train_ml2 (df, INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2):

  modeln = models
  print_report("\n#########################################")
  print_report("###### MODEL 02: TRAIN \n")
  print_report("#########################################")
  
  # filtering raw data
  df_pre_processing = data_pre_processing(df)

  # Featuring
  #INI_SPLIT_TIME = datetime.datetimetime(2018,1,20)
  df_features_ml2, catekmc_column = create_features_ml2(df_pre_processing,'TRAIN',0,0,INI_SPLIT_TIME, END_DATE, SLIDE_STEP)

  with open(model_path+'/catekmc_column_'+modeln+'.txt', 'w') as fw:
    if not (len(catekmc_column)<2):
      fw.write('#'.join(catekmc_column))
    else:
      fw.write(catekmc_column)

  # Train model (XGBoosting)
  if skfold:
    train_test_ml2_skfold (df_features_ml2,'TRAIN',savingmodel_path_ml2)
  else:
    train_test_ml2(df_features_ml2,'TRAIN',savingmodel_path_ml2)
	
  return True

##################################################################################
# Function: main_train_ml2 (df, INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2)
# Description:
#    main_train_ml2 is full flow of processing data, create feature, train and test model.
#    - data_pre_processing() --> create_features_ml2() --> train_test_ml2()
#
# Usage:
#    Input: - df:  customer purchase order_line 
#           - INI_SPLIT_TIME: initial date for split date slide
#           - END_DATE: the day split date slide to
#           - SLIDE_STEP: number of day for split time slide
#           - loadmodel_path_ml2: path to load model
#
#    Output: model 02, save at loadmodel_path_ml2
# 
# 

def main_test_ml2 (df,INI_SPLIT_TIME, END_DATE, SLIDE_STEP, loadmodel_path_ml2):

  modeln = models
  print_report("\n#########################################")
  print_report("###### MODEL 01: TEST \n")
  print_report("#########################################") 
  
  # filtering raw data
  df_pre_processing = data_pre_processing(df)
  
  #get catekmc_column
  with open(model_path+'/catekmc_column_'+modeln+'.txt', 'r') as fr:
    catekmc_column = fr.readline()
    catekmc_column = catekmc_column.split("#")
    print_debug("[##4] .split' {}".format(catekmc_column))
  
  # Featuring
  df_features_ml2  = create_features_ml2(df_pre_processing,'TEST',0,catekmc_column,INI_SPLIT_TIME, END_DATE, SLIDE_STEP)
  # TEST model 02 (XGBoosting)
  #train_test_ml2(df_features_ml2,'TEST',loadmodel_path_ml2)
  
  
  # Test model (XGBoosting)
  if skfold:
    train_test_ml2_skfold(df_features_ml2,'TEST',loadmodel_path_ml2)
  else:
    train_test_ml2(df_features_ml2,'TEST',loadmodel_path_ml2)
	
  return True
  

##################################################################################
# Function: main_predict_out (df, CURRENT_DATE, savingmodel_path_ml1, savingmodel_path_ml2 )
# Description:
#    main_train_ml1 is full flow of processing data, create feature, train and test model.
#    - data_pre_processing() --> create_features_ml2() --> train_test_ml2()
#
# Usage:
#    Input: - df:  customer purchase order_lines
#           - CURRENT_DATE: the date from there you predict (eg.now)
#           - savingmodel_path_ml1: model 1 saving path
#           - savingmodel_path_ml2: model 2 saving path
#
#    Output: dataframe [customer_id, nextpurchase_day]
# 
# 

def main_predict_out (df, CURRENT_DATE, savingmodel_path_ml1, savingmodel_path_ml2):

  modeln = models
  #today = date.today()
  #CURRENT_DATE = today
  #CURRENT_DATE = datetime.datetime(2019,12,22)
  #############################################################
  # MODEL 01: Predict for customer having 2 purchases or more
  # filtering raw data
  print_debug ('[1] customer raw: {}'.format(df.customer_id.nunique()))
  total_cus = df.customer_id.nunique()
  
  df_pre_processing = data_pre_processing(df)
  print_debug ('[2] customer after preprocessing: {}'.format(df_pre_processing.customer_id.nunique()))
  num_cus_unnormal = total_cus - df_pre_processing.customer_id.nunique()
  
  print_debug("[Main] Total customer need to process: \n{}".format(df_pre_processing.customer_id.nunique()))

  # get cutomer having two purchases before current date, PREDICT_MODE = True
  df_predict_ml1, df_cus_bf_gte2 = data_pre_processing_ml1(df_pre_processing, CURRENT_DATE, True)
 

  # Featuring: PREDICT mode
  df_features = create_features_ml1(df_predict_ml1, df_cus_bf_gte2, CURRENT_DATE, 'PREDICT')

  # Predict
  df_pre_ml1 = predict_ml1(df_features, savingmodel_path_ml1)
  print_debug ('[3] customer having 2 purchases and more: {}'.format(df_features.customer_id.nunique())) 
  num_cus_2pur = df_features.customer_id.nunique()
  
  print_debug("[Main] Model 01: Number of customers: {}".format(df_pre_ml1.shape[0]))
  print_debug("[Main] Model 01: Predict result \n{}".format(df_pre_ml1))

  #############################################################
  # MODEL 02: Predict for customer having first purchasing
  #############################################################
  with open(model_path+'/catekmc_column_'+modeln+'.txt', 'r') as fr:
    catekmc_column = fr.readline()
    catekmc_column = catekmc_column.split("#")
    #print_debug("[##4] .split' {}".format(catekmc_column))

  # Featuring
  df_features_ml2  = create_features_ml2(df_pre_processing,'PREDICT',CURRENT_DATE,catekmc_column,0,0,0)

  # TEST model 02 (XGBoosting)
  df_pre_ml2 = predict_ml2(df_features_ml2,savingmodel_path_ml2)
  print_debug ('[4] customer having 1 purchase: {}'.format(df_pre_ml2.customer_id.nunique())) 
  num_cus_1pur = df_pre_ml2.customer_id.nunique()
  
  print_debug("[Main] Model 02: Number of customers: {}".format(df_pre_ml2.shape[0]))
  print_debug("[Main] Model 02: Predict result \n{}".format(df_pre_ml2))
  #print_debug("[Ma cusin] Total customers are predicted: {}".format(df_pre_ml1.shape[0]+df_pre_ml2.shape[0]))
  
  # Add accuracy column
  df_acc_ml1 = pd.read_csv(ml_model_path+'/ml1_y_predict_cluster_accuracy_'+modeln+'.csv', index_col=None)
  df_acc_ml2 = pd.read_csv(ml_model_path+'/ml2_y_predict_cluster_accuracy_'+modeln+'.csv', index_col=None)
  df_pre_ml1 = pd.merge(df_pre_ml1,df_acc_ml1,on='y_label_pred',how='left')
  df_pre_ml2 = pd.merge(df_pre_ml2,df_acc_ml2,on='y_label_pred',how='left')
  
  # Add column for marking 1 purchase customer and 2+ purchase customers by columns 'cus_type'
  df_pre_ml1.loc[:,'cus_type'] = int(2)
  df_pre_ml2.loc[:,'cus_type'] = int(1)
  
  # Concate result of two model
  df_predict = pd.concat([df_pre_ml1,df_pre_ml2])
  
  # round() accuracy of predict result
  #df_predict['clus_acc_round'] = df_predict.round({'accuracy': 3})
  
  # Column 'from_day' and 'to_day'
  df_predict['from_day'] = df_predict.nextpurchase_day_pred - PRERROR
  df_predict['to_day'] = df_predict.nextpurchase_day_pred + PRERROR
  
  # Column 'from_date' and 'to_date'
  df_predict['from_date'] = 0
  df_predict['to_date'] = 0
  
  df_predict['pred_standing_date'] = CURRENT_DATE
  df_predict.reset_index(drop=True, inplace=True)
  
  df_predict['from_date'] = pd.to_datetime(df_predict['pred_standing_date']) + pd.to_timedelta(df_predict['from_day'], unit='days')
  df_predict['to_date'] = pd.to_datetime(df_predict['pred_standing_date']) + pd.to_timedelta(df_predict['to_day'], unit='days')
  
  print_debug("[main_predict_out()] Number of customer: {}\n df_predict =\n{}".format(df_predict.shape[0],df_predict.head()))
  print_report("[main_predict_out()] Number of customer: {}\n df_predict =\n{}".format(df_predict.shape[0],df_predict.head()))
  
  #################################
  df_cus = pd.DataFrame(data=[total_cus, num_cus_1pur, num_cus_2pur, num_cus_unnormal], index = ['Total','One-purchase','Two-Purchases-or-more','Unnormal'], columns=['Customer_Group'])

  df_cus.reset_index(inplace=True)
  df_cus.columns = ['Customer_Group','Quantity']
  sns.set(style="whitegrid")
  ax1 = sns.barplot(x="Customer_Group", y="Quantity", data=df_cus,palette="Blues_d")
  for ind, row in df_cus.iterrows():
    ax1.text(row.name,row.Quantity, row.Quantity, color='black', ha="center")
  #save image
  fig = ax1.get_figure()
  fig.set_size_inches(8.0, 4.5)
  fig.savefig(log_path+'/customer_quantity_in_group.png')
  ################################
  
  return df_predict[['customer_id','pred_standing_date','nextpurchase_day_pred','from_day','to_day','from_date','to_date','accuracy','cus_type']]
  
##################################################################################
# Function: pseudo_labeling_train_ml1 (df, SPLIT_DATE, savingmodel_path_ml1)
# Description:
#    - data_pre_processing() --> data_pre_processing_ml1() --> create_features_ml1() --> train_test_ml1()
#    - run train model for input data, using model to predict customer get pseudo label
#    - Use new label for retrain model [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day(y), y_label ] 
#           Number of features input to model for training can be adjusted.
# Usage:
#    Input: - df:   df_features [customer_id, r, f, m, r_clus, f_clus, m_clus, daydiff_mean, daydiff_std, nextpurchase_day, y_label]
#           - SPLIT_DATE 
#           - savingmodel_path_ml1: path to save model
#    Output: pseudo model
#

#def train_with_pseudo_labeling (df, mode, model_savepath):  
def pseudo_labeling_train_ml1 (df, SPLIT_DATE, savingmodel_path_ml1):

  modeln=models
  #############################################################
  # MODEL 01: Predict for customer having 2 purchases or more
  # filtering raw data
  df_pre_processing = data_pre_processing(df)

  print_debug("[pseudo_labeling_train_ml1()] Total customer need to process: \n{}".format(df_pre_processing.customer_id.nunique()))

  # get cutomer having two purchases before current date, PREDICT_MODE = False <--> TRAIN_MODE
  df_train_ml1, df_cus_bf_gte2_train = data_pre_processing_ml1(df_pre_processing, SPLIT_DATE, False)
  
  # get cutomer having two purchases before current date, PREDICT_MODE = True
  df_predict_ml1, df_cus_bf_gte2 = data_pre_processing_ml1(df_pre_processing, SPLIT_DATE, True)
  
  # remove customer_id are used in train mode
  df_predict_ml1 = df_predict_ml1[~df_predict_ml1.customer_id.isin(df_train_ml1.customer_id)]
  df_cus_bf_gte2 = df_cus_bf_gte2[~df_cus_bf_gte2.customer_id.isin(df_cus_bf_gte2_train.customer_id)]  
  
  # Featuring: PREDICT mode
  df_features_train = create_features_ml1(df_train_ml1, df_cus_bf_gte2_train, SPLIT_DATE, 'TRAIN')
  df_features = create_features_ml1(df_predict_ml1, df_cus_bf_gte2, SPLIT_DATE, 'PREDICT')
  
  # predict for all customer having two purchase or more before split time --> create some new sample
  X = df_features[['r_clus', 'f_clus', 'm_clus', 'daydiff_mean', 'daydiff_std']]
  #######################################################################
  # Load model and predict
  model=pickle.load(open(savingmodel_path_ml1,'rb'))
  y_label_pred = model.predict(X)
  df_features['y_label'] = y_label_pred
  
  with open(model_path+'/model01_cluster_accuracy_'+modeln+'.txt', 'r') as fr:
    cluster_name = [int(e) for e in fr.readline().split('#')]
    f1_score = [float(e) for e in fr.readline().split('#')]
    
  print_debug("[1] cluster_name = {}  \n[2] f1_score = {}".format(cluster_name,f1_score))
  
  ########################################
  #ONLY CHOOSE UNLABELING DATA OF 0CLUSTER 1 (HIGH ACCURACY OF 0.61)
  #for i in f1_score(y, ypred, average=None)>0.55:
  for f1_score_i, clus_i in zip(np.array(f1_score)>0.55,cluster_name):
    if f1_score_i:
      df_features = df_features[df_features.y_label==clus_i]
  
  #create num of day after split date the customer could purchase
  df_features['nextpurchase_day'] = (df_features.y_label - 1)*CLUSTER_PERIOD - df_features.r + int(CLUSTER_PERIOD/2)
  
  ########################################
  # Combine pseudo labeling + labeled data --> TRAIN
  df_features = pd.concat([df_features,df_features_train]).reset_index()
  print_debug("[XYZ] df_features = \n{} \n df_features_train = \n{}'".format(df_features.head(),df_features_train.head()))
  df_features.to_csv("./check_features.csv")
  # train model with TRAIN mode
  train_test_ml1(df_features,'TRAIN',savingmodel_path_ml1)   
  
  return True
  
#################################################################################################################
# MAIN
#################################################################################################################


print_debug("####################################################")
print_debug("# Find customer with usual in purchase and Predict for them")
print_debug("####################################################")

#df_pre_processing = data_pre_processing(df)
#a = lib1.predict_cus_freq(df_pre_processing,'EVAL')
#print(a)
#sys.exit(1)

SPLIT_DATE=''
END_DATE=''
CURRENT_DATE=''
INI_SPLIT_TIME=''
TEST_SPLIT_DATE=''

if not commitonly:
  # PARAMETER
  SPLIT_DATE   = pd.to_datetime(df.date_order.max()) + datetime.timedelta(days=-SPLIT_BACKWARD)  #datetime.datetime(2019,7,1)
  END_DATE     = pd.to_datetime(df.date_order.max())                                             #datetime.datetime(2019,12,22)
  #CURRENT_DATE = datetime.datetime(2020,2,25)
  year = datetime.datetime.today().year
  month = datetime.datetime.today().month
  day = datetime.datetime.today().day
  CURRENT_DATE = datetime.datetime(int(year),int(month),int(day)) + datetime.timedelta(days=-backsplit)
  
  INI_SPLIT_TIME  = pd.to_datetime(df.date_order.min()) + datetime.timedelta(days=20)            #datetime.datetime(2018,1,20)
  TEST_SPLIT_DATE = pd.to_datetime(df.date_order.max()) + datetime.timedelta(days=-SPLIT_BACKWARD-2)  #datetime.datetime(2019,8,15)
  TOTAL_PUR_DAY_UNNORMAL_THRESHOLD = int((pd.to_datetime(df.date_order.max()) - pd.to_datetime(df.date_order.min())).days/PUR_PERIOD)
  #TOTAL_DAY_OF_DATA = int((pd.to_datetime(df.date_order.max()) - pd.to_datetime(df.date_order.min())).days)+1
  TOTAL_DAY_OF_DATA = int((pd.to_datetime(df.date_order.max()) - pd.to_datetime(df.date_order.min())).days/BASE_TIME_BLOCK)
  
  print_debug("[main()] Important datetime")
  print_debug("[main()] SPLIT_DATE     : {}".format(SPLIT_DATE))
  print_debug("[main()] END_DATE       : {}".format(END_DATE))
  print_debug("[main()] CURRENT_DATE   : {}".format(CURRENT_DATE))
  print_debug("[main()] INI_SPLIT_TIME : {}".format(INI_SPLIT_TIME))
  print_debug("[main()] TEST_SPLIT_DATE: {}".format(TEST_SPLIT_DATE))

#df['order_date'] = pd.to_datetime(df['order_date'])

# mode
#     - train_test_split_mode: split data in two files (File A 80% data, File B 20% data), training+testing with data A and more testing with data B
#     - train_mode: train with full data
#     - test_mode: run test only for any data file with saved model
#     - predict_mode: predict with split date (current day)

# model 01 saved path
#savingmodel_xpath_ml1 = savemodel_path+"/ml1_xgboost_next_purchasing_forecast_xmode.sav"
# model 02 saved path
#savingmodel_xpath_ml2 = savemodel_path+"/ml2_xgboost_next_purchasing_forecast_xmode.sav"

savingmodel_path_ml1 = ''
savingmodel_path_ml2 = ''
if pseudo_labeling:
  # model 01 saved path
  savingmodel_path_ml1 = savemodel_path+"/ml1_"+models+"_np_forecast_pseudo.sav"
  # model 02 saved path
  savingmodel_path_ml2 = savemodel_path+"/ml2_"+models+"_np_forecast.sav"
else:
  # model 01 saved path
  savingmodel_path_ml1 = savemodel_path+"/ml1_"+models+"_np_forecast"
  # model 02 saved path
  savingmodel_path_ml2 = savemodel_path+"/ml2_"+models+"_np_forecast"
  
predict_out_path = predict_output+"/"+input.split('.')[-1]+"_"+skf_en+"_"+models+"_PRED_nextpurchase.csv" 
predict_out_path_final = predict_output+"/"+input.split('.')[-1]+"_"+models+"_PRED_nextpurchase_refSTATS.csv" 

dfa = pd.DataFrame()
dfb = pd.DataFrame()

# Step 2: check `mode` and run
#------------------------------------------------------------------------------------------------------------------------------------
# TRAIN TEST SPECIAL MODE

if train_test_split_mode:
  #get customer id for FileB
  customer_fileb = np.array(random.sample(list(df.customer_id.unique()), int(df.customer_id.nunique()/5)))
  #len(customer_id_test)
  dfb = df[ df.customer_id.isin(customer_fileb)]
  dfa = df[~df.customer_id.isin(customer_fileb)]

  ##################################################################################################
  #                            A. MODEL 01 : TRAIN + TEST                                         ##
  #                   Predict for customer with at least two purchased                            ##
  ##################################################################################################  

  # TRAIN Model with dfa
  #response = main_train_ml1 (dfa,SPLIT_DATE,savingmodel_path_ml1)
  response = main_train_ml1 (df,SPLIT_DATE,SLIDE_BACKWARD_STEP,savingmodel_path_ml1)
  if response:
    print_debug("-->> [TRAIN_TEST_SPLIT_MODE] MODEL 01 TRAIN COMPLETED")
   
  ### PSEUDO ###########
  # continue to train with pseudo labeling
  if pseudo_labeling:
    response = pseudo_labeling_train_ml1 (dfa, SPLIT_DATE, savingmodel_path_ml1)
    if response: 
      print_debug("-->> [TRAIN_MODE-PSEUDO] MODEL 01 WITH PSEUDO_LABELING TRAIN COMPLETED")
  ######################
  
  # TEST model with dfb
  response = main_test_ml1 (dfb,TEST_SPLIT_DATE,savingmodel_path_ml1)
  if response:
    print_debug("-->> [TRAIN_TEST_SPLIT_MODE] MODEL 01 INDEPENDENT TEST COMPLETED") 
  
  #!!! NOTE: Pseudo labeling method only has excellent effect for base_model with high accuracy (eg. >0.8)
  #      for the base model with low accuracy like this case (only 0.54), the Pseudo labeling does not have more improvable.
  # So Pseudo is not continue to deploy for model 2 (only 0.39 in accuracy)

  ##################################################################################################
  #                             B. MODEL 02 : TRAIN + TEST                                        ##
  #                      Predict for customer having 01 purchased                                 ##
  ##################################################################################################

  # TRAIN Model with dfa  
  #INI_SPLIT_TIME = datetime.datetime(2018,1,20)
  response = main_train_ml2 (dfa, INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2)
  if response:
    print_debug("-->> [TRAIN_TEST_SPLIT_MODE] MODEL 02 TRAIN COMPLETED")

  # TEST Model with dfb
  response = main_test_ml2 (dfb,INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2)
  if response:
    print_debug("-->> [TRAIN_TEST_SPLIT_MODE] MODEL 02 INDEPENDENT TEST COMPLETED")

#------------------------------------------------------------------------------------------------------------------------------------
# TRAIN MODE

if train_mode:
  ##################################################################################################
  #                            A. MODEL 01 : TRAIN WITH ALL DATA                                  ##
  #                       Predict for customer with at least two purchased                        ##
  ##################################################################################################

  #response = main_train_ml1 (df,SPLIT_DATE,savingmodel_path_ml1)
  response = main_train_ml1 (df,SPLIT_DATE,SLIDE_BACKWARD_STEP,savingmodel_path_ml1)
  if response:
    print_debug("-->> [TRAIN_MODE] MODEL 01 TRAIN COMPLETED")
  
  # continue to train with pseudo labeling
  if pseudo_labeling:
    response = pseudo_labeling_train_ml1 (df, SPLIT_DATE, savingmodel_path_ml1)
    if response: 
      print_debug("-->> [TRAIN_MODE-PSEUDO] MODEL 01 WITH PSEUDO_LABELING TRAIN COMPLETED")
    
  ##################################################################################################
  #                             B. MODEL 02 : TRAIN WITH ALL DATA                                 ##
  #                           Predict for customer having 01 purchased                            ##
  ##################################################################################################


  response = main_train_ml2 (df, INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2)
  if response:
    print_debug("-->> [TRAIN_MODE] MODEL 02 TRAIN COMPLETED")

#------------------------------------------------------------------------------------------------------------------------------------
# TEST MODE

if test_mode:

  # MODEL 01
  #TEST_SPLIT_DATE = datetime.datetime(2019,8,15)
  response = main_test_ml1 (df,TEST_SPLIT_DATE,savingmodel_path_ml1)
  if response:
    print_debug("-->> [TEST_MODE] MODEL 01 INDEPENDENT TEST COMPLETED")
  
  # MODEL 02
  response = main_test_ml2 (df,INI_SPLIT_TIME, END_DATE, SLIDE_STEP, savingmodel_path_ml2)
  if response:
    print_debug("-->> [TEST_MODE] MODEL 02 INDEPENDENT TEST COMPLETED")
  
#------------------------------------------------------------------------------------------------------------------------------------
# PREDICT ML MODE

if predict_ml_mode:
  ##################################################################################################
  #              C. PREDICT AND DUMP THE NEXT PURCHASE DAY TO CSV FILE (by customer)              ##
  #                                 Run prediction for new data file                              ##
  ##################################################################################################
  
  df_predict = main_predict_out (df, CURRENT_DATE, savingmodel_path_ml1, savingmodel_path_ml2) 
  df_predict.fillna(0,inplace=True)
  df_predict.to_csv(predict_out_path,index=False)

  predict_out_path_final = predict_output+"/"+input.split('.')[-1]+"_"+skf_en+"_"+models+"_PRED_nextpurchase.csv" 
#------------------------------------------------------------------------------------------------------------------------------------
# PREDICT MODE

if predict_mode:
  ##################################################################################################
  #              C. PREDICT AND DUMP THE NEXT PURCHASE DAY TO CSV FILE (by customer)              ##
  #                                 Run prediction for new data file                              ##
  ##################################################################################################
  
  df_predict = main_predict_out (df, CURRENT_DATE, savingmodel_path_ml1, savingmodel_path_ml2) 
  df_predict.fillna(0,inplace=True)
  df_predict.to_csv(predict_out_path,index=False)
  predict_stats_mode = True
  predict_merge = True
 
  #predict_out_path_final = predict_output+"/"+input.split('.')[-1]+"_"+skf_en+"_"+models+"_PRED_nextpurchase_FINAL.csv" 
#------------------------------------------------------------------------------------------------------------------------------------
# PREDICT STATS MODE

if predict_stats_mode:
  ##################################################################################################
  #              D. PREDICT AND DUMP THE NEXT PURCHASE DAY TO CSV FILE (by customer)              ##
  #                                 Run prediction for new data file                              ##
  ##################################################################################################
  df_pre_processing = data_pre_processing(df)
  print_debug ('[main] customer after preprocessing: {}'.format(df_pre_processing.customer_id.nunique()))
  df_stats = df_pre_processing[df_pre_processing.date_order<=CURRENT_DATE]
  response = predict_customer_frequent_purchase_active(df_stats)
  
  if response:
    print_debug("-->> [PREDICT_STATS] PREDICT CUSTOMER NEXT PURCHASE THROUGH FREQUENTLY-PURCHASE STATISTIC COMPLETED")

  predict_out_path_final = predict_output+"/stats_freq_customer_PRED_nextpurchase.csv"
  
#--------------------------------------
# MERGE RESULT  
if predict_merge:
  df_pred_ml = pd.read_csv(predict_out_path,index_col=None)
  df_pred_stats = pd.read_csv(predict_output+"/stats_freq_customer_PRED_nextpurchase.csv",index_col=None)
  
  df_pred_ml_pass = df_pred_ml[~df_pred_ml.customer_id.isin(df_pred_stats.customer_id)]
  df_pred_ml_comp = df_pred_ml[df_pred_ml.customer_id.isin(df_pred_stats.customer_id)]
  
  df_pred_acc_check = pd.merge(df_pred_ml_comp[['customer_id','accuracy']],df_pred_stats[['customer_id','accuracy_stats']], on='customer_id',how='left')
  df_pred_acc_check.loc[:,'check'] = df_pred_acc_check.loc[:,'accuracy_stats'] < df_pred_acc_check.loc[:,'accuracy']
  
  df_pred_1 = df_pred_ml_comp[df_pred_ml_comp.customer_id.isin(df_pred_acc_check[df_pred_acc_check.check==True].customer_id)]
  df_pred_2 = df_pred_stats[df_pred_stats.customer_id.isin(df_pred_acc_check[df_pred_acc_check.check==False].customer_id)]
  df_pred_2.columns = ['customer_id','pred_standing_date','nextpurchase_day_pred','from_day','to_day','from_date','to_date','accuracy','cus_type']
  df_pred_final = pd.concat([df_pred_ml_pass,df_pred_1,df_pred_2])
  df_pred_final.to_csv(predict_output+"/"+input.split('.')[-1]+"_"+skf_en+"_"+models+"_PRED_nextpurchase_refSTATS.csv" ,index=False)  
  
  # final OUTPUT OF PREDICTION
  predict_out_path_final = predict_output+"/"+input.split('.')[-1]+"_"+skf_en+"_"+models+"_PRED_nextpurchase_refSTATS.csv" 
  
#------------------------------------------------------------------------------------------------------------------------------------
# COMMIT

if commit:
  ##################################################################################################
  #                              D. COMMIT PREDICT RESULT TO BIG QUERY                            ##
  #                                 Commit result file to Big Query                               ##
  ##################################################################################################

  print ('##################################')
  print ('# COMMIT RESULT TO SANDBOX')
  print ('##################################')

  src_path = predict_out_path_final
  dest_name = input+"_PRED_nextpurchase"  
  os.system('python commit_bq.py --source {} --destination {}'.format(src_path,dest_name))  
  
print ('##################################')
print ('# ENDING ! CONGRATULATION')
print ('##################################')











