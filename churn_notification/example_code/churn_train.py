########################################################################################################################
# CHURN PREDICTION TRAIN
# File name : churn_train.py
# Author    : Nhan Thanh Ngo
########################################################################################################################

###################################################
# IMPORT LIBRARY
#
#

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import sys
import shutil

import yaml
from general_func import eda_data
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle

####################################################
# General

NUM_OBSERVED_DAY = 365

####################################################
# GetOptions
import sys, getopt
import glob
import time

global debug
debug = False
input = "need_to_specified_table"
query = False
limit = ''
commit = False
commitonly = False
weight_inherit = False

                      
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hq:i:w:c:u:d', ['help','query','input=','weight_inherit','commit','commitonly','debug'])
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-w', '--weight_inherit'):
    weight_inherit = True
  if opt in ('-c', '--commit'):
    commit = True
  if opt in ('-u', '--commitonly'):
    commitonly = True
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)

if commitonly:
  mode = 'COMMIT' 
  query = False
  limit = ''
  commit = True


print ("CHURN PREDICTION TRAIN")
print ("This is {}\n".format(filename))
  
###########################################
#create DEBUG directory
#
homepath = "./"+input.split('.')[-1]+"_churn_train"
temppath = outpath = clusfile_path = modelpath = backuppath = ''

model_info_dict = {}
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


#######################################################
# FUNCTION
#######################################################

#######################################################
# Function: setup_folder(homepath)
# Description: 
#   set up neccessary folder for
#

def setup_folder(homepath):

  global temppath, outpath, clusfile_path, modelpath, backuppath
  
  # homepath
  if os.path.exists(homepath):
    print ("\'{}\' is already EXISTED!".format(homepath))
  else:
    os.mkdir(homepath)
    print ("\'{}\' is CREATED!".format(homepath)) 

  # temppath
  temppath = homepath+'/temp'
  if os.path.exists(temppath):
    print ("\'{}\' is already EXISTED!".format(temppath))
  else:
    os.mkdir(temppath)
    print ("\'{}\' is CREATED!".format(temppath))

  # modelpath
  modelpath = homepath+'/model'
  if os.path.exists(modelpath):
    print ("\'{}\' is already EXISTED!".format(modelpath))
  else:
    os.mkdir(modelpath)
    print ("\'{}\' is CREATED!".format(modelpath))

  # outpath
  outpath = homepath+'/output'
  if os.path.exists(outpath):
    print ("\'{}\' is already EXISTED!".format(outpath))
  else:
    os.mkdir(outpath)
    print ("\'{}\' is CREATED!".format(outpath))

  # backup
  backuppath = homepath+'/backup'
  if os.path.exists(backuppath):
    print ("\'{}\' is already EXISTED!".format(backuppath))
  else:
    os.mkdir(backuppath)
    print ("\'{}\' is CREATED!".format(backuppath))
    
  # clusterfile
  clusfile_path = homepath+"/clusfile"
  if os.path.exists(clusfile_path):  
    print ("\'{}\' is already EXISTED!".format(clusfile_path))
    shutil.rmtree(clusfile_path)
    os.mkdir(clusfile_path)
  else:
    os.mkdir(clusfile_path)
    print ("\'{}\' is CREATED!".format(clusfile_path))  



######################################################
# Function: query_data(input,query)
# Description:
#     query data from DW to local PC
# 
# Input: 
#    - input: table name in DW (eg. DW.customer_recommend_datas
#    - query: True/False
# Output:
#    query data and store in local location
#    

def query_data(input, query):

  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  # query in DW (credentials is available in production machine)
  client = bigquery.Client()
  
  sql = "SELECT * FROM "+input
  bq_cus_purchase = input.split('.')[-1]+".csv"

  if query:
    # Run a Standard SQL query with the project set explicitly
    print("Querying data from sandbox...\nit will take a few minutes...")
    df = client.query(sql).to_dataframe() 
    #df = client.query(sql, project=project_id).to_dataframe()
    
    if not glob.glob(bq_cus_purchase):
      print ("[INFO] No BigQuery datafile available")
    else:
      print("[INFO] Remove exist bq datafile")
      os.remove(bq_cus_purchase)
    
    print("[INFO] Store query data from Big Query to file")
    df.to_csv(bq_cus_purchase,index=False)
  else:
    print("[INFO] Read input data from offline file, need update please run again with -q to query new data from Big Query")
    print("Read offline input data file...\nit will take a few minutes...")
    df = pd.read_csv(bq_cus_purchase, index_col=None)

  return df

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def labelling(df):
  '''
    labelling churn and alert signal for all customers.
    input:
      df: ['customer_id', 'date_order', 'product_id', 'items', 'size', 'quantity']
    output:
      df: ['customer_id','churn','alert']
  '''
  #####################################
  def purx_summary(df_gte2, N):
    '''
      get numday value covering 95% customer in Nth purchasing distance. 
      input: 
        - df_gte2 (customer purchased greater than 2 times): including distance btw 2 first purchases to 2 last purchases
        - N (nth distance of customer)
      output:
        - number of day which cover 95% of customer in nth distance
    '''
    
    df_dist_x = df_gte2[df_gte2.indexing==N]
    df_dist_x = df_dist_x.sort_values(['pur_dist'],ascending=True).reset_index()
    numday_churn = df_dist_x.iloc[int(df_dist_x.shape[0]*0.95)].pur_dist
    return numday_churn
  
  ###################################
  # begin 

  # get relevant columns for processing
  column = ['customer_id','date_order','quantity']
  df = df.groupby(['customer_id','date_order'])['quantity'].sum().reset_index().sort_values(['customer_id','date_order'])

  # compute distance between two nearest purchases
  df['date_sh1'] = df.groupby(['customer_id']).date_order.shift(1)
  df['pur_dist'] = df['date_order'] - df['date_sh1']

  # fill NaT cell with 0
  df.fillna(pd.Timedelta(seconds=0),inplace=True)
  df['pur_dist'] = df['pur_dist'].apply(lambda x:  int(x.days))

  # compute purchasing frequency of customer
  df_freq = df.groupby('customer_id').date_order.count().reset_index()
  df_freq.columns = ['customer_id','freq']
  df = pd.merge(df,df_freq, on='customer_id',how='left')
  
  df = df[['customer_id','date_order','quantity','pur_dist','freq']]
  df.date_order = pd.to_datetime(df.date_order)

  total_numday = (df.date_order.max() - df.date_order.min()).days

  ############################################################################
  # process for customer who has one purchase and greater than two purchase
  ############################################################################
  # 1 purchase
  df_e1 = df[df['freq'] == 1]
  df_e1 = df_e1[['customer_id','pur_dist','freq']]

  # from 02 purchases
  df_gte2 = df[df.pur_dist!=0]
  df_gte2['indexing'] = df_gte2.groupby('customer_id').cumcount()+1

  df_label = df_gte2.groupby(['customer_id','freq']).pur_dist.max().reset_index()
  df_label = pd.concat([df_label,df_e1])

  # call function above for number of freq max and churn value
  numday_churn_arr = []
  for i in np.arange(df_gte2.freq.max()-1):
    numday_churn_arr.append(purx_summary(df_gte2,i+1))

  # compute df looup value for churn
  # customer has 1 purchase get churn distance of customer having 2 purchasing
  # customers have high frequency (>15) and low distance => limit churn at 10 days, alert at 8 days (just for relevant action)
  df_churn_lookup = pd.DataFrame({'freq':np.arange(df_gte2.freq.max()-1)+1,'numday_prechurn':np.array(numday_churn_arr)})  
  df_churn_lookup.loc[(df_churn_lookup.freq > 15) & (df_churn_lookup.numday_prechurn < 10) ,'numday_prechurn'] = 10 
  
  # save data for labeling to file
  df_churn_lookup.to_csv(modelpath+'/label_data.csv', index = False)
  
  line_plot(df_churn_lookup.freq.values, df_churn_lookup.numday_prechurn.values, temppath+'/critical_numday_PLOT.png','Critical Numday of Return equivalent Customer Purchasing Frequency')

  #df_churn_lookup[df_churn_lookup['freq']<50].plot.bar(x='freq',y='numday_prechurn')

  # create new columns 
  df_label = pd.merge(df_label,df_churn_lookup,on='freq',how='left')
  df_label['numday_critical'] = df_label[["pur_dist", "numday_prechurn"]].max(axis=1)
  df_label['numday_alert'] = df_label['numday_critical'].apply(lambda x: np.round(x*0.8))
  df_label['numday_churn'] = df_label['numday_critical'].apply(lambda x: np.round(x*1.2))
  
  df_recency = df.groupby(['customer_id']).date_order.max().reset_index()
  df_recency['today'] = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d")) # f"{datetime.datetime.now():%Y-%m-%d}"
  df_recency['R'] = (df_recency['today'] - df_recency['date_order']).astype('timedelta64[D]').astype(int)

  df_label = pd.merge(df_label, df_recency[['customer_id','R']], on='customer_id', how='left')
  df_label['alert'] = (df_label['R'] >= df_label['numday_alert'])
  df_label['critical'] = (df_label['R'] >= df_label['numday_critical'])
  df_label['churn'] = (df_label['R'] >= df_label['numday_churn'])

  df_label['alert']    = df_label['alert'].apply(lambda x: 1 if x else 0)  
  df_label['critical'] = df_label['critical'].apply(lambda x: 1 if x else 0)
  df_label['churn']    = df_label['churn'].apply(lambda x: 1 if x else 0)

  df_label['churn'] = df_label['churn'].astype(int)
  df_label['critical'] = df_label['critical'].astype(int)
  df_label['alert'] = df_label['alert'].astype(int)
  
  df_label.to_csv(temppath+'/df_label_full.csv',index=False)
  df_label[['customer_id','churn','critical','alert']].to_csv(temppath+'/df_label.csv',index=False)
  
  return df_label[['customer_id','churn','critical','alert']]
 

def feature_engineering(df):

  '''
    feature engineering for time series model (LSTM)
    input:
      df: ['customer_id', 'date_order', 'product_id', 'items', 'size', 'quantity']
    output:
      df: ['customer_id',0,1,2,3,4]
  '''
  
  column = ['customer_id','date_order','quantity']
  df = df.groupby(['customer_id','date_order'])['quantity'].sum().reset_index().sort_values(['customer_id','date_order'])

  # CREATE TABLE OF TIME RANGE FOR MIN DAY TO MAX DAY
  # AND TIMESCALES FOR PERIOD set by PARAMETERs
  # contineous day list 
  TIMESERIES_TIMESCALES = 7

  lastday = df.date_order.max()
  total_numday = (df.date_order.max() - df.date_order.min()).days + 1

  if (total_numday%TIMESERIES_TIMESCALES)!= 0:
    total_numday = TIMESERIES_TIMESCALES*(int(total_numday/TIMESERIES_TIMESCALES)+1)

  total_timescales = int(total_numday/TIMESERIES_TIMESCALES)

  #datetime.datetime(2019,9,13)
  date_list = [lastday - datetime.timedelta(days=x) for x in range(total_numday)]
  date_list = [date_list[i].strftime('%Y-%m-%d') for i in range(len(date_list))]

  df_day = pd.DataFrame()
  df_day['date_order'] = date_list
  #df_day = df_day.sort_values(['date_order'], ascending=True)

  df_day['indexing'] = np.arange(total_numday)
  #df_day['indexing'] = df_day.groupby('customer_id').cumcount()+1
  df_day['timescales'] = df_day['indexing'].apply(lambda x: int(x/TIMESERIES_TIMESCALES))
  df_day['date_order'] = pd.to_datetime(df_day.date_order)

  # MERGE TABLE
  df = pd.merge(df,df_day[['date_order','timescales']],on='date_order',how='left')

  # df quantity by timescales instead of date_order
  df_ts = df.groupby(['customer_id','timescales'])['quantity'].sum().reset_index()
  df_ts.quantity = df_ts.quantity.astype(int)

  # dummy customer
  df_dummy = pd.DataFrame({
    'customer_id' : ['D']*(total_timescales),
    'timescales'  : np.arange(total_timescales)
  })

  # CONCAT DUMMY
  df = pd.concat([df_ts,df_dummy])

  # PIVOT FOR TIME SERIES INPUT OF LSTM
  df_pivot = pd.pivot_table(df, values='quantity', index=['customer_id'], columns=['timescales'], aggfunc=np.sum, fill_value=0).reset_index()
  df_pivot = df_pivot[df_pivot.customer_id!='D'] # remove dummy customer
  column = [str(col) for col in df_pivot.columns.values]
  df_pivot.columns = column

  # RE-ORDER COLUMNS
  column = df_pivot.columns.values
  column = column[::-1]
  adj_col = np.concatenate((['customer_id'],column[:-1]),axis=0)
  df_pivot = df_pivot[adj_col]
  
  return df_pivot

################################################################################################


def build_model(INPUT_LEN,OUTPUT_NUM,BATCH_SIZE):

    model = keras.Sequential()
    # Add an Embedding layer expecting input vocab of size TIMESCALES_DIM*BATCH_SIZE, and
    # output embedding dimension of size OUTPUT_NUM.

    model.add(layers.Embedding(input_dim=INPUT_LEN*BATCH_SIZE, output_dim=OUTPUT_NUM))

    # Add a LSTM layer with TIMESCALES_DIM internal units.
    model.add(layers.LSTM(INPUT_LEN))
    #model.add(layers.LSTM(INPUT_LEN), input_shape=(INPUT_LEN,1))
    
    #model.add(layers.Dense(32, activation='relu'))
    
    # Add a Dense layer with OUTPUT_NUM units.
    model.add(layers.Dense(OUTPUT_NUM, activation='sigmoid'))

    model.summary()
    
    return model


def normalization(features, labels, label_name):

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.05, random_state=42, stratify=labels)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    X_train = X_train.astype(np.int32)
    X_test  = X_test.astype(np.int32)
    X_valid = X_valid.astype(np.int32)
    y_train = y_train.astype(np.int32)
    y_test  = y_test.astype(np.int32)
    y_valid = y_valid.astype(np.int32)    

    transformer = StandardScaler().fit(X_train)
    X_train = transformer.transform(X_train)
    X_test  = transformer.transform(X_test)
    X_valid = transformer.transform(X_valid)
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32) 
    X_test  = tf.convert_to_tensor(X_test, dtype=tf.float32) 
    X_valid = tf.convert_to_tensor(X_valid, dtype=tf.float32) 
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32) 
    #y_test  = tf.convert_to_tensor(y_test, dtype=tf.float32) 
    y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float32) 

    pickle.dump(transformer, open(modelpath+'/transformer.pkl', 'wb'))
    
    return X_train, X_test, X_valid, y_train, y_test, y_valid

    
def train_model(BUILD_MODEL_PARA, X_train, y_train, X_valid, y_valid, label_name, weight_inherit):

    #model = build_model(allow_cudnn_kernel=True)
    INPUT_LEN = BUILD_MODEL_PARA[0]
    OUTPUT_NUM = BUILD_MODEL_PARA[1]
    BATCH_SIZE = BUILD_MODEL_PARA[2]
    
    # define model structure
    model = build_model(INPUT_LEN,OUTPUT_NUM,BATCH_SIZE)  
    
    if weight_inherit:
      print("Load previous weight and retrain model...")
      # load previous model weight
      # model.load_weights(modelpath+'/'+label_name+'_model.ckpt')  
      # cannot do this because INPUT_LEN is difference, if you fit data in same length, open the load_weights command above to run.
      #transformer = pickle.load(open(modelpath+'/transformer.pkl', 'rb'))
      
    else:
      print("Train with white model structure...")

    
    # define loss function
    loss_func = tfa.losses.SigmoidFocalCrossEntropy(gamma=3)

    # model compile with loss function, optimizer and metric
    model.compile(
        loss=loss_func,
        optimizer="adam",
        metrics=["accuracy"],
    )

    # early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode = 'min',
        patience=5,
        verbose=0,
        baseline=None,
        restore_best_weights=True,
    )
	
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.00000000001)
							  
    # checkpoint to save file
    checkpoint_file = modelpath+'/'+label_name+'_model.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                 save_weights_only=True,
                                                 verbose=0)   
        
    # model train by fit                 
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=BATCH_SIZE, epochs=100, callbacks=[early_stop,checkpoint, reduce_lr])
     
    #model.save_weights(modelpath+'/'+label_name+'_model.pkl')
    
    return model
   

def line_plot(x,y,savepath,title):

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)

  # plotting
  plt.title(title)  
  plt.xlabel("X axis")  
  plt.ylabel("Y axis")  
  plt.plot(x, y, color ="red")
  chart.savefig(savepath)


def train_model_main(features, label, label_name, weight_inherit):

  global model_info_dict
  
  INPUT_LEN = features.shape[1] #len(df.columns.values)-4 #4 = 3 label columns + customer_id
  OUTPUT_NUM = 1
  BATCH_SIZE = 100 #number of customers
  
  
  print('TRAINNING MODEL FOR LABEL: {}'.format(label_name))
  #with open(modelpath + '/model_feature_len.txt', 'w') as logfile:
  #  logfile.write(str(INPUT_LEN))
  
  # model build, normalization, train
  BUILD_MODEL_PARA = [INPUT_LEN, OUTPUT_NUM, BATCH_SIZE]
  
  X_train, X_test, X_valid, y_train, y_test, y_valid = normalization(features, label, label_name)
  model = train_model(BUILD_MODEL_PARA, X_train, y_train, X_valid, y_valid, label_name, weight_inherit)

  # evaluation
  loadmodel = build_model(INPUT_LEN, OUTPUT_NUM, BATCH_SIZE)
  loadmodel.load_weights(modelpath+'/'+label_name+'_model.ckpt')
  y_pred = loadmodel(X_test).numpy()

  #############################################
  # choose Threshold
  acc_score_threshold = []
  for th in np.arange(0, 1, 0.01):
    y_pred_i = [1 if i > th else 0 for i in y_pred]
    acc_score_threshold.append(accuracy_score(y_test, y_pred_i))
  
  line_plot(np.arange(0, 1, 0.01),acc_score_threshold,temppath+'/'+label_name+'_accuracy_by_threshold_PLOT.png',label_name+' Accuracy by Threshold')
  
  index_max = np.argmax(acc_score_threshold,axis=0)  
  BEST_THRESHOLD = np.arange(0, 1, 0.01)[index_max]
  
  model_info_dict[label_name+'_THRESHOLD'] = BEST_THRESHOLD  
  #with open(modelpath + '/'+label_name+'_THRESHOLD.txt', 'w') as logfile:
  #  logfile.write(str(BEST_THRESHOLD))
    
  ############################################  
  # get final y_pred
  y_pred = [1 if i > BEST_THRESHOLD else 0 for i in y_pred]  
  #rate = np.sum(y_pred == y_test)/len(y_pred)  
  
  model_info_dict[label_name+'_ACCURACY'] = accuracy_score(y_test, y_pred)  
  
  ###########################################
  # report to log file
  current_date = datetime.datetime.now().strftime('%Y-%m-%d')
  with open(modelpath + '/training_accuracy_report.txt', 'a') as logfile:
    logfile.write('{} MODEL ({})\n'.format(label_name,current_date))
    logfile.write('[{}] Accuracy Score: {}\n'.format(label_name, accuracy_score(y_test, y_pred)))
    logfile.write('[{}] Classication Report \n{}\n'.format(label_name, classification_report(y_test, y_pred)))
    logfile.write('[{}] Confustion Matrix \n{}\n\n'.format(label_name, confusion_matrix(y_test, y_pred)))
  
  print('###############################################')
  print('[{}] Accuracy Score: {}'.format(label_name, accuracy_score(y_test, y_pred)))
  print('[{}] Classication Report \n{}'.format(label_name, classification_report(y_test, y_pred)))
  print('[{}] Confustion Matrix \n{}'.format(label_name, confusion_matrix(y_test, y_pred, labels=[0,1])))


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

print ('##################################################################')
print ('# [1] FOLDER SETUP/QUERY DATA')
print ('##################################################################')
setup_folder(homepath) 
df = query_data(input, query)
#df = df[df.customer_id.isin(df.customer_id.values[:30000])]

current_day_str = datetime.datetime.now().strftime('%y-%m-%d')

# clusterfile
backup_subdir = backuppath+"/"+current_day_str
if os.path.exists(backup_subdir):  
  print ("\'{}\' is already EXISTED!".format(backup_subdir))
else:
  os.mkdir(backup_subdir)
  print ("\'{}\' is CREATED and Backing up current release model".format(backup_subdir))  
  #backup model
  for file in os.listdir(modelpath):
    filename = file.split('/')[-1]
    shutil.copy2(modelpath+'/'+filename, backup_subdir+'/'+filename)

print ('##################################################################')
print ('# [2] EDA DATA')
print ('##################################################################')
df = eda_data(df)
df.date_order = pd.to_datetime(df.date_order) 

# only train with data in one nearest year
#start_date = df.date_order.max() - datetime.timedelta(days=NUM_OBSERVED_DAY)
#df = df[df.date_order>=start_date]
#EXPERIMENTAL RESULT: limit data in 12months in not good, it bias customer insight.

print ('##################################################################')
print ('# [3] LABELING')
print ('##################################################################')

df_label = labelling(df)

print ('##################################################################')
print ('# [4] FEATURE ENGINEERING')
print ('##################################################################')

df_feature_eng = feature_engineering(df)

print ('##################################################################')
print ('# [5] MERGE FEATURES and LABELLING')
print ('##################################################################')

df = pd.merge(df_feature_eng,df_label,on='customer_id',how='left')
del df_label
del df_feature_eng

df.to_csv(temppath+'/df_features_labels.csv',index=None)

print ('##################################################################')
print ('# [6] BUILD, TRAIN, EVALUATION, RELEASE MODEL')
print ('##################################################################')

#INPUT_LEN = len(df.columns.values)-4 #4 = 3 label columns + customer_id
#OUTPUT_NUM = 1
#BATCH_SIZE = 100 #number of customers

# features
model_input_col = [col for col in df.columns.values if col not in ['customer_id','alert','critical','churn']]
model_info_dict['INPUT_LEN'] = len(model_input_col)
model_input_matrix = df[model_input_col].values
print('features matrix shape: {}'.format(model_input_matrix.shape))

print('model_info_dict:', model_info_dict)
# label
churn_label = df['churn'].values
train_model_main(model_input_matrix, churn_label, 'churn', weight_inherit)
print('Model for churn label completed!')
print('model_info_dict:', model_info_dict)

# critical
critical_label = df['critical'].values
train_model_main(model_input_matrix, critical_label, 'critical', weight_inherit)
print('Model for critical label completed!')

# alert
alert_label = df['alert'].values
train_model_main(model_input_matrix, alert_label, 'alert', weight_inherit)
print('Model for alert label completed!')

print('model_info_dict:', model_info_dict)

with open(modelpath + '/model_info_dict.pkl', 'wb') as file:
  pickle.dump(model_info_dict, file)
file.close()


print ('##################################################################')
print ('# TRAINING COMPLETED!')
print ('##################################################################')