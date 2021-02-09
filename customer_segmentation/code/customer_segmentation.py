##############################################################
# CUSTOMER SEGMENTATION by PRODUCT
# Author: Nhan Thanh Ngo

import re
import pickle
import math
import unidecode

version = '1' 
versionx=re.sub("_",".",version)

GROUP_SIZE_LIMIT = 10000

####################################################
# GET OPTIONS
import sys, getopt
import glob
import time

#default
#group = 'Group 4'
kcluster = 4 
global custom_k 
custom_k = False
#groupx = ''
global debug
debug = False

debug_log = "DEBUG LOG:\n"
outdir = "."
input = "DW.table_name"
pivot_off = False
cus_outdir=0
commit = False
query = False
limit = ''
commitonly = False

from optparse import OptionParser

usage = "usage: %prog [options] arg1 arg2\n\n"\
        "Example:"\
		"\n\tpython %prog [-k 7] -i DW.inputtable --query --debug"\
		"\n\tpython %prog [-k 10] -i DW.table [-o ./folder_path] [--query] [--commit] "\
		"\nFeatures:"\
		"\n\tAllow user in INDICATING NAME OF TABLE in BigQuery to query data"\
		"\n\tAllow user in INDICATING OUT-DIRECTORY (output will include file and debug part)"\
		"\n\tAllow user in TURNING OFF QUERY data from BigQuery if NO NEED UPDATE DATA (previous queried data is stored offline already)"\
		"\n\tAllow user in COMMITING data to Bigquery"\
		"\n\tAllow user defines kcluster through -k, or kcluster will be automatically estimated based on input data."

parser = OptionParser(usage=usage)

#parser.add_option("-h", "--help",
#                  action="store_true", dest="verbose", default=True,
#                  help="print help information of the script such as how to run and arguments")

parser.add_option("-i", "--input",
                  default="`DW.rfm_segment_by_behavior`",
                  metavar="SANDBOX", help="Sandbox dataname"
                                         "[default: %default]")                        
parser.add_option("-k", "--kcluster",
                  action="store_false", dest="verbose",
                  help="number cluster that you want to generate by K-Mean Method [Default: AUTO ESTIMATION BASED ON DATA]")
                  
parser.add_option("-o", "--outdir",
                  default="./"+version+"_ouput",
                  metavar="OUTDIR", help="write output to OUTDIR"
                                         "[default: %default]")        
                                         
parser.add_option("-d", "--debug",
                  default="False",
                  help="Debug mode "
                       "[default: %default]")
                       
parser.add_option("-q", "--query",
                  default="False",
                  help="Query mode: query data from Big Query"
                       "[default: %default]")
                       
parser.add_option("-l", "--limit",
                  action="store_false", dest="verbose",
                  default="NO-LIMIT",
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
                       
parser.add_option("-n", "--nopivot",
                  default="pivot ON",
                  help="pivot OFF"
                       "[default: %default]")

try:
  opts, args = getopt.getopt(sys.argv[1:], 'hk:o:i:l:n:c:q:u:d', ['help','kcluster=','outdir=','input=','limit=','nopivot','commit','query','commitonly', 'debug'])
    
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-k', '--kcluster'):
    kcluster = int(arg)
    custom_k = True
  if opt in ('-d', '--debug'):
    debug = True
  if opt in ('-o', '--outdir'):
    outdir = str(arg) 
    #cus_outdir = 1
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-n', '--nopivot'):
    pivot_off = True
  if opt in ('-c', '--commit'):
    commit = True
  if opt in ('-q', '--query'):
    query = True
  if opt in ('-l', '--limit'):
    limit = " LIMIT "+str(arg)
  if opt in ('-u', '--commitonly'):
    commitonly = True
  if opt in ('-h', '--help'):
    parser.print_help()
    sys.exit(2)
      
if kcluster == None:
  sys.exit(3)

if outdir == ".":
  outdir = "./"+input.split('.')[-1]+"_"+version

print("##################################")
print("RUN INFORMATION")
print("K-cluster   : {}".format(kcluster))
print("Debug Mode  : {}".format(debug))
print("OUT-DIR     : {}".format(outdir))
print("Input       : {}".format(input))
print("Pivot OFF   : {}".format(pivot_off))
print("Query Mode  : {}".format(query))
print("Query LIMIT : {}".format(limit))
print("Commit      : {}".format(commit))
print("##################################")

#system.exit()
#------------------------------------------------------
# Rule of label
# Gt than 55 and 1st/2nd >2.0 -->

#-----------------------------------------------------
#create DEBUG directory
import os
import glob
import shutil

cwd = os.getcwd()
print ("at {}\nBEGIN...".format(cwd))

# creat debug log file
from datetime import date

today = date.today()
d_m_y = today.strftime("%d_%m_%Y")
print("d_m_y =", d_m_y)

#if cus_outdir:
if not os.path.exists(outdir):    
  os.mkdir(outdir)
    
# create debug folder    
debug_path = outdir+"/"+version+"_debug"
if os.path.exists(debug_path):
  print ("\'{}\' is already EXISTED! --> REMOVE OLD DIR...".format(debug_path))
  #shutil.rmtree(debug_path)
else:
  os.mkdir(debug_path)
  print ("\'{}\' is CREATED!".format(debug_path))

debug_log = 'cl_cus_cluster_debug.log'
if os.path.isfile(debug_path+"/"+debug_log):
  os.remove(debug_path+"/"+debug_log)
 
# create output folder
output_path = outdir+"/"+version+"_output"

if os.path.exists(output_path):
  print ("\'{}\' is already EXISTED! --> REMOVE OLD DIR...".format(output_path))
  #shutil.rmtree(output_path)
else:
  os.mkdir(output_path)
  print ("\'{}\' is CREATED!".format(output_path))

# create saved_model folder
model_path = outdir+"/"+version+"_model"

if os.path.exists(model_path):
  print ("\'{}\' is already EXISTED! --> REMOVE OLD DIR...".format(model_path))
  #shutil.rmtree(model_path)
else:
  os.mkdir(model_path)
  print ("\'{}\' is CREATED!".format(model_path))
 
#########################################################################
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
print(tf.__version__)

#import sklearn
# import k-means from clustering stage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

#!conda install python-graphviz --yes
import graphviz
from sklearn.tree import export_graphviz
import itertools

###################################################
# GENERAL FUNCTION
#

def print_debug (strcont):
  debugn = debug
  if debugn:
    print(strcont)  
    with open(debug_path+"/"+debug_log, "a") as logfile:
      logfile.write("\n"+strcont)

###################################################
# QUERY DATA FROM BIGQUERY
#
#

if not commitonly:
  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  sql = "SELECT * FROM "+input+limit
  bq_datafile = "./"+input+".csv"

  df_bq = pd.DataFrame()

  if query:

    # Run a Standard SQL query with the project set explicitly
    df_bq = client.query(sql).to_dataframe()
  
    if not glob.glob(bq_datafile):
      print ("[INFO] No BigQuery datafile available")
    else:
      print ("[INFO] Remove exist bq datafile")
      os.remove(bq_datafile)
    
    print ("[INFO] Store query data from Big Query to file")
    df_bq.to_csv(bq_datafile,index=False)
  else:
    print ("[INFO] Read input data from offline file, need update please run again with -q to query new data from Big Query")
    df_bq = pd.read_csv(bq_datafile, index_col=False)
  
  #df_bq = df_bq[:20000]
  
  #df_bq = group_df
  print_debug ("[INFO] input data for prediction as below\n{}".format(df_bq[:5]))
  print_debug ("[INFO] length of input data before process: {}".format(len(df_bq)))
  
  print_debug ("[INFO] df_bq.columns = {}".format(df_bq.columns))
  print_debug('[before] df_bq customer = {}'.format(df_bq.partner_id.nunique()))

  #############################################
  #df_bq = df_bq.loc[~df_bq.rfm_id.isnull(),:]
  #print_debug('[after] df_bq customer = {}'.format(df_bq.partner_id.nunique()))
  
  # Fill null value by null
  df_bq['rfm_id'].fillna(value='null', inplace=True)
  print ("rfm_id unique() = ",df_bq['rfm_id'].unique())
  #print ("[INFO] input data for prediction as below\n{}".format(df_bq[:25]))
  
  df_bq = df_bq[['partner_id', 'rfm_id','product_name','qty']]
  df_bq = df_bq.groupby(['partner_id', 'rfm_id','product_name']).qty.sum().reset_index()
  df_bq.columns = ['cus_id','rfm_group','style','quantity_item']
  print_debug("[INFO] df_bq.style.nunique = {}".format(df_bq['style'].nunique()))
  ##############################################

  # Remove sign from text
  df_bq['style'] = df_bq['style'].apply(lambda x: unidecode.unidecode(str(x)))
  #df_bq.iloc[:,2] = df_bq.iloc[:,2].apply(lambda x: unidecode.unidecode(str(x)))
  
  #df_bq = df_bq[df_bq.rfm_group == '7 to 30 days']
  ##############################
  # Split Big Group to Small Group. Setting Group at NUM_CUS_SPLIT = xxx (default: 15000)
  
  df_cus_cnt = df_bq.groupby(['rfm_group']).cus_id.nunique().reset_index()
  df_cus_cnt.columns = ['rfm_group','num_cus']
  print_debug("[INFO] df_cus_cnt = {}".format(df_cus_cnt))
    
  for group, num_cus_group in zip(df_cus_cnt.rfm_group.values, df_cus_cnt.num_cus.values):
    # process for each rfm_group
    if num_cus_group > GROUP_SIZE_LIMIT:
      customer_id_array = df_bq[df_bq.rfm_group == group].cus_id.unique()
      
      NUM_CUS_SPLIT = math.ceil(num_cus_group/(math.ceil(num_cus_group/GROUP_SIZE_LIMIT)))
	  
      for i in np.arange(math.ceil(len(customer_id_array)/NUM_CUS_SPLIT)):
        end = 0
        if i >= (len(customer_id_array)/NUM_CUS_SPLIT-1):
          end = None
        else:
          end = (i+1)*NUM_CUS_SPLIT

        df_bq.loc[df_bq.cus_id.isin(customer_id_array[i*NUM_CUS_SPLIT:end]),'rfm_group'] = str(group)+"_"+str(i)
		
  df_cus_cnt1 = df_bq.groupby(['rfm_group']).cus_id.nunique().reset_index()
  df_cus_cnt1.columns = ['rfm_group','num_cus']
  print_debug("[INFO] df_cus_cnt1 = {}".format(df_cus_cnt1))

  #####################################################
  # clear rfm
  df_bq["rfm_group"] = df_bq["rfm_group"].astype(str) 
  print_debug ("[INFO] Final input file: df_bq.shape {} \ndf_bq=\n{}".format(df_bq.shape,df_bq.head()))
  ###################################################
  # PIVOT TABLE PROCESS
  #
  #
  
  if not pivot_off:

    df_pivot = pd.pivot_table(df_bq, values='quantity_item', index=['rfm_group', 'cus_id'], columns=['style'], aggfunc=np.sum, fill_value=0)
    df_pivot.reset_index(inplace=True)

    while len(df_bq['cus_id'].unique()) == len(df_pivot):
      try:
        print_debug ("[PIVOT PASS] Length of original query data in unique equals length of pivot data")
        break
      except ValueError:
        print_debug ("ERROR: Length of df_bq.unique does not equal length of df_pivot. Please check...")

  else:
    df_pivot = df_bq.copy()

##################################################################################################
##################################################################################################
# General Functions
##################################################################################################

##################################################################################################
# Function: plot_pie (df, xlabel, ylabel, title, color='grey', fontsize=12, file_name='./')
#    - plot PIE chart 
#

# df_name is df with 2 columns, 1st cols is name, 2nd col is number
def plot_pie (df, xlabel, ylabel, title, color='grey', fontsize=12, file_name='./'):

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)

  print("Total items customer purchased: {}".format(df['count'].sum()))
  df_plot = df[df['count']!=0][-4:] 
  df_plot.loc['others'] = df.iloc[:-4,0].sum()

  label_list = np.array(df_plot.index.values)
  colors_array = cm.rainbow(np.linspace(0, 0.8, len(df_plot.index.values)))
  rainbow = [colors.rgb2hex(i) for i in colors_array]

  ax=df_plot['count'].plot(kind='pie',
          figsize=(9, 6),
          autopct='%1.1f%%',
          startangle=90,
          shadow=True,
          labels=label_list,         # turn off labels on pie chart
          pctdistance=.5,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
          colors=rainbow  # add custom colors
          #explode=explode_list # 'explode' lowest 3 continents
          )

  ax.set_ylabel(ylabel,fontsize=fontsize)
  ax.set_xlabel(xlabel,fontsize=fontsize)
  ax.set_title(title,fontsize=fontsize)
  
  chart.savefig(file_name)
  plt.close('all')

##################################################################################
# Function: plot_barh(df, xlabel, ylabel, title, color='grey', fontsize=12,file_name='./')
#    - plot bar chart by horizontal of dataframe
#

def plot_barh(df, xlabel, ylabel, title, color='grey', fontsize=12,file_name='./'):

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)
  
  ax=df.plot(kind='barh', 
             figsize=(12,6),
             fontsize=fontsize,
             color=color,
             rot=0
            )

  ax.set_ylabel(ylabel,fontsize=fontsize)
  ax.set_xlabel(xlabel,fontsize=fontsize)
  ax.set_title(title,fontsize=fontsize)

  idx_adj=-0.3
  for col in df.columns.values:
    #print (col)
    #plt.legend(col,loc=4)

    idx_adj+=0.3 #[-.4,-.1,.2]
    for index, value in enumerate(df[col]): 
        #print (index,value)
        #label = format(int(value), ',') # format int with commas (dau phay hang ngan)
        #print (label)
        #place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
        plt.annotate(value, xy=(value+0.3, index+idx_adj), color='black', fontsize=fontsize)   
  
  #save chart
  chart_name = debug_path+"/"+str(kcluster)+"_"+groupx+"_"+title+"_"+ylabel+"_"+xlabel+".png"
  chart.savefig(chart_name)
  chart.close()


##################################################################################
# Function: return_most_favourite(row, num_top_favourite)
#    - sort_values of dataframe, and get the top values of them
#    - return the name of top venues
# Usage:
#    Input: - row: dataframe with one row (one customer and product they purchase with quantity)
#           - num_top_favourite: number of top favourite product will be got
#        
#    Output: array of favourite products
#

def return_most_favourite(row, num_top_favourite):
  row_categories = row.iloc[1:]    
  row_categories_sorted = row_categories.sort_values(ascending=False)
  
  return [row_categories_sorted.index.values[0:num_top_favourite], row_categories_sorted.values[0:num_top_favourite]]

##################################################################################
# Function: create_group_label_product_quan_table(df,group,cluster)
#    - Labeling for each cluster 
#    - Labeling RULE is defined in code
# Usage:
#    Input: - df: dataframe [customer_id, productA, B, C, D, ....], value is quantity of purchasing
#           - group: rfm_group value
#           - kcluster: cluster value
#    Output: df_ret
#

def create_group_label_product_quan_table(df,group,cluster):
  #IMPORTANT NOTE: df in order from low to high
  
  cluster_type = ''
  cluster_labeling = ''
  total_percentage = 0
  
  df_ret = df.copy()
  total = df_ret['count'].sum(axis=0)
  df_ret['percentage']=df_ret['count']/total
  
  print("df_ret.nan.sum = {}".format(df_ret.isnull().sum().values.sum()))
  print("df_ret = \n{}".format(df_ret))
  if (df_ret.isnull().sum().values.sum() > 0):
    sys.exit(1)
	  
  df_ret.loc[:,'rfm_group']= [group]*len(df_ret)
  df_ret.loc[:,'Cluster Labels']= [cluster] * len(df_ret)
  df_ret.reset_index(level=0, inplace=True)
  df_ret.rename(columns={'index':'category'},inplace=True)
  
  #-----labeling------
  #DOM 1
  #rule for label: dominant: 1st > 0.55 and gt 2nd more than 2 
  #dom = 1
  if (((df_ret['percentage'][len(df_ret)-1]>0.55) & 
     (df_ret['percentage'][len(df_ret)-1]/df_ret['percentage'][len(df_ret)-2]>2)) | 
     (df_ret['percentage'][len(df_ret)-1]/df_ret['percentage'][len(df_ret)-2]>5)):
    cluster_type = 'dominance'
    cluster_labeling = cluster_type[:3]+'.'+df_ret['category'][len(df_ret)-1]
    
  #DOM 2
  elif ((((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2])>0.55) & 
       ((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2])/df_ret['percentage'][len(df_ret)-3]>2)) | 
       ((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2])/df_ret['percentage'][len(df_ret)-3]>5)):
    cluster_type = '2-dominances'
    cluster_labeling = cluster_type[:5]+'.'+df_ret['category'][len(df_ret)-1]+'.'+df_ret['category'][len(df_ret)-2]

  #DOM 3
  elif ((((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2]+df_ret['percentage'][len(df_ret)-3])>0.55) & 
       ((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2]+df_ret['percentage'][len(df_ret)-3])/df_ret['percentage'][len(df_ret)-4]>2)) | 
       ((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2]+df_ret['percentage'][len(df_ret)-3])/df_ret['percentage'][len(df_ret)-4]>5)):
    cluster_type = '3-dominances'
    cluster_labeling = cluster_type[:5]+'.'+df_ret['category'][len(df_ret)-1]+'.'+df_ret['category'][len(df_ret)-2]+'.'+df_ret['category'][len(df_ret)-3]
    
  else:
    cluster_type = 'mixture'
    cluster_labeling=cluster_type[:3]
    
    for i in range(len(df_ret)):
      #print(df_ret['percentage'][len(df_ret)-i-1])

      total_percentage+=df_ret['percentage'][len(df_ret)-i-1]
      cluster_labeling+='.'+df_ret['category'][len(df_ret)-i-1]
      if (total_percentage>=0.70) | (i>4):
        break
  
  print_debug ("\n[INFO] [{}, cluster {}] Type: {}, Labeling: {}".format(group,cluster,cluster_type,cluster_labeling))
  print ("[INFO] Labeling completed...\n")
  
  df_ret.loc[:,'cluster_type']= [cluster_type] * len(df_ret)
  df_ret.loc[:,'cluster_labeling']= [cluster_labeling] * len(df_ret)
  #----end labeling-------
  return df_ret  

##################################################################################
# Function: df_group_cluster(df,group,kcluster):
#    - main function clustering for one RFM_GROUP
#    - feature including:
#          [1] Data Preprocessing: drop line having all zeros
#          [2] Insight Customer: find most favourite item of each customer
#          [3] Data Scaling: scaling data before run kmeans
#          [4] Choose K: Elbow and Silhouette method is applied (Option Step)
#          [5] Kmean Cluster
#          [6] Labeling
# Usage:
#    Input: - df: dataframe [customer_id, productA, B, C, D, ....], value is quantity of purchasing
#           - group: rfm_group name
#           - kcluster: number of clusters
#    Output: two dataframe: df_clustered, df_group_cluster_info
#

def df_group_cluster(df,group,kcluster):

  debug_n = debug
  custom_kn = custom_k
  #this DataFrame using for create summary table for each cluster by product quantity
  df_group_cluster_info = pd.DataFrame()
  
  ###################################################
  # DATAFRAME INFO AND PREPROCESSING
  #
  #  

  groupx=re.sub(" ","_",group)
  
  df.reset_index(inplace=True)
  df.drop(columns='index',inplace=True)
  
  print_debug("***************************************************")
  print_debug("[INFO] Information of Customer {} will be clustered:".format(group))
  print_debug("Shape  : {}".format(df.shape))
  print_debug("Columns: {}".format(df.columns.values))
  print_debug("df=\n{}".format(df[:10]))
  print_debug("***************************************************")

  ###################################################
  # [1] DROP LINE WITH ALL ZEROS
  # check if there are any row with all zero value
  col = df.columns.values
  df_data = df[col[2:]]
  rs = df_data[df_data.eq(0).all(1)]

  print_debug("rs = {}".format(rs))
  if len(rs) == 0:
    print_debug ("[INFO] Number line of all features get value of zero is {}".format(len(rs)))
    df_rest = df.copy()
    df_drop = pd.DataFrame(columns=df.columns.values)
    print_debug ("SUCCESSFUL!!!")
  else:
    print_debug ("[INFO] Number line of all features get value of zero is {}".format(len(rs)))
    print_debug ("[INFO] PROCESS to DROP All Zero value line")    

    # drop line having zero values
    df_data['sum']=df_data.sum(axis=1)
    drop_id = df[df_data['sum']==0].index.values
    
    print_debug ("[INFO] drop_id list is: \n{}".format(drop_id))
    print_debug ("[INFO] df.index[drop_id] = \n{}".format(df.index[drop_id]))
    df_drop = df.iloc[drop_id,:]
    
    print_debug ("[INFO] Length of drop_id: \n{}".format(len(drop_id)))
    print_debug ("[INFO] Length of df_drop: \n{}".format(len(df_drop)))
    print_debug ("[INFO] df_drop.head(): \n{}".format(df_drop[:5]))
    df_drop['sum']=df_data[1:].sum(axis=1)
    
    df_rest = df.drop(drop_id)
    print_debug ("[INFO] Length of df_rest: \n{}".format(len(df_rest)))
    print_debug ("[INFO] df_rest: \n{}".format(df_rest[:5]))
    
    df_rest.reset_index(inplace=True)
    df_rest.drop(['index'],axis=1, inplace=True)
    if len(df_rest) == 0:
      print_debug("[WARNING] There is no customer in df_rest.")
    else:
      print_debug("[INFO] Number of valid customer, having product quantity is {}".format(len(df_rest)))
	  
  print_debug ("[INFO] Total row number of dataframe {}".format(len(df)))


  #kcluster estimation
  #kcluster = num_product/4
  DIV = 4
  if not custom_kn:
    kcluster = int((len(df_rest.columns.values)-2)/DIV)
  #process for case kcluster > num of sample in groupi
  if len(df_rest) < 4*kcluster:
    kcluster = int(len(df_rest)/DIV) #kcluster=num_cutomer/4
  if (len(df_rest) == 1) | ((len(df_rest) > 1) & (kcluster == 0)):
    kcluster = 1
  print_debug('*** After K estimate: Group: {}, kcluster={} for total customer is {} with {} types of product'.format(group, kcluster,len(df_rest),len(df_rest.columns.values)-2))

  ###################################################
  # [2] FIGURE OUT MOST FAVOURITE ITEMS OF EACH CUSTOMER
  #    Arange by number of items folowing order.
  #    Choose the most favourite item to DataFrame
  # OUTPUT: df_cus_favourite
  #
  
  df_x = df_rest.copy()
  
  df_x.drop(['rfm_group'],axis=1, inplace=True)
  
  #--------------------------------------
  # Call Function
  # Arrange customer product purchase by most Favourite
  num_top_favourite = 7

  indicators = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'nineth', 'tenth']

  # create columns according to number of top favourite
  columns = ['cus_id']

  for ind in np.arange(num_top_favourite):
    columns.append('{}_favourite'.format(indicators[ind]))

  # create a new dataframe
  df_cus_favourite = pd.DataFrame(columns=columns)
  df_cus_favourite['cus_id'] = df_x['cus_id']

  df_cus_favourite_quan = pd.DataFrame(columns=columns)
  df_cus_favourite_quan['cus_id'] = df_x['cus_id']

  print_debug ("[INFO] Clustering for {} is processing...\nit could take few minutes to complete......".format(group))
  for ind in np.arange(df_x.shape[0]):
    df_cus_favourite.iloc[ind, 1:] = return_most_favourite(df_x.iloc[ind, :], num_top_favourite)[0]
    df_cus_favourite_quan.iloc[ind, 1:] = return_most_favourite(df_x.iloc[ind, :], num_top_favourite)[1] # return_most_favourite_quan(df_x.iloc[ind, :], num_top_favourite)

  list_fav_colname = df_cus_favourite.columns.values[1:]
  print_debug ("[INFO] list_fav_colname = \n{}".format(list_fav_colname))
  print_debug ("[INFO] df_cus_favourite_quan = \n{}".format(df_cus_favourite_quan[:5]))
  print_debug ("[INFO] df_cus_favourite = \n{}".format(df_cus_favourite[:5]))

  ###################################################
  # [3] DATA PROCESSING: SCALING 
  #

  # list_features: all input feature for cluster
  list_features = df_rest.columns.values[2:]
  print("list_features = {}".format(list_features))
  df_features = df_rest[list_features]

  df_features_scale = pd.DataFrame(columns=df_features.columns.values)
  for i, product in zip(range(len(list_features)),list_features):
    df_features_scale[product]=df_features[product]/df_features.sum(axis=1)

  print_debug ("Length of df_feature_scale is {}".format(len(df_features_scale)))
    
  '''
  ###################################################
  # [4] RUN DIFFERENT K TO FIND SUITABLE K value
  # Elbow and Silhouette method is applied
  #

  # CHOOSING K: Run Clustering Evaluation, consider SSE and Silhouette score.
  sse = {} #sse: sum of squared error
  silhouette_kclus = []

  for k in range(2, 25):
    # Kmeans Model
    kmeans = KMeans(n_clusters = k, max_iter=1000).fit(df_features_scale)
    
    # add cluster result following k to top venues df
    #cluster_x="cluster_"+str(k)    
    #df_hcmc_top_venues[cluster_x] = kmeans.labels_   
    
    #df_clustering["clusters_"+k] = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    print("For n_clusters={}: SSE is {}".format(k, sse[k]))
    
    if (k>1):
      # For Silhouette score
      label = kmeans.labels_
      sil_coeff = silhouette_score(df_features_scale, label, metric='euclidean')  
      silhouette_kclus.append(sil_coeff)
      print("For n_clusters={}: The Silhouette Coefficient is {}".format(k, sil_coeff))

   #df_hcmc_top_venues.head()

  #--------------------------------------
  #plot for SSE 
  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)
  plt.plot(list(sse.keys()), list(sse.values()), marker='^')
  plt.plot(4,list(sse.values())[2], marker='o', color='w', markersize=12, markeredgewidth=4, markeredgecolor='r')
  plt.xlabel("Number of cluster (K)")
  plt.ylabel("Sum of Squared Errors (SSE)")
  plt.title("K-Means Cluster Evaluation by Elbow Method")
  chart.savefig(debug_path+"/Elbow_KMean_SSE_by_k.png")
  # by the Elbow method as in below figure, the value of 3 is chosen

  #--------------------------------------
  #plot for Silhouette score
  index_of_max = silhouette_kclus.index(max(silhouette_kclus[1:]))

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)
  plt.plot(range(2,len(silhouette_kclus)+2),silhouette_kclus, marker='^', color='b')
  plt.plot(index_of_max+2,silhouette_kclus[index_of_max], marker='o', color='w', markersize=12, markeredgewidth=4, markeredgecolor='r')

  plt.xlabel("Number of cluster (K)")
  plt.ylabel("Silhouette Score")
  plt.title("K-Means Cluster Evaluation by Silhouette Analysis")
  chart.savefig(debug_path+"/Silhoutte_by_k.png")
  
  system.exit(1)  
  '''
  
  ###################################################
  # [5] Kmean Cluster
  # THROUGH Elbow method and Silhouette value, choose suitable k value
  # Here chooses k=4
  # 

  # run Kmean again with k = 4
  # set number of clusters
  kclusters = kcluster

  # run k-means clustering
  kmeans = KMeans(n_clusters = kclusters, max_iter=1000, random_state=0).fit(df_features_scale)
  
  #save Kmean Model
  model_saving_path=model_path+"/rfm_group_"+str(group)+"_kmeans.sav"
  pickle.dump(kmeans,open(model_saving_path,'wb'))
  print_debug("[SAVE MODEL] saved model at {}".format(model_saving_path))
  
  # check cluster labels generated for each row in the dataframe
  #kmeans.labels_[0:10]

  #-----------------------------------
  # add clustering labels
  df_cus_favourite2 = df_cus_favourite.copy()
  df_cus_favourite2["Cluster Labels"] = kmeans.labels_


  ###################################################
  # [6] LABELING 
  # Join cluster label to customer and labeling 
  # 

  # join
  df_merge2 = pd.merge(df_cus_favourite2,df_x, how='left', on=['cus_id'])
  print("df_merge2.sum = {}".format(df_merge2.isnull().sum().values.sum()))
  if (df_merge2.isnull().sum().values.sum() > 0):
    sys.exit(1)
  #return df_merge2
  # export csv file
  outfile = output_path+"/"+version+"_cus_cluster_by_product_"+str(groupx)+"_"+str(kclusters)+".csv"
  print_debug ("\n[DONE] Result of Cluster is at {}".format(outfile))

  print_debug ('####################################')
  print_debug ('# KMEAN CLUSTER REPORT: Group: {}, kcluster {}'.format(group,kcluster))
  print_debug ('####################################')
  
  #print lenght of each cluster
  for i in range(kclusters):
    print_debug ("[INFO] length of cluster {} is {}".format(i,len(df_cus_favourite2[df_cus_favourite2['Cluster Labels']==i])))

  #summarize the quantity of product sold by each cluster and pie chart plot
  for i in range(kclusters):
    df_cluster_counti = pd.DataFrame(columns=['count'],data=df_merge2[df_merge2['Cluster Labels']==i][list_features].sum(axis=0).sort_values(ascending=True))

    print("df_cluster_counti.nan.sum = {}".format(df_cluster_counti.isnull().sum().values.sum()))
    print("group = {}, i = {}\ndf_cluster_counti = \n{}".format(group, i, df_cluster_counti))
    #sys.exit(1)
    if (df_cluster_counti.isnull().sum().values.sum() > 0):
      sys.exit(1)
	
    df_temp = create_group_label_product_quan_table(df_cluster_counti,group,i)
    
    df_group_cluster_info = pd.concat([df_group_cluster_info,df_temp])
    
    xlabel = 'Categories'
    ylabel = "Cluster "+str(i)
    title = 'Percentage most purchase category'
    fname = debug_path+"/"+str(kcluster)+"_"+groupx+"_"+title+"_"+ylabel+"_"+xlabel+".png"
    print_debug("[INFO] file name is {}".format(fname))
    
    if debug_n:
      plot_pie(df_cluster_counti,'Categories',"Cluster "+str(i),'Percentage most purchase category',file_name=fname)
    
  df_group_cluster_label = df_group_cluster_info[['rfm_group','Cluster Labels','cluster_type','cluster_labeling']]
  df_group_cluster_label.drop_duplicates(subset=['rfm_group','Cluster Labels','cluster_type','cluster_labeling'], keep='first',inplace=True)

  # add two labeling columns to df_merge2
  df_merge2['cluster_type']=np.zeros(len(df_merge2))
  df_merge2['cluster_labeling']=np.zeros(len(df_merge2))

  for j in range(len(df_merge2)):
    df_merge2.loc[j,'cluster_type']=df_group_cluster_label[df_group_cluster_label['Cluster Labels']==df_merge2['Cluster Labels'][j]]['cluster_type'][0]
    df_merge2.loc[j,'cluster_labeling']=df_group_cluster_label[df_group_cluster_label['Cluster Labels']==df_merge2['Cluster Labels'][j]]['cluster_labeling'][0]
  
  # export csv file
  outfile = output_path+"/"+version+"_cus_cluster_by_product_"+str(groupx)+"_"+str(kclusters)+".csv"
  print ("\n[DONE] Result of Cluster is at {}".format(outfile))
  
  df_merge3=df_merge2.copy()  
  
  df_merge3['rfm_group']=df_rest['rfm_group']
  colum = list(df_merge3.columns[-3:]) + list(df_merge3.columns[:-3])
  df_merge3=df_merge3[colum]
  
  #process drop cluster

  drop_column = df_merge3.columns.values
  df_drop_cluster = pd.DataFrame(columns=drop_column)
  df_drop_cluster.loc[:,'rfm_group']=df_drop['rfm_group']
  df_drop_cluster.loc[:,'Cluster Labels']=(np.zeros(len(df_drop),dtype=int)+1)*kcluster
  df_drop_cluster.loc[:,'cus_id']=df_drop['cus_id']
  df_drop_cluster.loc[:,list_features]=df_drop[list_features]
  df_drop_cluster.loc[:,list_fav_colname]=np.zeros((len(df_drop),len(list_fav_colname)))
  df_drop_cluster.loc[:,'cluster_type']=['refund']*len(df_drop)
  df_drop_cluster.loc[:,'cluster_labeling']=['refund']*len(df_drop)

  df_clustered = pd.concat([df_merge3,df_drop_cluster])
  
  df_clustered.to_csv(outfile)
  # DONE: This file is sent to DW.
  final_column1 = df_clustered.columns.values
  final_column2 = df_group_cluster_info.columns.values
  print ('##################################')
  print ('# COMPLETED: {} k={}'.format(groupx,kcluster))
  print ('##################################')
  print (df_group_cluster_info.shape)
  print (df_group_cluster_info)
  
  return df_clustered,df_group_cluster_info
  
### END
############################################################

##################################################################################################
##################################################################################################
##################################################################################################
# MAIN
#

if not commitonly:
  output_df = pd.DataFrame()
  output_gc_df = pd.DataFrame()

  for groupi in list(df_pivot['rfm_group'].unique()):
    print_debug('Cluster for {}'.format(groupi))
    df_groupi = df_pivot[df_pivot['rfm_group']==groupi]
  
    #process for case kcluster > num of sample in groupi
    kcluster_adj = kcluster
    df_groupi_cluster,df_groupi_cluster_info = df_group_cluster(df_groupi,groupi,kcluster_adj)
  
    output_df = pd.concat([output_df, df_groupi_cluster])
    output_gc_df = pd.concat([output_gc_df,df_groupi_cluster_info])

  print (">> COMPLETED CLUSTERING ...")
  #---------------------------------

  output_short_df = output_df.iloc[:,0:12]

  outfinal = output_path+"/"+version+"_cus_cluster_by_product.csv"
  output_df.to_csv(outfinal, index=False)

  outfinal = output_path+"/"+version+"_cus_cluster_by_product_short.csv"
  output_short_df.to_csv(outfinal, index=False)

  outfinal = output_path+"/"+version+"_group_cluster_info.csv"
  output_gc_df.to_csv(outfinal, index=False)
  #----------------------------------

  cluster_summary_df = output_short_df[['rfm_group','Cluster Labels','cluster_type','cluster_labeling']]
  cluster_summary_df.drop_duplicates(subset=['rfm_group','Cluster Labels','cluster_type','cluster_labeling'], keep='first',inplace=True)


  #compute number customers for each cluster
  cluster_summary_df.loc[:,'total_cus']=np.zeros(len(cluster_summary_df),dtype=int)

  for i in list(cluster_summary_df['rfm_group'].unique()):
    for j in list(cluster_summary_df['Cluster Labels'].unique()):
      total_cus = output_short_df[(output_short_df['rfm_group']==i) & (output_short_df['Cluster Labels']==j)]['cus_id'].count().astype(int)      
      cluster_summary_df.loc[(cluster_summary_df['rfm_group']==i) & (cluster_summary_df['Cluster Labels']==j),'total_cus']=total_cus
    
  outfinal = output_path+"/"+version+"_group_cluster_summary.csv"
  cluster_summary_df.to_csv(outfinal, index=False)

  print_debug ('[INFO] Cluster Summary:\n{}'.format(cluster_summary_df))
  print_debug ("[CLUSTER RESULT] Please check at \n{}\n{}\n{}".format(output_path+"/"+version+"_cus_cluster_by_product.csv",output_path+"/"+version+"_cus_cluster_by_product_short.csv",output_path+"/"+version+"_group_cluster_info.csv"))

###################################################################
# Commit to sandbox
#

if (commit or commitonly):
  print ('##################################')
  print ('# COMMIT RESULT TO SANDBOX')
  print ('##################################')

  commit_filelist = [output_path+"/"+version+"_cus_cluster_by_product_short.csv",output_path+"/"+version+"_group_cluster_info.csv",output_path+"/"+version+"_group_cluster_summary.csv"]
  commit_sandbox_name = [input+"_SEGMENT_LABEL",input+"_SEGMENT_PRODUCT_INFO",input+"_SEGMENT_SUMMARY"]
  
  db = None
  if debug:
    db = '--debug'
  for i in range(len(commit_filelist)):
    os.system('python commit_bq.py --source {} --destination {}'.format(commit_filelist[i],commit_sandbox_name[i]))      

print ('##################################')
print ('# ENDING ! CONGRATULATION')
print ('##################################')