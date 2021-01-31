##############################################################################
# Function: Train Data Preparation
#-----------------------------------------------------------------------------
# File name:   train_data_preparation.py 
# Author:      Nhan Thanh Ngo
#############################################################################

import json
import pandas as pd
import numpy as np
from io import StringIO 
import boto3
import time
from time import sleep
from datetime import datetime

import datetime
import sys
import os
import re
import io
import argparse
import yaml
import glob

from my_lib import query_bq
from my_lib import data_eda

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="DW.input_table", type=str)
parser.add_argument("--query", default=False, type=bool)
args = parser.parse_args()

################################################################
s3upload_path = './s3upload'
temp_path = './temp'
extern_path = './extern'   # store features of items 

def folder_structure():
    global s3upload_path, temp_path, extern_path
	
    if not os.path.exists(s3upload_path):
        os.mkdir(s3upload_path)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    if not os.path.exists(extern_path):
        os.mkdir(extern_path)
		

def read_parameter(parameter_file):
    parameter_dict = {}
    with open(r'./'+parameter_file) as file:
      parameter_dict = yaml.load(file, Loader=yaml.FullLoader)
      
    for key in parameter_dict:
      if key == 'OBSERVED_TIME_PERIOD':
        OBSERVED_TIME_PERIOD = parameter_dict[key]
    return OBSERVED_TIME_PERIOD


def df_truncate(df, PERIOD):
    df['date_order'] = pd.to_datetime(df['date_order'])
    trunc_date = df.date_order.max() - datetime.timedelta(days=PERIOD)
    return df[df.date_order >= trunc_date]

def interacts_metadata(df):
    df['timestamp'] = df['date_order'].apply(lambda x: datetime.datetime.timestamp(x)) 
    df = df[['customer_id','product_id','timestamp','quantity']]
    df = df.groupby(['customer_id','product_id','timestamp'])['quantity'].sum().reset_index()
    df.columns = ['USER_ID', 'ITEM_ID','TIMESTAMP','QUANTITY']
    df.TIMESTAMP = df.TIMESTAMP.astype(int)
    df.QUANTITY = df.QUANTITY.astype(int)
    df = df.sort_values('TIMESTAMP', ascending = True)
	
    df.to_csv(s3upload_path+'/interacts_metadata.csv',index=False)

def get_item_info(df):
    # df is df_eda.csv 
    df = df[['items','size','product_id']]
    df['product_id'] = df['product_id'].astype(int)
    df['size'] = df['size'].astype(str)
    df.loc[df['size']=='0','size'] = 'B' #Banh + others (NO SIZE)
  
    df['rm'] = 1
    df = df.groupby(['items','size','product_id'])['rm'].sum().reset_index()
    df.drop('rm',axis=1,inplace=True)
    df.columns = ['ITEM_NAME','ITEM_SIZE','ITEM_ID']
	
    df.to_csv(s3upload_path+'/items_info.csv', index=False)
    
def items_metadata():

    df_items = pd.read_csv(extern_path+'/item_features.csv', index_col=None)
    items_col = df_items.columns.values[1:]
	
    ITEM_FEA_arr = []
    for i in np.arange(df_items.shape[0]):
      arr1 = df_items.iloc[i,1:].values
      rs = '|'.join([items_col[ind] for ind, val in enumerate(arr1) if val]) 
      ITEM_FEA_arr.append(rs)
	  
    df_items['ITEM_FEA'] = ITEM_FEA_arr

    df_items.rename(columns = {'items':'ITEM_NAME'}, inplace = True)
    df_items = df_items[['ITEM_NAME','ITEM_FEA']]

    df_item_info = pd.read_csv(s3upload_path+'/items_info.csv',index_col=None)
    df_items = pd.merge(df_items,df_item_info,on='ITEM_NAME',how='right')
    df_items = df_items[['ITEM_ID','ITEM_SIZE','ITEM_FEA']]
	
    df_items.to_csv(s3upload_path+'/items_metadata.csv',index=False)

def main(args):

    global s3upload_path, temp_path, extern_path
	
    OBSERVED_TIME_PERIOD = read_parameter('parameters.yml')
    print('Get Parameter... completed')

    folder_structure()    
    print('Build folder structure... completed')
	
    # query data
    df = query_bq(args.input, args.query)
    print('Query data from bigquery... completed')  
	
    # eda
    df = data_eda(df)    
    df.to_csv(temp_path+'/df_eda.csv',index=False)

    print('data EDA... completed') 
	
    # gen item info file
    get_item_info(df)
    print('get item information... completed')
    
    # truncate data (only get data in nearest 6 months for eg)
    df = df_truncate(df, OBSERVED_TIME_PERIOD)
    df.to_csv(temp_path+'/df_eda_truncate.csv',index=False)	
    print('truncate data... completed')
	
    # interact metadata
    interacts_metadata(df)
    print('transform interacts metadata... completed')
    
    # item metadata
    items_metadata()
    print('transform items metadata... completed')
	
if __name__ == "__main__":  
    main(args)
   



