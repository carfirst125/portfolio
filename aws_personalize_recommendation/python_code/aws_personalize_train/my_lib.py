##############################################################################
# MY LIBRARY
#-----------------------------------------------------------------------------
# File name:   my_lib.py 
# Author:      Nhan Thanh Ngo

#############################################################################

import pandas as pd
import numpy as np
import json
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
import glob

def query_bq(input, query, limit=''):

  print ('#############################')
  print ('# QUERY DATA FROM BIGQUERY')
  print ('#############################')

  from google.cloud import bigquery
  from google.oauth2 import service_account

  client = bigquery.Client()

  sql = "SELECT * FROM "+input+limit
  bq_cus_purchase = input.split('.')[-1]+".csv"

  ###################################3
  # QUERY
  ###################################
  if query:
    # Run a Standard SQL query with the project set explicitly
    print("Querying data from sandbox...\nit will take a few minutes...")

    df = client.query(sql).to_dataframe() 
    
    if not glob.glob(bq_cus_purchase):
      print ("[INFO] No BigQuery datafile available")
    else:
      print("[INFO] Remove exist bq datafile")
      os.remove(bq_cus_purchase)
    
    print("[INFO] Store query data from Big Query to file")
    df.to_csv(bq_cus_purchase,index=False)
  else:
    print("[INFO] Read input data from offline file, need updateall please run again with -q to query new data from Big Query")
    print("Read offline input data file...\nit will take a few minutes...")
    df = pd.read_csv(bq_cus_purchase, index_col=None)
  return df


def data_eda(df):

    df = df[['customer_id','date_order','product_id','items','size','quantity']]  
    ############################################################
    # process item_name which does not match with standard
    ############################################################
       
    main_items_list = []

    for item_name in df['items'].unique():
      #print('Process: item_name: {}'.format(item_name))      

      matched = bool(re.match("(^EXTRA$)",item_name)) | \
                bool(re.match("(^COMBO$)",item_name)) | \
                bool(re.match("(^PHIN COFFEE$)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(200G)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(185ML)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(1KG)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(STICK)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(CHAI)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(GOI)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(HOP)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(LON)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(FREE PRODUCT)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("([\w\.\, \-\(\)]*)(DISCOUNT)([\w\.\, \-\(\)$]*)",item_name)) | \
                bool(re.match("(^NUOC TINH KHIET$)",item_name)) 
                
      if matched:
        print('[REMOVE] item_name: {}'.format(item_name))
        continue
      else:
        print('[KEEP]  main items: {}'.format(item_name))
        main_items_list.append(item_name) 
        
    # remove items which is not in the main_items_list
    df = df[df['items'].isin(main_items_list)]
	
    # store features to file
    print('The number of main items is {}'.format(len(main_items_list)))
    main_items_list_str = '\n'.join(main_items_list)    
    with open('./temp/main_items_list.txt', 'w') as logfile:
      logfile.write(main_items_list_str)

    return df        