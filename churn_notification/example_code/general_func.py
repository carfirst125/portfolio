########################################################################################################################
# GENERAL FUNCTION
# File name : general_func.py
# Author    : Nhan Thanh Ngo

import numpy as np
import pandas as pd

import warnings
import re
import datetime
import random
import pickle
import yaml
import math
import os
import sys
import shutil
import yaml

def eda_data(df):

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

    return df        


