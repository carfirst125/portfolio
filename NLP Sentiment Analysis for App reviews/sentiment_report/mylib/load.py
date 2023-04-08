import json
import pandas as pd
from tqdm import tqdm
from string import punctuation
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby
from numpy import *
import regex as re
from google_play_scraper import Sort, reviews, app, reviews_all
from deep_translator import GoogleTranslator
from app_store_scraper import AppStore
import global_materials

# validate short typing word by custom dictionary.
def transform(x):
    x = str(x) + ' '
    x = x.lower()
    for key in global_materials.custom_dict.keys():
        x = x.replace(key, global_materials.custom_dict[key])
    return x

# remove punctuation out of string
def removeDupPunc(strg):
    newText = []
    punc = set(punctuation) - set('.') # remove '.' out of punc
    for k, g in groupby(strg):
        if k in punc:
            newText.append(k)
        else:
            newText.extend(g)
    return(''.join(newText))

reg = re.compile(r"[^\w\d\\(.,;!?:\\-\\') ]+", re.UNICODE)
translator = GoogleTranslator(source = 'vi', target = 'en')

# translate vietnamese to english
def Englishtranslate(x):
    x = transform(x)
    x = reg.sub('Meaningless words', x)
    x = re.sub('Meaningless words','',x)
    x = removeDupPunc(x)
    if x is None:
        return ''
    elif x == '': 
        return x 
    elif len(x) > 1:
        print(x)
        x = translator.translate(x)
        return x 
    # x = re.sub('Meaningless words','',str(x))
    # if x is None:
    #     return ''
    # else:
    #     return x 