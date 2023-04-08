

import regex as re 
import pandas as pd
from collections import Counter
import numpy as np
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
import sys
import pickle

import textacy.preprocessing.normalize as tprep
from textacy.preprocessing.remove import accents

import regex as re
from deep_translator import GoogleTranslator

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

def normalize(text):
    text =str(text)
    # text = re.sub('vib', 'VIB', text)
    text = tprep.hyphenated_words(text)
    text = tprep.quotation_marks(text)
    text = tprep.unicode(text)
    text = accents(text)
    return text

from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
import string
from nltk import pos_tag
import nltk

import sys
sys.path.append('mylib')
import global_materials

import enchant
dict_english = enchant.Dict("en_US")

###################################################
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('opinion_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# #stopwords is not the primary words without effecting sentiment of sentence --> processing remove stopword
# stopwords = set(stopwords.words('english')) # nltk data
# include_stopwords = {'dear', 'regards', 'must', 'would', 'also', 'app', 'application'}
# exclude_stopwords = {'against', "don't", "didn't", "doesn't", "can't", "wouldn't"}
# stopwords |= include_stopwords
# stopwords -= exclude_stopwords
##################################################

from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = str(text)
    #print('[1] {}',text)
    
    # lower text
    text = text.lower()
    #print('[2] {}',text)
    
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    #print('[3] {}',text)
    
    # remove words that contain numbers
    text = [word for word in text if not any([c.isdigit() for c in word])]
    #print('[4] {}',text)
    
    # remove stop words
    # stop = stopwords.words('english')
    text = [x for x in text if x not in global_materials.stopwords]
    #print('[5] {}',text)

    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    #print('[6] {}',text)
    
    # pos tagtext
    pos_tags = pos_tag(text)
    #print('[7] {}',text)
    
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    #print('[8] {}',text)
        
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    #print('[9] {}',text)
    
    # join all
    text = " ".join(text)
    print('[10] {}', text)
    
    return text


####################################################################

pos_score = 1
neg_score = -1
word_dict = {}

#dirty_words = ['qq','cc','lol','dm', 'vl', 'cock', 'l', 'c', 'shit', 'cobs']
global_materials.dirty_words.extend([k for k, v in global_materials.custom_dict.items() if v == '***'])

for word in opinion_lexicon.positive():
    word_dict[word] = pos_score
for word in opinion_lexicon.negative():
    word_dict[word] = neg_score
for word in global_materials.dirty_words:
    word_dict[word] = 2*neg_score  
    
def bing_liu_score(text):
    sentiment_score = 0
    print(text)
    #sys.exit()
    bag_of_words = word_tokenize(text.lower())
    for word in bag_of_words:
        if word in word_dict:
            sentiment_score += word_dict[word]
    return sentiment_score / len(bag_of_words)

#########################################################################

def review_vietnamese_adj (sentence):

    '''Description: 
          correct typo mistake by looking up dictionary. The manual built dictionary includes couples of key and corrected words.'''

    # remove punctuation out of sentence
    sentence=re.sub('[,.-]',' ', sentence)
    
    sentence_adj = []
    split_word = sentence.split()

    # --> process single word
    for j in split_word:
        #print(j)
        if j in global_materials.custom_dict.keys():            
            sentence_adj.append(global_materials.custom_dict[j])
        else:
            sentence_adj.append(j)
    #print(sentence_adj)
    
    # --> process couple of words
    ignore_flag=False
    sentence_adj2 = []
    for j in range(len(sentence_adj)):
        #print(sentence_adj[j])
        if ignore_flag:
            ignore_flag = False
            continue
        
        if j==(len(sentence_adj)-1):
            if not ignore_flag:
                sentence_adj2.append(sentence_adj[j])
                break
        if (sentence_adj[j]+' '+sentence_adj[j+1]) in global_materials.custom_dict.keys():                  
            sentence_adj2.append(global_materials.custom_dict[sentence_adj[j]+' '+sentence_adj[j+1]])            
            ignore_flag = True
        else:
            sentence_adj2.append(sentence_adj[j])
    
    # get sentence and process punctualtion    
    sentence = ' '.join(sentence_adj2)
    text = [word.strip(string.punctuation) for word in sentence.split(" ")]
    text = [word for word in text if not any([c.isdigit() for c in word])]
    text = [t for t in text if len(t) > 0]
    
    sentence = ' '.join(text)    
    print(sentence)
    
    return sentence

##############################################################################

def review_vietnamese_match_negative_word (sentence):
    
    '''Description: 
          explore in sentence and count the number of words appearing in the negative array.
          The negrative array includes words which express negative sentiment of customer.'''

    # remove punctuation out of sentence
    sentence=re.sub('[,.-]',' ', sentence)
    split_word = [word.strip(string.punctuation) for word in sentence.split(" ")]
    split_word = [word for word in split_word if not any([c.isdigit() for c in word])]
    split_word = [t for t in split_word if len(t) > 0]

    neg_count=0
    #split_word = sentence.split()

    # process single word
    for j in split_word:
        if j in global_materials.negative_word:        
            neg_count+=1
            #print(j)

    # process couple of words
    for w in range(len(split_word)):
        if w==(len(split_word)-1):
            break
        else:
            if (split_word[w]+" "+split_word[w+1]) in global_materials.negative_word:    
                neg_count+=1
                #print(split_word[w]+" "+split_word[w+1])
     
    return neg_count

##############################################################################

# CLEAN ENGLISH TEXT FOR SENTIMENT ANALYSIS
def review_en_clean(df):
    #df['review_normalize'] = df['review_en'].map(normalize)          # text normalization
    #print(df['review_normalize'])
    print(df['review_en'])
    df['review_clean']     = df['review_en'].map(clean_text)  # clean text 
    print(df['review_clean'])
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('can\'t ',' can not ', x)) 
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('don\'t ',' do not ', x)) 
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('doesn\'t ',' does not ', x))
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('didn\'t ',' did not ', x)) 
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('\'m',' am ', x))
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('\'s','', x))
    df['review_clean']     = df['review_clean'].apply(lambda x: re.sub('\"','', x))
    print(df['review_clean'])
    df['review_clean']     = df['review_clean'].apply(lambda x: ' '.join([word for word in x.split() if dict_english.check(word)])) # remove NOT english words
    IGNORE_LIST = ['oh','ah','uh','uhm','ooh','cc']
    print(df['review_clean'])
    df['review_clean']     = df['review_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in IGNORE_LIST]))
    df['review_empty']     = np.where(df['review_clean'].str.len() != 0, 0, 1)
    df                     = df[df['review_empty']==0]
    df                     = df[df['username'] != 'Người dùng Google']
    return df
###########################################################################################################

from deep_translator import GoogleTranslator

def google_translate(df, temp_dir, row_p_time=10):

    today = date.today()
        
    translator = GoogleTranslator(source = 'vi', target = 'en')
    df_rs = pd.DataFrame()
    
    try:

        for i in np.arange(0, df.shape[0]//row_p_time+1, 1):
            end = row_p_time*(i+1) if ((row_p_time*(i+1) < df.shape[0])) else df.shape[0]
            df_run = pd.DataFrame(df.iloc[row_p_time*i:end])

            df_run['review_en'] = df_run['review_adj'].apply(lambda x: translator.translate(x))
            df_run.to_csv(temp_dir+'/df_run_{}_{}.csv'.format(today.strftime("%d_%m_%Y"),i),index=False, sep='\t')
            df_rs = pd.concat([df_rs,df_run])
            
            # print("i={} -- {}:{} -- df_rs length = {}".format(i,row_p_time*i,end,df_rs.shape[0]))
            # print(df_run['review_en'])
            #break
        
        print(f'Content: out length:{df_rs.shape[0]},in length: {df.shape[0]}')

        if (df_rs.shape[0]==df.shape[0]):
            return df_rs, True
            #shutil.rmtree('./temp')
        else:
            return df_rs, False        

    except:

        #df_rs.to_csv('./content_translation.csv',index=False,sep='/')
        return df_rs, False
        
###########################################################################################################


def sentiment_prediction(df, tfidf_convertor, sentiment_model, pred_out_file):
    ''' Description: this function has input data in Vietnamese
             Then it will: - process Vietnamese (correct typo error)
                           - translate to English (google translate)
                           - clean English text
                           - extract Feature for model (TF-IDF + VADER)
                           - Load sentiment model and run prediction
        Input: df: vietnamese_df ['username','review','rating']    
               tfidf_convertor: Saved TF-IDF (dump by the train process)
               sentiment_model: saved Sentiment Model (trained model in train step)
               pred_out_file:   file of predicted result      
    '''
    
    # [1] Vietnamese Process
    df = df[~(df['review'].isnull()|df['review'].isna())]  
    df['review_adj'] = df['review'].apply(lambda x: review_vietnamese_adj(x.lower())) # correct vietnamese
    print("There are {} new reviews".format(df.shape[0]))


    # [2] English translation
    df['review_en'] = df['review_adj'].apply(lambda x: GoogleTranslator(source = 'vi', target = 'en').translate(x))
    df['username'] = np.zeros(df.shape[0])
    df = review_en_clean(df)

    # [3] Extract Feature for model prediction
    # [3.1] Compute VADER score
    df['vader_compound'] = df['review_clean'].apply(lambda x: sentiment.polarity_scores(x)['compound'])
    #df_pred['vader_compound'] = df_pred['vader_compound'].apply(lambda x: 0 if x<-0.05 else 1 if x>0.05 else 0.5 )
    df['vader_compound'] = df['vader_compound'].apply(lambda x: 0 if x<0.05 else 1)

    # [3.2] Compute TFIDF value
    tfidf_vectorizer = pickle.load(open(tfidf_convertor,'rb'))
    col_list = [i.lstrip() for i in tfidf_vectorizer.get_feature_names_out()]
    col_list = np.array(col_list)
    df_tfidfx = pd.DataFrame(data=tfidf_vectorizer.transform(df['review_clean'].values).toarray(), columns=col_list)    
    df_tfidfx['vader_compound'] = df['vader_compound'].values    
    
    # [4] load SENTIMENT MODEL and predict
    sentiment_MODEL = pickle.load(open(sentiment_model,'rb'))
    y_pred = sentiment_MODEL.predict(df_tfidfx[list(col_list)+['vader_compound']].values)
    
    df['sentiment'] = y_pred   
    #df[['review_adj','review_en','review_clean','vader_compound','sentiment']].to_csv(pred_out_file, index=False, sep='\t')
    df.to_csv(pred_out_file, index=False, sep='\t')
    return df #[['review_adj','review_en','review_clean','vader_compound','sentiment']]

##################################################################################################################
from datetime import date
# Android
from google_play_scraper import Sort, reviews, app, reviews_all
# iOS
from app_store_scraper import AppStore

# Android
def get_rwdata_android(rw_android_src_dict, bank_name):
    today = date.today()

    Andreviews = reviews_all(
        rw_android_src_dict[bank_name][0],
        #'vn.com.techcombank.bb.app',
        sleep_milliseconds=0, # defaults to 0
        lang=rw_android_src_dict[bank_name][1], # defaults to 'en'
        country=rw_android_src_dict[bank_name][2], # defaults to 'us'
        sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
    )
         
    Andreviews_df = pd.DataFrame(Andreviews)
    Andreviews_df.to_csv('AndRaw_'+bank_name+'_'+str(today)+'.csv', index=False)
    print(f'{bank_name}: {Andreviews_df.shape}')
    #print(Andreviews_df.columns.values)
    return Andreviews_df

# iOS
def get_rwdata_ios(rw_ios_src_dict, bankapp):
    
    appStore = AppStore(country=rw_ios_src_dict[bankapp][1], app_name=rw_ios_src_dict[bankapp][0], app_id=int(rw_ios_src_dict[bankapp][2])) #Techcombank Mobile
    #appStore = f'AppStore(country=\'{rw_ios_src_dict[bank_name][1]}\', app_name=\'{rw_ios_src_dict[bank_name][0]}\', app_id=\'{int(rw_ios_src_dict[bank_name][2])}\')'
    print(f'{bankapp}: AppStore(country=\'{rw_ios_src_dict[bankapp][1]}\', app_name=\'{rw_ios_src_dict[bankapp][0]}\', app_id=\'{int(rw_ios_src_dict[bankapp][2])}\')')
    #time.sleep(3)
    appStore.review()    
    
    df = pd.DataFrame(np.array(appStore.reviews),columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    print(f'{bankapp}: {df.shape}')
    return df