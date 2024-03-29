## Sentiment Analysis for Mobile App Reviews 

This demo shows you contents as following:
  * How to make sentiment analysis for mobile app reviews.
  * Topic Modeling: Classify reviews into Topic groups.
  * Finally, making the dashboard for a reviews sentiment report.

#### 1) Sentiment Scoring

The scoring is got from probability of the review to be a negative one. The **Stochastic Gradient Descent** regression model (SGDRegressor) is applied.

**Step 1: Labelling for review data**
     
Three methods are applied to recognize, and measure negative sentiment strenght of a review:
 
  * VADER: is the popular model which its self measure sentiment of a sentence. However, VADER is built for English (not Vietnamese). It might get more bias when run with a Vietnamese-English Google Translate review. Moreover, VADER itself does not have high accuracy.
     
  * Negative words and positive words: this is where we can add our insights into model based on our experience/understanding about the language. By the way, the list of critial negative words and positive words is defined which the appearance of them in the sentence could deliver the strong conclusion in sentiment. Based on my particular Vietnamese review dataset, I figure out those list of words, and set rule number of negative words > positive words, so the review likely negative sentiment.
  
  * Customer Rating: this is implicit assessment of review sentiment. Following that, the review of low rating (less than 3) could be assumed having negative sentiment.

Each method having one point. With one point more, the more sentence reaches to negative. In my demo, the sum of method point is used. Following that, each review has 4 values of point [0,1,2,3] with 3 is significant-negative, 2 is slight-negative, 1 is neutral, 0 is non-negative.

It is noted that the review data need processing before using. Processing including removing emoji, correct vietnamese words, translate to English, English stopwords processing, etc.

**Step 2: Vectorize data and Model train**

The review after processing including English translation fed into Tf-Idf Vectorizer to get the sentence vector.

Those embedded vector is used to train with label in step 1 by SGDRegressor.

The result return is regression value of the review. This value is used to calculate the sentiment score by formula (-2)\*(Pred_val/3 - 0.5). By this formula, sentiment score will be in \[-1, 1] with \[-1.0) is negative sentiment. 

#### 2) Topic Modeling

Topic Modeling uses BERTopic Model with using
```python
topic_model = BERTopic(top_n_words=15,
                       nr_topics = 30,
                       n_gram_range=(1,3), 
                       calculate_probabilities=True,
                       umap_model= umap_model,
                       hdbscan_model=hdbscan_model,
                       embedding_model=embedding_model,
                       vectorizer_model = vectorizor_model,
                       ctfidf_model=ctfidf_model,
                       #similarity_threshold_merging=0.5,
                       verbose=True,
                       language="english",
                       #calculate_probabilities=True,
                       min_topic_size=300 ) # better [100,500]
  ```
  * Vectorizor Model: CountVectorizer with ngram range in (1,3) with stopwords processing.
 ```python
 from sklearn.feature_extraction.text import CountVectorizer
 vectorizor_model = CountVectorizer(ngram_range=(1,3), stop_words=stopwords)
 ```
  * Sentence Transformer: "all-mpnet-base-v2"
 ```python
 from sentence_transformers import SentenceTransformer
 embedding_model = SentenceTransformer("all-mpnet-base-v2")
 ```
   * UMAP model to reduce embedding dimensions

```python
  import umap
  umap_model = umap.UMAP(n_neighbors=5,
                         n_components=5,
                         min_dist=0.05,
                         metric='cosine',
                         low_memory=False)
```                 
  * HBSSCAN Model for document clustering

```python  
   import hbscan
   hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=40,
                                   min_samples=20,
                                   metric='euclidean',
                                   cluster_selection_method='leaf',  #'eom'
                                   gen_min_span_tree=True,
                                   prediction_data=True)
```
The result getting from running by BERTopic Model give information of topic group such as 

  * **Basket of representative words in each group**: This is a hint for you to get the topic content.
  
<div align="center"><img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/review_topic_modeling/image/topic_representative_keywords.PNG" width="80%" height="80%" alt="Topic representative words">
  

  * **InterTopic distance map**: The figure shows you how likely the topics is similar each other. So you could group nearest distance topic groups into one.
 
 <img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/review_topic_modeling/image/topic_distance_map.PNG" width="40%" height="40%" alt="InterTopic distance map">
 
  * **Topic Hierarchical clustering**: Like the intertopic distance map, the hierarchical clustering supports you in combining similar topics in one to reduce the number of topics, and make result more condense.

 <img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/review_topic_modeling/image/topic_hierarchical_clustering.PNG" width="80%" height="80%" alt="Topic Hierarchical clustering">
 
<div align="left">
 
#### 3) Sentiment Dashboard Report
 
 <img
src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/sentiment_report/image_export/1_review_by_month_barline_vib2_android.png" width="80%" height="80%" alt="Number of reviews by months">
 
 
 <img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/sentiment_report/image_export/1_sentiment_by_month_barline_vib2_android.png" width="80%" height="80%" alt="Sentiment by months">
 
<img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/sentiment_report/image_export/2_percent_share_of_review_rating_score_12m_vib2_android.png" width="80%" height="80%" alt="Review rating percentage in March 2023">
  
 <img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/sentiment_report/image_export/3_valid_reviews_by_month_table_vib2_android.png" width="80%" height="80%" alt="Number review by Topics">
 
 <img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/sentiment_report/image_export/4_valid_reviews_by_month_table_vib2_android.png" width="80%" height="80%" alt="Number review by Topics">
 
<img src="https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/sentiment_report/image_export/5_sentiment_rating_by_topic_month_table_vib2_android.png" width="80%" height="80%" alt="Sentiment rating of topics by months">

 
 - - - - - 
Feeling Free to contact me if you have any question around.

    Nhan Thanh Ngo (BE/MSc/MBA)
    Email: ngothanhnhan125@gmail.com
    Skype: ngothanhnhan125
    Phone: (+84) 938005052

