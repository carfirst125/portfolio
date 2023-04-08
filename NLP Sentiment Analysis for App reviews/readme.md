## Sentiment Analysis for Mobile App Reviews 

This demo shows you contents as following:
  * How to make sentiment analysis for mobile app reviews.
  * Topic Modeling: Classify reviews into Topic groups.
  * Finally, making the dashboard for a reviews sentiment report.

#### Sentiment Scoring


#### Topic Modeling

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
  
  ![alt text](https://github.com/carfirst125/portfolio/blob/main/NLP%20Sentiment%20Analysis%20for%20App%20reviews/review_topic_modeling/image/topic_representative_keywords.PNG?raw=true){:height=60% width=60%}

  * **InterTopic distance map**: The figure shows you how likely the topics is similar each other. So you could group nearest distance topic groups into one.
  * **Topic Hierarchical clustering**: Like the intertopic distance map, the hierarchical clustering supports you in combining similar topics in one to reduce the number of topics, and make result more condense.

