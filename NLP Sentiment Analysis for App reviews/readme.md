## Sentiment Analysis for Mobile App Reviews 

This demo shows you contents as following:
  * How to make sentiment analysis for mobile app reviews.
  * Topic Modeling: Classify reviews into Topic groups.
  * Finally, making the dashboard for a reviews sentiment report.

#### Sentiment Scoring


#### Topic Modeling

Topic Modeling uses BERTopic Model with using
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
```python
topic_model = BERTopic(top_n_words=15,
                       nr_topics = 30,
                       n_gram_range=(1,3), 
                       calculate_probabilities=True,
                       umap_model= umap_model,
                       hdbscan_model=hdbscan_model,
                       embedding_model=embedding_model,#word_doc_embedder,#embedding_model,
                       vectorizer_model = vectorizor_model,
                       ctfidf_model=ctfidf_model,
                       #similarity_threshold_merging=0.5,
                       verbose=True,
                       language="english",
                       #calculate_probabilities=True,
                       min_topic_size=300 ) # better [100,500]
  ```
