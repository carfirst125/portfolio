## Sentiment Analysis for Mobile App Reviews 

This demo shows you contents as following:
  * How to make sentiment analysis for mobile app reviews.
  * Topic Modeling: Classify reviews into Topic groups.
  * Finally, making the dashboard for a reviews sentiment report.

#### Sentiment Scoring


#### Topic Modeling

Topic Modeling uses BERTopic Model with using
  * Vectorizor Model: CountVectorizer with ngram range in (1,3) with stopwords processing.
  * Sentence Transformer: "all-mpnet-base-v2"
  * UMAP model to reduce embedding dimensions

    umap_model = umap.UMAP(n_neighbors=5,
                           n_components=5,
    Skype: ngothanhnhan125
    Phone: (+84) 938005052
    
  umap_model = umap.UMAP(n_neighbors=5,
  n_components=5,
  min_dist=0.05,
  metric='cosine',
  low_memory=False)
                          
  * HBSSCAN Model for document clustering
  
     hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=40,
                                     min_samples=20,
                                     metric='euclidean',
                                     cluster_selection_method='leaf',  #'eom'
                                     gen_min_span_tree=True,
                                     prediction_data=True)
