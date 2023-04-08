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
  * HBSSCAN Model for document clustering
