import os

DATA_FOLDER = os.path.join("data", "app")
TOPIC_RELEVANCE_FPATH = os.path.join(DATA_FOLDER, "multi_topic_modelling_with_relevance.parquet")
TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH = os.path.join(DATA_FOLDER, "multi_topic_modelling_with_relevance_sentiment.parquet")
TOPIC_RELEVANCE_Q_AGG_FPATH = os.path.join(DATA_FOLDER, "multi_topic_modelling_with_relevance_quarter_agg.parquet")
TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_Q_AGG_FPATH = os.path.join(DATA_FOLDER, "multi_topic_modelling_with_relevance_sentiment_quarter_agg.parquet")
RISK_CATEGORY_MAPPING_FPATH = os.path.join(DATA_FOLDER, "risk_category_mapping.parquet")
Q_A_ANALYSIS_FPATH = os.path.join(DATA_FOLDER, "data_qa_pairs.parquet")
