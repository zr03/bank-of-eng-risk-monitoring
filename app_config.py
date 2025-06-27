import os

DATA_FOLDER_PATH = os.path.join("data", "app")
TOPIC_RELEVANCE_FPATH = os.path.join(DATA_FOLDER_PATH, "multi_topic_modelling_with_relevance.parquet")
TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH = os.path.join(DATA_FOLDER_PATH, "multi_topic_modelling_with_relevance_sentiment.parquet")
TOPIC_RELEVANCE_Q_AGG_FPATH = os.path.join(DATA_FOLDER_PATH, "multi_topic_modelling_with_relevance_quarter_agg.parquet")
TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_Q_AGG_FPATH = os.path.join(DATA_FOLDER_PATH, "multi_topic_modelling_with_relevance_sentiment_quarter_agg.parquet")
RISK_CATEGORY_MAPPING_FPATH = os.path.join(DATA_FOLDER_PATH, "risk_category_mapping.parquet")
