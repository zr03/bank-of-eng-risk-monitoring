import os

DATA_FOLDER_PATH = "data"
NEWS_DATA_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "news_all_banks")
AGGREGATED_DATA_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "aggregated")
APP_DATA_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "app")
SHARE_PRICE_HISTORY_START_DATE = "2022-01-01"
PERMISSIBLE_BANK_NAMES = ["citigroup", "jpmorgan", "bankofamerica"] # These match the folder names in the data directory
BANK_NAME_MAPPING = {
    "citigroup": "Citigroup",
    "jpmorgan": "JPMorgan",
    "bankofamerica": "Bank of America",
}

TICKER_MAPPING = {
    "citigroup": "C",
    "jpmorgan": "JPM",
    "bankofamerica": "BAC",
}

PERMISSIBLE_VECTOR_DB_PROVIDERS = ["pinecone"]
