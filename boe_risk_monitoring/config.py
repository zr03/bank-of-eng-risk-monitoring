import os

DATA_FOLDER = "data"
AGGREGATED_DATA_FOLDER_NAME = "aggregated"
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