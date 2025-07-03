# bank-of-eng-risk-monitoring
Risk monitoring tool for Global Systemically Important Banks developed for the Bank of England as part of Cambridge University's Data Science Career Accelerator

# Setting up your environment
1. Create a new environment (pip or conda) with python 3.11 or higher e.g. 'conda create -n boe python==3.11"

2. Activate the environment with 'conda activate boe'

3. Now install poetry which we use to manage dependencies in this project. You have two options. If you want a global install of poetry on your machine then follow step a) below. If you want to install poetry within your conda environment, follow step b) below.

a) Follow the instructions here to set up poetry on your machine. Make sure to add Poetry to PATH as per the instructions so you can call poetry form the command line
https://python-poetry.org/docs/#installing-with-the-official-installer
b) Run pip install poetry AND pip install poetry-plugin-export.

4. Clone/pull down the latest version of the repository.

5. Move into the root of the repository.

6. When it comes to installing the dependencies in your environment, note that there are three groups of dependencies as outlined in the toml file:
- Core dependencies found under [tool.poetry.dependencies] in the toml file. These are core dependencies common to both development and production settings.
- Dev dependencies found under [tool.poetry.group.dev.dependencies]. These are dependencies required to develop and run backend scripts and notebooks such as ETL and NLP. This is an optional group of dependencies - to install you need to be explicity in the poetry install command (see below).
- App dependencies found under [tool.poetry.group.app.dependencies]. These are dependencies required to develop and run the Dash app itself - to install you need to be explicity in the poetry install command (see below).

To install all dependencies including the optional dev and app dependency groups, run 'poetry install --all-groups'

To install core dependencies plus one of the optional group (e.g. say you only need to work on the app), then use 'poetry install --with app'

NOTE: If you chose to install poetry locally in your environment (option 3b above), you may need to call poetry through python in which case you replace all instances of 'poetry' in the commands above with 'python -m poetry'.

# Adding dependencies
To add a dependency to the core group simply run: poetry add 'package name'
To add a dependency to one of the custom groups run e.g.: poetry add 'package name' --group dev

# Environment variables
Some scripts will not run unless the following environment variables are set up in a .env file in the root of the repository.
OPENAI_API_KEY
GEMINI_API_KEY
PINECONE_API_KEY

# Running the app
To run the app, make sure you are in the root of the repository and run 'python app.py'
You will be able to develop the app while it is running (may need to manually refresh for certain code changes to appear)