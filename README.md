# Disaster Response Pipeline Project

This project uses disaster response data from [Figure 8](https://appen.com/).
During and after a natural disaster there are millions of messages sent to aid organizations
, right when their resources are already being stretched to the limit. 
The goal of this project is to use NLP techniques to classify those messages into
the correct categories so that they can be routed to the relevant organization.

## Project Components

This project has three main components.

### 1. ETL Pipeline
   
The ETL script does the following:

- Load messages and categories datasets
- Merge and clean the data
- Save the cleaned data in a SQLite database

### 2. ML Pipeline

- Loads the cleaned data from the database
- Splits the data into train and test sets
- Builds an ML Pipeline using NLTK and tunes parameters with grid search
  that can predict categories (multi-class classification) for the messages.
- Evaluates the model performance on the test set
- Saves the model weights in a pickle file

### 3. Flask app

- A Flask web app to display various charts about the data
- A form that takes a new message and classifies it by categories in real-time.


## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

