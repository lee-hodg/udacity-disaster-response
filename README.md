# Disaster Response Pipeline Project

This project uses disaster response data from [Figure 8](https://appen.com/).
During and after a natural disaster there are millions of messages sent to aid organizations
, right when their resources are already being stretched to the limit. 
The goal of this project is to use NLP techniques to classify those messages into
the correct categories so that they can be routed to the relevant organization.

## Web application screenshots

![Index](images/screenshot1.png?raw=true "Dashboard 1")
![Index 2](images/screenshot2.png?raw=true "Dashboard 2")
![Index 3](images/screenshot3.png?raw=true "Classify Message")


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

## Files in this repository 

- app
  - templates
    - go.html  # classification result page of web app
    - master.html  # main page of web app
  - run.py  # Flask web app
- data
  - __init__.py
  - disaster_categories.csv   # data to process
  - diaster_messages.csv  # data to process
  - process_data.py  # ETL script
- models
  - __init__.py 
  - train_classifier.py   # ML pipeline script
  
- README.md   # This README
- requirements.txt   # Third-party dependencies
- settings.py  # Common config settings
- utils.py   # Re-usable functionality

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.
   
### To run ETL pipeline that cleans data and stores in database

```bash
cd data
python process_data.py --m disaster_messages.csv -c disaster_categories.csv --d ../DisasterResponse.db
```

### To run ML pipeline that trains classifier and saves

```bash
cd models
python train_classifier.py -d '../DisasterResponse.db' -p '../saved_model.pkl'
```

Optionally you can specify the `-o` to run a grid search to optimize the parameters, and
also `-u` to "upsample" the test data to try and address the imbalance in category labels
(i.e. many messages labeled with some categories and very few with other categories).

### Run the following command in the app's directory to run your web app.

```bash
cd app
python run.py -p../saved_model.pkl -d ../DisasterResponse.db
```

Go to http://0.0.0.0:3001/

