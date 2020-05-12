# Disaster Response Pipeline Project


### Table of Contents
1. [Installations](#installations)
2. [Project Motivation](#project_motivation)
3. [File Descriptions](#file_descriptions)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors, Acknowledgements](#licensing)

### Installations<a name="installations"></a>
python 3.7.4
* numpy
* pandas
* sqlalchemy
* sqlite3
* joblib
* re
* nltk
* sklearn
* xgboost
* json
* plotly
* flask

### Project Motivation<a name="project_motivation"></a>

During disaster events a myraid of messages are sent either directly or through news and social media. It is important that relevant messages are sent to appropriate disaster relief agencies. In this project an ETL pipeline and a ML pipeline are developed to extract cleaned data and build a supervised learning model for an API that classifies disaster messages. 
The disaster dataset is provided by Figure Eight and it contains real messages that were sent during disaster events. 
A web app is also developed where an emergency worker can input a new message and get classification results in several categories.

### File Descriptions<a name="file_descriptions"></a>
data
* disaster_categories.csv : categories dataset
* disaster_messages.csv : message dataset
* process_data.py : ETL pipeline script

models
* train_classifier.py : ML pipeline script

app
* templates
  * go.html : flask template
  * master.html : flask template
  
* run.py : flask main run script

images : screen shots of web application

ETL Pipeline Preparation.ipynb: ETL Pipeline Preparation Jupyter notebook

ML Pipeline Preparation.ipynb: ML Pipeline Preparation Jupyter notebook


LICENSE.txt: MIT License

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Results<a name="resluts"></a>
#### ETL Pipeline


There are two datasets- one for messages and the other for categories. 
The datasets are merged into a dataframe on id column that is common for both. 
String type categories under a single column were split and expanded into 36 individual category columns, and converted into numeric value of either 0 or 1. 
The duplicate 171 rows were also removed. 
The resulting dataframe had 26215 rows and 40 columns.
The cleaned dataset was then saved into an sqlite database. 


### Licensing, Authors, Acknowledgements<a name="licensing"></a>


