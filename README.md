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

During disaster events a myraid of messages are sent either directly or through news and social media. 
It is important that relevant messages are sent to appropriate disaster relief agencies. 
In this project an ETL pipeline and a ML pipeline are developed to extract cleaned data and build a supervised machine learning model for an API that classifies disaster messages. 
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
The cleaned dataset was then saved into a sqlite database. 

#### ML Pipeline

Although there are features such as 'genre' and 'original', only 'message' was chosen as initial feature for this project.
Besides, there are 16045 missing values for 'original'. 
There are urls in the text messages, and they are replaced with 'urlplaceholder'. 
The text messages are processed to make lower cases, remove puntuations, and stop words. 
They are then tokenized and lemmatized. 

The ML pipeline consists of CountVectorizer and TfidfTransformer as text_pipeline along with two custom transformers 
TextLengthExtractor() which extracts text length and StartingVerbExtractor() which returns true if the first word is an appropriate verb or RT for retweet.
The text_pipeline and the custom transformers need to be processed in parallel. FeatureUnion from sklearn is used to achieve this goal. 
The rest of the ML pipeline includes a normalizer since the values for text lengths can be much larger than the values for other features,  and  XGBClassifier. 
Hyperparameters tuning with grid search was also performed.

#### App
A web app is developed where an emergency worker can input a new message and get classification results in several categories.
How to run the app can be found in the Instructions section.


![GitHub Logo](https://github.com/alwz1/disaster_response_pipelines/blob/master/images/Screen%20Shot%202020-05-10%20at%201.29.38%20PM.png)


### Licensing, Authors, Acknowledgements<a name="licensing"></a>
MIT License


