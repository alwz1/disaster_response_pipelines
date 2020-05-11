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


### Licensing, Authors, Acknowledgements<a name="licensing"></a>


