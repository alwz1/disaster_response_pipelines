# Disaster Response Pipeline Project


### Table of Contents
1. [Installations](#installations)
2. [Project Motivation](#project_motivation)
3. [File Descriptions](#file_descriptions)
4. [Results](#results)
5. [Licensing, Authors, Acknowledgements](#licensing)

### Installations<a name="installations"></a>
Standard libraries installed with the Anaconda distribution.

python 3.7.4

### Project Motivation<a name="project_motivation"></a>



### File Descriptions<a name="file_descriptions"></a>
data
* DisasterResponse.db
* disaster_categories.csv
* disaster_messages.csv
* process_data.py : ETL pipeline

models
* classifier.pkl
* train_classifier.py : ML pipeline

app
* templates
  * go.html : flask template
  * master.html : flask template
  
* run.py : flask main run file

images : screen shots of web application

ETL Pipeline Preparation.ipynb: ETL Pipeline Preparation Jupyter notebook

ML Pipeline Preparation.ipynb: ML Pipeline Preparation Jupyter notebook


LICENSE.txt: MIT License

### Instructions:
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


