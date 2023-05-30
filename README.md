The aqi project contains 3 sub-directories:

API, models and restored_csvs:
* API contains the necessary files to deploy our ML model locally 
(README.md file inside).
* models is the directory used to store the different models obtained 
for different stations and hyper-parameters 
* Restored_csvs is not of general interest.

# functions.py
The aqi project contains one python file, functions.py. It contains the
general functions used to develop the project and general documentation.


The aqi project contains two jupyter notebooks main_analysis.ipynb and 
EDA.ipynb:
# main_analysis.ipynb:
 Divided for daily case and monthly case.
 It is the first document advised to check. It is the main analysis
 carried on to create the models. It make use of the 
 general functions found in functions.py. It contains some general
 information about the process and the general code to create the 
 models.

 # EDA.ipynb:
 Divided for daily case and monthly case.
 It contains exploratory analysis and simplified code needed to build 
 the general functions and models. It contains the non-abstract method 
 (not very organized) used in main_analysis.ipynb. 



