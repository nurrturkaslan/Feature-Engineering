# Feature-Engineering
The functions we created are included in a script. The necessary parts for pre-processing were taken. Analysis complete.

![image](https://user-images.githubusercontent.com/63192605/136262704-a9a500e0-582b-4bde-b0c0-2e5da3db985a.png)

*Business Problem*

/Required for a machine learning pipeline data preprocessing and variable engineering script needs to be prepared.

/When the dataset is passed through this script, the modeling starts. Expected to be ready.

*Dataset Story*
/The data set is the data set of the people who were in the Titanic shipwreck.
/It consists of 768 observations and 12 variables.

** The target variable is specified as "Survived";
** 1: one's survival,
** 0: indicates the person's inability to survive.

*Variables*
* Survived
* 0 Died, 1 Survived

* Pclass – Ticket Class
* 1 = Grade 1, 2 = Grade 2, 3 = Grade 3

* Age – Age

* Sibsp – Number of siblings / spouses on the Titanic

* Sex – Gender

* Parch – Number of parents/children on Titanic

* Embarked: – Passenger embarkation port
* (C = Cherbourg, Q = Queenstown, S = Southampton

* Fare – Ticket fare

* Cabin: Cabin number

*Project Tasks*


1- Open a directory called helpers in the working directory and enter it.
Add a script named data_prep.py.
In the Feature Engineering section, all of our own
collect functions into this script.
Functions that should be here:
▪ outlier_thresholds
▪ replace_with_thresholds
▪ check_outlier
▪ grab_outliers
▪ remove_outlier
▪ missing_values_table
▪ missing_vs_target
▪ label_encoder
▪ one_hot_encoder
▪ rare_analyser
▪ rare_encoder

2- Write a function called titanic_data_prep.
Data preprocessing or EDA functions required for this function,
Get it from the eda.py and data_prep.py files in the helpers.

3- Save the data set you preprocessed to the disk with pickle.
