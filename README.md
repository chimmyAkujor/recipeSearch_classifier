# Data Mining Project - Recipe Search and Classification
This project has two parts. 
i. Recipe Search (Textsearch.py) searches through a recipe dataset using ingredients as keywords, the results are ranked using tf-idf weight scores.
ii. Recipe Classification (NaiveBayes.py) classifies a group of ingredient keywords and tries to predict which cuisine it belongs to. Naive Bayes is used to calculate the probability.

## How to Run
```bash
#run following commands from current directory
# Install Dependencies
# the dependencies required to run this project are already curated in the requirements.txt file
# just run the following command to install the dependencies

pip3 install -r requirements.txt

#app.py is the main flask file that starts entire application. 
#start server on localhost:5000
python3 app.py

```

### then go to http://127.0.0.1:5000 or  http://localhost:5000 on your browser to access the homepage
# Reference
https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/
https://github.com/BhaskarTrivedi/QuerySearch_Recommentation_Classification
https://www.youtube.com/watch?v=CPqOCI0ahss
