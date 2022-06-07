# Tweet-Classification
Using BERT and Neural Network Architecture to classify tweets

## PROBLEM STATEMENT:

Given: 

A) A labeled verified dataset (“Gold Standard”) of about 4,000 tweets. These tweets have been labeled antisemitic/ not antisemitic by expert annotators who went over the same tweets and discussed any disagreements. Tweets were taken out of the dataset if no agreement was found. The tweets in the dataset are from January 2019 to April 2021. The were scraped with one of the four keywords “Jews,” “Israel,” “kikes,” or “ZionNazi*.”

B) 500 tweets that were labeled antisemitic/ not antisemitic by a group of students (“lay annoytators”). The tweets in this dataset are from January to April 2021 and were scraped with the keyword “Jews.” The same tweets were also classified by the expert annotators and are part of the Gold Standard.

## TASK:
 
1) Using data from Gold Standard (data from A)

- Find a model that can predict antisemitic/ non-antisemitic tweets within the gold standard (take 80 or 90 percent of the data and run your test model on the remaining 20 or 10 percent). You might want to run your model separately for the tweets with generic keywords (“Jews” and “Israel”) and for the tweets with slur keywords (“kikes” and ZioNazi*”). Generate accuracy and efficiency results.
- Use name recognition tools for information extraction to identify prominent categories such as geopolitical entities, person names, organisations and locations in the antisemitic and in the non-antisemitic tweets.
- Examine the gold standard for linguistic radicalization patterns. At what periods do derogatory and pejorative terms and slurs appear?
- Visualize differences between antisemitic and non-antisemitic tweets.

## BUILT WITH / MODULES USED

Built with Python 3.7. Modules used:
 - SMOTE
 - BERT as a Service
 - Keras
 - Tensorflow
 
 
 ## RESULTS

The Final report with data cleaning, EDA, model results and visualizations are all present in the notebook file 'Tweet_Classification_report' added to the repository
