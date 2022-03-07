
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 23:35:37 2022

@author: apple
"""


import pandas as pd
import re
from bert_serving.client import BertClient
from keras.layers import Conv2D
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pickle
import emoji
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
from sklearn.ensemble import VotingClassifier

def performance(x, y):
    
    print(confusion_matrix(x, y))
    
    print(classification_report(x, y))
    print('F-1 score: ',f1_score(x, y, average = 'macro'))
    print('Accuracy: ',accuracy_score(x, y))
    print('Balanced Accuracy: ',balanced_accuracy_score(x, y))
    print('ROC AUC: ', roc_auc_score(x, y,  average = 'macro'))
    
    fpr, tpr, threshold = roc_curve(x, y)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

bc = BertClient(check_length = False)

df = pd.read_csv('train_dataset (1).csv', encoding = 'latin-1')

df.drop(['ID','create_date','user','Sample.ID.x','Sample.ID.y','Still.Exists.x','Still.Exists.y','In.English.x',
         'In.English.y','User.x','User.y','sample_name'],axis=1,inplace=True)

df['Is.About.The.Holocaust.x'] = df['Is.About.The.Holocaust.x'].fillna(df['Is.About.the.Holocaust.x'])
df['Is.About.The.Holocaust.y'] = df['Is.About.The.Holocaust.y'].fillna(df['Is.About.the.Holocaust.y'])
df.drop(['Is.About.the.Holocaust.x','Is.About.the.Holocaust.y'],axis=1,inplace=True)

def text_clean(x):

    ### Light
    #x = x.lower() # lowercase everything
    x = x.encode('ascii', 'ignore').decode()  # remove unicode characters
    x = re.sub(r'https*\S+', ' ', x) # remove links
    x = re.sub(r'http*\S+', ' ', x)
    # cleaning up text
    x = re.sub(r'\'\w+', '', x) 
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    x = re.sub(r'\s[^\w\s]\s', '', x)
    x = emoji.demojize(x, language='en')
    
    return x

def cleaning(df, val):

    df.columns = df.columns.str.replace('.', ' ')
    
    df = df.astype(str)
    
    columns = df.columns
    
    df['Full_text']=''
    
    for index, row in df.iterrows():
       
        combined = ""
        
        if row['Sarcasm'] == 'False':
            combined += 'Annotator '+val+' thinks this tweet is not sarcastic. '
        else:
            combined += 'Annotator '+val+' thinks this tweet is sarcastic. '
        
        if row['Disagree With'] == 'True':
            combined += 'Annotator '+val+' thinks this tweet is anti-semitic but cannot justify it with the IHRA guidelines. '  
        
        if row['Is About The Holocaust'] == '0.0':
            combined += 'Annotator '+val+' thinks this tweet is not about the holocaust. '
        else:
            combined += 'Annotator '+val+' thinks this tweet is about the holocaust. '
            
        if row['IHRA Section'] == '13':
            combined += 'Annotator '+val+' thinks that this tweet does not fall under IHRA guidelines and is therefore not anti-semitic. '
        else:
            combined += 'Annotator '+val+' thinks that this tweet falls under IHRA guideline '+row['IHRA Section']+' and is therefore anti-semitic. '
        
        if row['Calling Out'] == '0':
            combined += 'Annotator '+val+' thinks this tweet is not calling out anti-semitics. '
        else:
            combined += 'Annotator '+val+' thinks this tweet is calling out anti-semitics. '
        
        if row['Sentiment Rating']=='1':
            combined += 'Annotator '+val+' thinks that this tweet is very negative. '
        elif row['Sentiment Rating']=='2':
            combined += 'Annotator '+val+' thinks that this tweet is negative. '
        elif row['Sentiment Rating']=='3':
            combined += 'Annotator '+val+' thinks that this tweet is neutral. '
        elif row['Sentiment Rating']=='4':
            combined += 'Annotator '+val+' thinks that this tweet is positive. '
        else:
            combined += 'Annotator '+val+' thinks that this tweet is very positive. '
        
        if row['Additional Comments'] != 'No comments.':
            combined += 'Additional comments by Annotator '+val+': {:}'.format(row['Additional Comments']) 
        
        df['Full_text'][index] = combined
        
    return df
        

df_x = df.loc[:, df.columns.str.endswith(".x")]
df_x.columns = df_x.columns.str.rstrip('.x')

clean_df_x = cleaning (df_x, 'X')

df_y = df.loc[:, df.columns.str.endswith(".y")]
df_y.columns = df_y.columns.str.rstrip('.y')

clean_df_y = cleaning (df_y, 'Y')

df['Text_x'] = clean_df_x['Full_text']
df['Text_y'] = clean_df_y['Full_text']

df['full_text']=df['full_text'].apply(lambda x: text_clean(str(x)))
df['Text_x']=df['Text_x'].apply(lambda x: text_clean(str(x)))
df['Text_y']=df['Text_y'].apply(lambda x: text_clean(str(x)))

    
df['Full_Text'] = ''

for index, row in df.iterrows():
   
    combined = ""
    
    combined += row["full_text"]
    
    combined += "\n\n"

    combined += row["Text_x"]
    
    combined += "\n\n"

    combined += row["Text_y"]

    df['Full_Text'][index] = combined
    

df1 = df[df['key'].isin(['IsraelNAS', 'JewNAS', 'IsraelAS', 'JewAS'])]
df2 = df[df['key'].isin(['ZioNaziAS', 'ZioNaziNAS', 'KikesNAS', 'KikesAS'])]


X1 = df1['Full_Text']
X1 = bc.encode(list(X1))
y1 = df1['Target']

X2 = df2['Full_Text']
X2 = bc.encode(list(X2))
y2 = df2['Target']


def nn_model(X_train, y_train):
    
    model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation = 'relu', input_shape = (768,)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                      metrics = ['accuracy'])
        
    model.fit(X_train, y_train, epochs = 75, batch_size = 100)
    
    return model

#Model 1

over = SMOTE(sampling_strategy=0.3)
X_over, y_over = over.fit_resample(X1, y1)

model1 = nn_model(X_over, y_over)
model2 = nn_model(X2, y2)



#Test data set
df = pd.read_csv('/Users/apple/Downloads/test_dataset (1).csv', encoding = 'latin-1')

df.drop(['ID','create_date','user','Sample.ID.x','Sample.ID.y','Still.Exists.x','Still.Exists.y','In.English.x',
         'In.English.y','User.x','User.y','sample_name'],axis=1,inplace=True)

df['Is.About.The.Holocaust.x'] = df['Is.About.The.Holocaust.x'].fillna(df['Is.About.the.Holocaust.x'])
df['Is.About.The.Holocaust.y'] = df['Is.About.The.Holocaust.y'].fillna(df['Is.About.the.Holocaust.y'])
df.drop(['Is.About.the.Holocaust.x','Is.About.the.Holocaust.y'],axis=1,inplace=True)

df_x = df.loc[:, df.columns.str.endswith(".x")]
df_x.columns = df_x.columns.str.rstrip('.x')

clean_df_x = cleaning (df_x, 'X')

df_y = df.loc[:, df.columns.str.endswith(".y")]
df_y.columns = df_y.columns.str.rstrip('.y')

clean_df_y = cleaning (df_y, 'Y')

df['Text_x'] = clean_df_x['Full_text']
df['Text_y'] = clean_df_y['Full_text']

df['full_text']=df['full_text'].apply(lambda x: text_clean(str(x)))
df['Text_x']=df['Text_x'].apply(lambda x: text_clean(str(x)))
df['Text_y']=df['Text_y'].apply(lambda x: text_clean(str(x)))

df['Full_Text'] = ''

for index, row in df.iterrows():
   
    combined = ""
    
    combined += row["full_text"]
    
    combined += "\n\n"

    combined += row["Text_x"]
    
    combined += "\n\n"

    combined += row["Text_y"]

    df['Full_Text'][index] = combined


df_new= pd.DataFrame(zip(df['Full_Text'], df['key']), columns=("Full_Text", "Key"))

df_new['row_num'] = np.arange(len(df_new))

df_1 = df_new[df_new['Key'].isin(['IsraelNAS', 'JewNAS', 'IsraelAS', 'JewAS'])]
df_2 = df_new[df_new['Key'].isin(['ZioNaziAS', 'ZioNaziNAS', 'KikesNAS', 'KikesAS'])]

X1 = df_1['Full_Text']
X1 = bc.encode(list(X1))
#X1_new = X1.reshape(-1, 768, 1)
prediction1 = model1.predict(X1, batch_size = 100).round()



X2 = df_2['Full_Text']
X2 = bc.encode(list(X2))
#X2_new = X2.reshape(-1, 768, 1)
prediction2 = model2.predict(X2, batch_size = 100).round()

pred1 = []
for item in prediction1:
    pred1.append(int(item[0]))

pred2 = []
for item in prediction2:
    pred2.append(int(item[0]))


df_pred = pd.concat([pd.DataFrame(zip(pred1, df_1['row_num']), columns=("Target", "row_num")), pd.DataFrame(zip(pred2, df_2['row_num']), columns=("Target", "row_num"))])

df_pred.sort_values("row_num")
df_pred = df_pred.reset_index(drop=True)

df['Target']=df_pred['Target']

res = pd.DataFrame(df_pred['Target']) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = df.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["Target"]
res.to_csv("prediction_results_ann.csv", index = False)      # the csv file will be saved locally on the same location where this notebook is located.
