import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import mysql.connector
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use('ggplot')

col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12'
             ,'s13','s14','s15','s16','s17','s18','s19','s20','s21']
features_col_name = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12'
             ,'s13','s14','s15','s16','s17','s18','s19','s20','s21']

# Load the model
model_path = 'C:/Users/N/Desktop/New folder/lstm_model.keras'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except ValueError as e:
    print(f"Error loading model: {e}")


def conn():
    #conn = mysql.connector.connect(host="localhost", database = 'Student',user="root", passwd="root",use_pure=True)
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='researchwork'
    )

    cursor = conn.cursor()
    return cursor

# Function to fetch data into DataFrame
def fetch_data_to_dataframe(table_name):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='researchwork'
    )

    cursor = conn.cursor()

    cursor.execute(f'SELECT * FROM {table_name}')
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    #conn.close()
    return df

#dataset_test=pd.read_csv('PM_test.txt',sep=' ',header=None).drop([26,27],axis=1)
dataset_test = fetch_data_to_dataframe('pm_test').drop('id', axis=1)
dataset_test.columns=col_names
#dataset_test.head()
print('Shape of Test dataset: ',dataset_test.shape)
dataset_test.head()

#pm_truth=pd.read_csv('PM_truth.txt',sep=' ',header=None).drop([1],axis=1)
pm_truth = fetch_data_to_dataframe('pm_truth').drop('id', axis=1)
pm_truth.columns=['more']
pm_truth['id']=pm_truth.index+1
pm_truth.head()

# generate column max for test data
rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
rul.head()

# run to failure
pm_truth['rtf']=pm_truth['more']+rul['max']
pm_truth.head()

pm_truth.drop('more', axis=1, inplace=True)
dataset_test=dataset_test.merge(pm_truth,on=['id'],how='left')
dataset_test['ttf']=dataset_test['rtf'] - dataset_test['cycle']
dataset_test.drop('rtf', axis=1, inplace=True)
dataset_test.head()

df_test=dataset_test.copy()
period=30
df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)
df_test.head()

target_col_name='label_bc'

sc=MinMaxScaler()
df_test[features_col_name]=sc.fit_transform(df_test[features_col_name])

'''# Function to generate sequences
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    sequences = []
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        sequences.append(data_array[start:stop, :])
    return sequences

# Function to generate labels
def gen_label(id_df, seq_length, label):
    num_elements = id_df.shape[0]
    y_label = []

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label].iloc[stop])

    return np.array(y_label).reshape(-1, 1)  # Ensure labels are 2D arrays

# Define sequence length and columns
seq_length = 50
seq_cols = features_col_name

# Generate X_test and y_test with consistent lengths
X_test_sequences = []
y_test_sequences = []
for id in df_test['id'].unique():
    id_df = df_test[df_test['id'] == id]
    sequences = gen_sequence(id_df, seq_length, seq_cols)
    labels = gen_label(id_df, seq_length, 'label_bc')
    
    if len(sequences) != len(labels):
        print(f"Mismatch in number of sequences and labels for id {id}:")
        print(f"Number of sequences: {len(sequences)}")
        print(f"Number of labels: {len(labels)}")
    else:
        X_test_sequences.extend(sequences)
        y_test_sequences.extend(labels)

# Ensure sequences and labels have the same length
print(f"Total number of X_test sequences: {len(X_test_sequences)}")
print(f"Total number of y_test labels: {len(y_test_sequences)}")

if len(X_test_sequences) == len(y_test_sequences):
    X_test = np.array(X_test_sequences)
    y_test = np.array(y_test_sequences).reshape(-1, 1)  # Ensure labels are 2D arrays
    print("Final shape of X_test:", X_test.shape)
    print("Final shape of y_test:", y_test.shape)
else:
    print("Mismatch in the final number of X_test sequences and y_test labels")
'''


def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length-1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)


# function to generate labels
def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length-1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label = []
    
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label].iloc[stop])
    
    return np.array(y_label)

# timestamp or window size
seq_length=50
seq_cols=features_col_name

# generate X_test
X_test=np.concatenate(list(list(gen_sequence(df_test[df_test['id']==id], seq_length, seq_cols)) for id in df_test['id'].unique()))
print(X_test.shape)
# generate y_test
y_test=np.concatenate(list(list(gen_label(df_test[df_test['id']==id], 50, seq_cols,'label_bc')) for id in df_test['id'].unique()))
print(y_test.shape)

# testing metrics
def testMetric():
    scores2 = model.evaluate(X_test, y_test, verbose=1, batch_size=200)
    print('Accurracy: {}'.format(scores2[1]))
    return scores2[1]

# Get predicted probabilities
def predictions():
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32") # Convert probabilities to class labels (assuming binary classification with a threshold of 0.5)
    # If it's multi-class classification, use argmax
    # y_pred = np.argmax(y_pred_prob, axis=1)
    print('Accuracy of model on test data:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    return [accuracy_score(y_test, y_pred),confusion_matrix(y_test, y_pred)]

def prob_failure(machine_id):
    machine_df=df_test[df_test.id==machine_id]
    machine_test=gen_sequence(machine_df,seq_length,seq_cols)
    m_pred=model.predict(machine_test)
    failure_prob=list(m_pred[-1]*100)[0]
    return failure_prob

def machinefail(machine_id=16):
    print('Probability that machine will fail within 30 days: ',prob_failure(machine_id))
    return prob_failure(machine_id)

machinefail(machine_id=16)