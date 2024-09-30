import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import mysql.connector
from keras.models import Sequential
import tensorflow as tf

# Connect to MySQL database
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
    return df

# Model loading
model_path = 'C:/Users/N/Desktop/New folder/lstm_model.keras'
model = None
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully.")
except ValueError as e:
    st.error(f"Error loading model: {e}")

# Data Loading
col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 
             's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
                     's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# Load data
dataset_test = fetch_data_to_dataframe('pm_test').drop('id', axis=1)
dataset_test.columns = col_names

pm_truth = fetch_data_to_dataframe('pm_truth').drop('id', axis=1)
pm_truth.columns = ['more']
pm_truth['id'] = pm_truth.index + 1

# Preprocessing
rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
pm_truth['rtf'] = pm_truth['more'] + rul['max']
pm_truth.drop('more', axis=1, inplace=True)
dataset_test = dataset_test.merge(pm_truth, on=['id'], how='left')
dataset_test['ttf'] = dataset_test['rtf'] - dataset_test['cycle']
dataset_test.drop('rtf', axis=1, inplace=True)

df_test = dataset_test.copy()
period = 30
df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)

# Scale features
sc = MinMaxScaler()
df_test[features_col_name] = sc.fit_transform(df_test[features_col_name])

# Generate sequences for LSTM
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length-1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# Predict probability of failure for a machine
def prob_failure(machine_id):
    machine_df = df_test[df_test.id == machine_id]
    machine_test = gen_sequence(machine_df, 50, features_col_name)
    m_pred = model.predict(machine_test)
    failure_prob = list(m_pred[-1] * 100)[0]
    return failure_prob

# Streamlit App UI
st.title("Machine Failure Prediction")
st.write("Select a machine to predict the probability of failure within 30 days.")

# Machine ID selection
machine_id = st.selectbox("Select Machine ID:", df_test['id'].unique())

if st.button("Predict Failure Probability"):
    if model:
        failure_probability = prob_failure(machine_id)
        st.write(f"Probability of failure for machine {machine_id}: {failure_probability:.2f}%")
    else:
        st.error("Model not loaded.")

# Metrics and evaluation
def predictions():
    X_test_sequences = np.concatenate([gen_sequence(df_test[df_test['id'] == id], 50, features_col_name) for id in df_test['id'].unique()])
    y_test = df_test['label_bc'].values.reshape(-1, 1)
    y_pred_prob = model.predict(X_test_sequences)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

if st.button("Evaluate Model"):
    if model:
        accuracy, cm = predictions()
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        st.write("Confusion Matrix:", cm)
    else:
        st.error("Model not loaded.")
