import streamlit as st
import pandas as pd
import time
import tensorflow as tf
import random
import numpy as np

# Load the saved model using TensorFlow
#@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load the class labels
#@st.cache
def load_classes():
    classes = pd.read_csv("csv_files/classes.csv")
    return classes

def load_labels():
    labels = pd.read_csv("csv_files/labels.csv")
    return labels

# Function to get the attack type from the label
def get_class_label(classes, label):
    attack_type = classes.loc[classes['attack_label'] == label, 'attack_type'].values[0]
    return attack_type

# Function to make predictions using the loaded model
def predict(model, data):
    prediction = model.predict(data)
    return prediction

def deploy_model(csv_data, model, labels, classes):
    while True:
        random_num = random.randint(0, len(csv_data)-1)
        test_row = csv_data.iloc[[random_num]].to_numpy()
        st.write("Random Data:")
        st.write(csv_data.iloc[random_num])
        predictions = predict(model, test_row)
        predictions = [np.argmax(i) for i in predictions]
        actual_label = labels.iloc[random_num][0]
        pred_label = get_class_label(classes, predictions[0])
        actual_label = get_class_label(classes, actual_label)
        st.write(f"Actual Label: {actual_label}")
        st.write(f"Predicted Label: {pred_label}")
        time.sleep(5)
    
def main():
    st.title("Deploy Pretrained models")
    selected_model = st.selectbox("Select pretrained model", ("Model 1", "Model 2"))
    # Load saved model 
    selected_model = selected_model.replace(" ", "").lower()
    csv_data = pd.read_csv("csv_files/data.csv")
    model = load_model(f"saved_models/{selected_model}.h5")
    # Load class labels
    classes = load_classes()
    labels = load_labels()
        
    if st.button("Deploy Model"):
        deploy_model(csv_data, model, labels, classes)
    

if __name__ == "__main__":
    main()

