import streamlit as st
import pandas as pd
import time
import tensorflow as tf
import random
import numpy as np

def load_model(model_path):
    # Load the saved model using TensorFlow
    model = tf.keras.models.load_model(model_path)
    return model


def load_classes():
    # Loading the classes and their labels
    classes = pd.read_csv("csv_files/my_models/classes.csv")
    return classes

def load_labels():
    # Loading labels
    labels = pd.read_csv("csv_files/my_models/labels.csv")
    return labels


def get_class_label(classes, label):
    # Get actual label of the dataset
    attack_type = classes.loc[classes['attack_label'] == label, 'attack_type'].values[0]
    return attack_type


def predict(model, data):
    prediction = model.predict(data)
    return prediction

def deploy_model(csv_data, model, labels, classes):
    while True:
        # Getting random data using randint function
        random_num = random.randint(0, len(csv_data)-1)
        # Getting random value from dataset
        test_row = csv_data.iloc[[random_num]].to_numpy()
        st.write("Random Data:")
        st.write(csv_data.iloc[random_num])
        # Getting prediction of random data
        predictions = predict(model, test_row)
        predictions = [np.argmax(i) for i in predictions]
        actual_label = labels.iloc[random_num][0]
        pred_label = get_class_label(classes, predictions[0])
        actual_label = get_class_label(classes, actual_label)
        st.write(f"Actual Label: {actual_label}")
        st.write(f"Predicted Label: {pred_label}")
        time.sleep(5)

def deploy_ano_model(csv_data, classes, model):
    while True:
        # Getting random data using randint function
        random_num = random.randint(0, len(csv_data)-1)
        test_row = csv_data.iloc[[random_num]].to_numpy()
        st.write("Random Data:")
        st.write(csv_data.iloc[random_num])
        # Getting prediction of random data
        predictions = predict(model, test_row)
        mse = np.mean(np.power(np.array(csv_data) - predictions, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred = [1 if e > threshold else 0 for e in mse]            
        pred_label = "Normal" if y_pred[random_num] == 1 else "Attack"
        actual_label = "Normal" if classes.label.iloc[random_num] == 1 else "Attack"
        st.write(f"Actual Label: {actual_label}")
        st.write(f"Predicted Label: {pred_label}")
        time.sleep(5)
    
def main():
    st.title("Deploy Pretrained models")
    selected_model = st.selectbox("Select pretrained model", ("Model 1: Imbalanced Dataset Model", 
                                                              "Model 2: Multiclass Classification Model",
                                                              "Model 3: Anamoly Detection Model", ))
    selected_model = "".join(selected_model.split(":")[0]).replace(" ","").lower()
    # Loading saved model 
    model = load_model(f"saved_models/my_models/{selected_model}.h5")
    if selected_model == "model3":
        csv_data = pd.read_csv("csv_files/my_models/ano_data.csv")
        classes = pd.read_csv("csv_files/my_models/ano_classes.csv")
        if st.button("Deploy Model"):
            deploy_ano_model(csv_data, classes, model)
            
    else:
        csv_data = pd.read_csv("csv_files/my_models/data.csv")
        # Load class labels
        classes = load_classes()
        labels = load_labels()
        if st.button("Deploy Model"):
            deploy_model(csv_data, model, labels, classes)
    

if __name__ == "__main__":
    main()

