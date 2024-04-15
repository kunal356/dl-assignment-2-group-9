import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import streamlit as st
import random
import time


def load_data():
    with st.spinner("Loading Data"):
        #Loading Data from csv file
        data = pd.read_csv("dataset/kddcup99.csv")
        # Removing duplicates from the dataset
        data = data.drop_duplicates()
    return data

def data_preprocessing(data_1):
    with st.spinner("Preprocessing Data"):
        encoder = LabelEncoder()
        unique_labels =data_1['label'].unique()
        data_1['label'] = encoder.fit_transform(
                                        data_1['label'])
        class_mapping = {class_name: label for label, class_name in enumerate(unique_labels)}
        classes_df = pd.DataFrame(list(class_mapping.items()), columns=['attack_type', 'attack_label'])
        classes_df.to_csv("csv_files/existing_models/classes.csv")
        labels_encoded = pd.get_dummies(data_1['label']).values
        data_1['protocol_type'] = LabelEncoder().fit_transform(data_1['protocol_type'])
        data_1['service'] = LabelEncoder().fit_transform(data_1['service'])
        data_1['flag'] = LabelEncoder().fit_transform(data_1['flag'])
    return data_1, labels_encoded

    
def data_split(data_1, labels_encoded):
    with st.spinner("Data Spliting"):
        column_names = data_1.keys()
        np_data = data_1.to_numpy()
        X_data = np_data[:,0:41]
        X_data = StandardScaler().fit_transform(X_data)
        req_data = pd.DataFrame(X_data,columns=column_names[:41])
        req_data.to_csv("csv_files/existing_models/data.csv",index=False )
        Y_data = np_data[:,41]
        req_data1 = pd.DataFrame(Y_data)
        req_data1.to_csv("csv_files/existing_models/labels.csv",index=False)
        Y_data = tf.keras.utils.to_categorical(Y_data, 23)
        # Split the preprocessed data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42, stratify=labels_encoded)
    return X_train, X_test, Y_train, Y_test

    

def build_model_1(input_shape, X_train, Y_train, NB_CLASSES):
    #First Model: Shallow Neural Network
    #Creating Model with 
    #Input Layer(512 neurons), 
    #Dropout of 0.05
    #Output Layer(23 neurons)
    
    model = Sequential([
         Dense(512, activation='relu', input_shape=input_shape,),
         Dropout(0.05),
         Dense(NB_CLASSES, activation='softmax')
     ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    VERBOSE = 1
    
    BATCH_SIZE = 64
    
    EPOCHS = 20
    
    VALIDATION_SPLIT = 0.2

    with st.spinner("Model training is in progress"):
        history = model.fit(X_train, Y_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    st.success("Model Training Completed")
    model.save("saved_models/existing_models/DNN 1.h5")
    return (model,history)

def build_model_2(input_shape, X_train, Y_train, NB_CLASSES):
    #Second Model: Deep Neural Network
    #Creating Model with 
    #Input Layer(1024 neurons), 
    #4 Hidden Layers (768, 512, 512, 128 neurons) and 
    #Output Layer(23 neurons)
    #Dropout of 0.01
    
    model = Sequential([
         Dense(1024, activation='relu', input_shape=input_shape),
         Dropout(0.01),
         Dense(768, activation='relu'),
         Dropout(0.01),
         Dense(512, activation='relu'),
         Dropout(0.01),
         Dense(512, activation='relu'),
         Dropout(0.01),
         Dense(128, activation='relu'),
         Dropout(0.01),
         Dense(NB_CLASSES, activation='softmax')
     ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Summarize the model
    model.summary()
    VERBOSE = 1
    
    BATCH_SIZE = 64
    
    EPOCHS = 20
    
    VALIDATION_SPLIT = 0.2
    # Train the model
    with st.spinner("Model training is in progress"):
        history = model.fit(X_train, Y_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    st.success("Model Training Completed")
    model.save("saved_models/existing_models/DNN 2.h5")
    return (model,history)



def build_model_3(input_shape, X_train, Y_train, NB_CLASSES):
    #Third Model: Convolutional Neural Network
    #Creating CNN Model with 
    #Convolutional Layer(32 filters, kernal size 3)
    #one max pooling layer
    #Flatten Layer
    #one dense layer(512 neuron)
    #Output Layer(23 neurons)
    #Dropout of 0.05
    
    num_classes = Y_train.shape[1]
    
    
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    VERBOSE = 1
    
    BATCH_SIZE = 64
    
    EPOCHS = 20
    
    VALIDATION_SPLIT = 0.2
    # Train the model
    with st.spinner("Model training is in progress"):
        history = model.fit(X_train, Y_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    
    st.success("Model Training Completed")
    model.save("saved_models/existing_models/DNN 3.h5")
    return (model,history)



def evaluate_model(model, history, X_test, Y_test):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Multi-Class Classification Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper left')

    st.pyplot(fig)

    # Evaluation against Test Dataset
    st.write("Evaluation against Test Dataset:")
    evaluation_result = model.evaluate(X_test, Y_test)
    st.write("Test Loss:", evaluation_result[0])
    st.write("Test Accuracy:", evaluation_result[1])

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]
    y_pred_classes_bin = tf.keras.utils.to_categorical(y_pred_classes,23)
    # Actual and predicted labels of first 5 classes
    st.write("Actual labels of first 5 classes:", [np.argmax(i) for i in Y_test[:5]])
    st.write("Predicted labels of first 5 classes:", y_pred_classes[:5])

    # Classification report
    st.write("Classification Report:")
    report = classification_report(Y_test, y_pred_classes_bin, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    accuracy = accuracy_score(Y_test, y_pred_classes_bin)
    st.write("Accuracy:", accuracy)

def load_model(selected_model):
    return tf.keras.models.load_model(f"saved_models/existing_models/{selected_model}.h5") 

def load_classes():
    classes = pd.read_csv("csv_files/my_models/classes.csv")
    return classes

def get_class_label(classes, label):
    attack_type = classes.loc[classes['attack_label'] == label, 'attack_type'].values[0]
    return attack_type
    
def deploy_model(X_test, Y_test, model, classes):
    while True:
        random_num = random.randint(0, len(X_test)-1)
        test_row = X_test[random_num].reshape(1, -1)
        st.write(test_row)
        st.write("Random Data:")
        predictions = model.predict(test_row)
        pred_idx = [np.argmax(i) for i in predictions]
        pred_label = get_class_label(classes, pred_idx[0])
        actual_idx = [i for i,k in enumerate(Y_test[random_num]) if k==1]
        actual_label = get_class_label(classes, actual_idx[0])
        st.write(f"Actual Label: {actual_label}")
        st.write(f"Predicted Label: {pred_label}")
        time.sleep(5)

def main():
    st.write("Train And Deploy Existing DNNs")
    data = load_data()
    data, labels_encoded = data_preprocessing(data)
    X_train, X_test, Y_train, Y_test = data_split(data, labels_encoded)
    NB_CLASSES = 23
    input_shape = X_train.shape[1]
    selected_model = st.selectbox(label="Select Existing DNN",options=["DNN 1", "DNN 2", "DNN 3"])
    if st.button("Train Model"):
        build_selected_model = {
            "DNN 1": build_model_1,
            "DNN 2": build_model_2,
            "DNN 3": build_model_3    
        }
        build_fn = build_selected_model.get(selected_model)
        if build_fn:
            if selected_model == "DNN 3":
                # Reshaping the X_train and X_test values to make them fit to CNN
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                model, history = build_fn((input_shape,), X_train, Y_train, NB_CLASSES)
                evaluate_model(model, history, X_test, Y_test)
            else:
                model, history = build_fn((input_shape,), X_train, Y_train, NB_CLASSES)
                evaluate_model(model, history, X_test, Y_test)
    if st.button("Deploy Model"):
        model = load_model(selected_model)            
        classes = load_classes()
        deploy_model(X_test, Y_test, model, classes)


if __name__ == "__main__":
    main()
