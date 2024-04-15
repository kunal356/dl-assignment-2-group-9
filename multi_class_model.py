import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,precision_recall_curve, roc_curve, auc, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    #Loading Data from csv file
    data = pd.read_csv("dataset/kddcup99.csv")
    # Removing duplicates from the dataset
    data = data.drop_duplicates()
    print("Data Loading Completed")
    return data

def data_preprocessing(data_1):
    encoder = LabelEncoder()
    unique_labels =data_1['label'].unique()
    data_1['label'] = encoder.fit_transform(
                                    data_1['label'])
    class_mapping = {class_name: label for label, class_name in enumerate(unique_labels)}
    classes_df = pd.DataFrame(list(class_mapping.items()), columns=['attack_type', 'attack_label'])
    classes_df.to_csv("csv_files/my_models/classes.csv")
    labels_encoded = pd.get_dummies(data_1['label']).values
    data_1['protocol_type'] = LabelEncoder().fit_transform(data_1['protocol_type'])
    data_1['service'] = LabelEncoder().fit_transform(data_1['service'])
    data_1['flag'] = LabelEncoder().fit_transform(data_1['flag'])
    print("Data Preprocessing Completed")
    return data_1, labels_encoded

    
def data_split(data_1, labels_encoded):
    column_names = data_1.keys()
    np_data = data_1.to_numpy()
    X_data = np_data[:,0:41]
    X_data = StandardScaler().fit_transform(X_data)
    req_data = pd.DataFrame(X_data,columns=column_names[:41])
    req_data.to_csv("csv_files/my_models/data.csv",index=False )
    Y_data = np_data[:,41]
    req_data1 = pd.DataFrame(Y_data)
    req_data1.to_csv("csv_files/my_models/labels.csv",index=False)
    Y_data = tf.keras.utils.to_categorical(Y_data, 23)
    # Split the preprocessed data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42, stratify=labels_encoded)
    print("Data Spliting Completed")
    return X_train, X_test, Y_train, Y_test

def model_building(X_train, Y_train):
    NB_CLASSES = 23
    #Creating Model with 
    #Input Layer(1024 neurons), 
    #4 Hidden Layers (512, 256, 128, 64 neurons) and 
    #Output Layer(23 neurons)
    #Dropout of 0.1
    
    model = Sequential([
        Dense(1024,
              name="Input-Layer-1",
              activation='relu',
              input_shape=(X_train.shape[1],)
              ),
        Dropout(0.1),
        Dense(512,
              name="Hidden-Layer-1",
              activation='relu',
              ),
        Dropout(0.1),
        Dense(256,
              name="Hidden-Layer-2",
              activation='relu',
              ),
        Dropout(0.1),
        Dense(128,
              name="Hidden-Layer-3",
              activation='relu',
              ),
        Dropout(0.1),
        Dense(64,
              name="Hidden-Layer-4",
              activation='relu'
              ),
        Dropout(0.1),
        Dense(
            NB_CLASSES,
            name="Output-layer",
            activation='softmax'
            )
    
    ])
    
    
    # Compile the model_1
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
    print("Model Training started")
    history = model.fit(X_train, Y_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print("Model Training Completed")
    model.save("saved_models/my_models/model2.h5")
    print("Model saved successfully")
    return (model,history)

def evaluate_model(model, history, X_test, Y_test):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Multi-Class Classification Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    print("Evaluation against Test Dataset:")
    model.evaluate(X_test, Y_test)

    y_pred = model.predict(X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]
    y_pred_classes_bin = tf.keras.utils.to_categorical(y_pred_classes,23)
    print("Actual labels of first 5 classes",y_pred_classes[:5])
    print("Predicting first 5 classes",[np.argmax(i) for i in Y_test[:5]])
    print(classification_report(Y_test, y_pred_classes_bin))
    accuracy = accuracy_score(Y_test, y_pred_classes_bin)
    print("Accuracy:", accuracy)
    

def main():
    data = load_data()
    data, labels_encoded = data_preprocessing(data)
    X_train, X_test, Y_train, Y_test = data_split(data, labels_encoded)
    model, history = model_building(X_train, Y_train)
    evaluate_model(model, history, X_test, Y_test)


if __name__ == "__main__":
    main()
