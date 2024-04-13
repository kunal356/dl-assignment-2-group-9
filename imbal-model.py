
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from sklearn.metrics import classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


def load_data():
    #Loading Data from csv file
    data = pd.read_csv("dataset/kddcup99.csv")
    # Removing duplicates from the dataset
    data = data.drop_duplicates()
    print("\nLoaded Data :\n------------------------------------")
    print(data.head())
    print(data.shape)
    data.shape
    data.label.value_counts()
    return data



# Assuming data is your DataFrame containing class labels
def oversampling(data):
    # Count the occurrences of each class
    class_counts = defaultdict(int)
    for _, row in data.iterrows():
        class_counts[row['label']] += 1
    
    # Find the majority and minority class labels
    majority_class = max(class_counts, key=class_counts.get)
    minority_classes = [cls for cls in class_counts if cls != majority_class]
    
    # Determine the desired count for each minority class
    desired_count = class_counts[majority_class]
    
    # Oversample the minority classes
    oversampled_data = pd.DataFrame()  # Initialize an empty DataFrame for oversampled data
    for cls in minority_classes:
        class_data = data[data['label'] == cls]
        if len(class_data) < desired_count:  # Add condition to avoid doubling majority class
            oversampled_data = pd.concat([oversampled_data, class_data.sample(n=desired_count - len(class_data), replace=True)])
        else:
            oversampled_data = pd.concat([oversampled_data, class_data.sample(n=desired_count, replace=True)])
    
    # Combine oversampled data with original data
    data_1 = pd.concat([data, oversampled_data])
    print(data_1.shape)
    
    print(data_1['label'].value_counts())
    
    print(data_1.head())
    return data_1

def data_preprocessing(data_1):
    encoder = LabelEncoder()
    unique_labels =data_1['label'].unique()
    data_1['label'] = encoder.fit_transform(
                                    data_1['label'])
    class_mapping = {class_name: label for label, class_name in enumerate(unique_labels)}
    classes_df = pd.DataFrame(list(class_mapping.items()), columns=['attack_type', 'attack_label'])
    classes_df.to_csv("csv_files/my_models/classes.csv")
    labels_encoded = pd.get_dummies(data_1['label']).values
    print("labels encoded", labels_encoded)
    
    data_1['protocol_type'] = LabelEncoder().fit_transform(data_1['protocol_type'])
    data_1['service'] = LabelEncoder().fit_transform(data_1['service'])
    data_1['flag'] = LabelEncoder().fit_transform(data_1['flag'])
    print("Dataset after encoding:")
    print(data_1.head())
    print(data_1.dtypes)
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
    print(X_data.shape)
    print(Y_data.shape)
    print(labels_encoded.shape)
    # Split the preprocessed data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42, stratify=labels_encoded)
    
    # Check the shapes of the resulting splits to confirm their sizes
    return X_train, X_test, Y_train, Y_test


def model_building(X_train, Y_train):
    NB_CLASSES = 23
    # Define the model architecture
    model_2 = Sequential([
        Dense(128,
              name="Hidden-Layer-1",
              activation='relu',
              input_shape=(X_train.shape[1],)
              ),
        Dropout(0.5),
        Dense(128,
              name="Hidden-Layer-2",
              activation='relu'
              ),
        Dropout(0.5),
        Dense(
            NB_CLASSES,
            name="Output-layer",
            activation='softmax'
            )
    
    ])
    
    
    # Compile the model_2
    model_2.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Summarize the model
    model_2.summary()
    VERBOSE = 1
    
    BATCH_SIZE = 64
    
    EPOCHS = 20
    
    VALIDATION_SPLIT = 0.2
    # Train the model
    history = model_2.fit(X_train, Y_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    model_2.save("saved_models/my_models/model1.h5")
    return (model_2,history)


# pd.DataFrame(history.history)["accuracy"].plot(figsize=(8,5))
# plt.title("Accuracy improvements with Epoch")
# plt.show()
def evaluate_model(model_2, history, X_test, Y_test):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Imbalanced Dataset Classification Model Accuracy')
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    print("Evaluation against Test Dataset:")
    model_2.evaluate(X_test, Y_test)

    y_pred = model_2.predict(X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]
    y_pred_classes_bin = tf.keras.utils.to_categorical(y_pred_classes,23)
    print("Actual labels of first 5 classes",y_pred_classes[:5])
    print("Predicting first 5 classes",[np.argmax(i) for i in Y_test[:5]])
    print(classification_report(Y_test, y_pred_classes_bin))
    
def main():
    data = load_data()
    data_1 = oversampling(data)
    data_1, labels_encoded = data_preprocessing(data_1)
    X_train, X_test, Y_train, Y_test = data_split(data_1, labels_encoded)
    model_2, history = model_building(X_train, Y_train)
    evaluate_model(model_2, history, X_test, Y_test)


if __name__ == "__main__":
    main()