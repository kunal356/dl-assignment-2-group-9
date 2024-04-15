
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix,precision_recall_curve, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data():
    #Loading Data from csv file
    data = pd.read_csv("dataset/kddcup99.csv")
    print("Data Loading Completed")
    return data


def data_preprocessing(df):
    X = df.drop('label', axis=1)
    y = df['label']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    X_processed = preprocessor.fit_transform(X)
    y_masked = y.apply(lambda x: 1 if x == "normal" else 0)
    print("Data Preprocessing Completed")    
    return X_processed, y_masked
   
    
def data_split(X_processed, y_masked):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_masked, test_size=0.2, random_state=42)
    ano_data = pd.DataFrame(X_test)
    ano_data.to_csv("csv_files/my_models/ano_data.csv", index=False )
    ano_classes = pd.DataFrame(y_test)
    ano_classes.to_csv("csv_files/my_models/ano_classes.csv", index=False )
    print("Data Spliting Completed")
    return X_train, X_test, y_train, y_test

def model_building(X_train, X_test):
    # Anamoly detection using Autoencoder
    #Creating Model with Input Layer(25 neurons), 
    #Middle(Bottleneck) Layer which compresses to 3 neuron
    #Hidden Layer expands it back 25 neurons
    #Output Layer expands back to input shape(118 neurons)
    #Dropout of 0.5
    model = Sequential()
    model.add(Dense(25, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(3, activation='relu')) 
    model.add(Dense(25, activation='relu'))
    model.add(Dense(X_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    model.summary()
    print("Model Training started")
    history = model.fit(X_train, X_train,
                              epochs=50,
                              batch_size=256,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              verbose=1)
    print("Model Training Completed")
    model.save("saved_models/my_models/model3.h5")
    print("Model saved successfully")
    return (model,history)

def evaluate_model(model, history, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                             'true_class': y_test})
    
    y_true = error_df['true_class'].values
    
    threshold = np.percentile(error_df['reconstruction_error'], 95)
    print("Threshold: ",threshold)
    y_pred = [1 if e > threshold else 0 for e in error_df['reconstruction_error'].values]

    # Training Loss vs Validation Loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()

    normal_error_df = error_df[error_df['true_class'] == 1]
    anomaly_error_df = error_df[error_df['true_class'] == 0]
    print("normal\n",normal_error_df)
    print("anomaly\n",anomaly_error_df)

    # Histogram of reconstruction error
    plt.figure(figsize=(16, 9))
    plt.hist(normal_error_df['reconstruction_error'],  bins=50, alpha=0.7, label='Normal')
    plt.hist(anomaly_error_df['reconstruction_error'], range=[0, 10], bins=50, alpha=0.7, label='Anomaly')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # ROC curve
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    #ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()


    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted Attack'],
                yticklabels=['True Normal', 'True Attack'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_true, y_pred))

def main():
    data = load_data()
    X_processed, y_masked = data_preprocessing(data)
    X_train, X_test, y_train, y_test = data_split(X_processed, y_masked)
    model, history = model_building(X_train, X_test)
    evaluate_model(model, history, X_test, y_test)


if __name__ == "__main__":
    main()










