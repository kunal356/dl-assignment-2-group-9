import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_data():
    # Load the dataset
    df = pd.read_csv('dataset/kddcup99.csv')
    print("Data Loading Completed")
    return df

def preprocess_data(df): 
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]  
    
    # Identifying categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # Creating a transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    X_processed = preprocessor.fit_transform(X)
    
    # Since LSTM expects a 3D input, we reshape X to have three dimensions [samples, timesteps, features]
    # Here, we consider each sample as a sequence of one timestep
    X_reshaped = np.reshape(X_processed, (X_processed.shape[0], 1, X_processed.shape[1]))
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
    print("Data preprocessing and spliting completed")
    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    # Creating LSTM Model
    # LSTM layer with 50 neuron
    # Dense Layer with 23 neuron
    model = Sequential([
        LSTM(50, input_shape=(1, X_train.shape[2])),  # Input shape: [timesteps, features]
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train the model
    print("Model Building started")
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    print("Model Building completed")
    return model
    
def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    #Plotting confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    disp.ax_.set_title('Confusion Matrix', fontsize=22)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.15, right=0.95)
    plt.show()
    report = classification_report(y_true, y_pred)
    
    print(report)
def main():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = build_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
if __name__ == "__main__":
    main()
