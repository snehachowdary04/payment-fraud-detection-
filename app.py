from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def index():
    # Load the dataset
    file_path = 'Fraud Detection Dataset.csv'  # CSV file should be in the same folder
    dataset = pd.read_csv(file_path, index_col=0)

    # Handling missing values
    if dataset.isnull().values.any():
        dataset = dataset.dropna()  # Option 1: Drop missing rows
        # Alternatively, you can fill missing values instead of dropping:
        # dataset = dataset.fillna(0)

    # Feature and Target separation
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Label Encoding for categorical data
    categorical_features = x.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        le = LabelEncoder()
        x[feature] = le.fit_transform(x[feature])

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Store model accuracies
    models = {}

    # Logistic Regression
    LR_model = LogisticRegression(random_state=0)
    LR_model.fit(x_train, y_train)
    y_pred = LR_model.predict(x_test)
    models['Logistic Regression'] = accuracy_score(y_test, y_pred) * 100

    # K-Nearest Neighbors
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(x_train, y_train)
    y_pred = KNN_model.predict(x_test)
    models['K-Nearest Neighbors'] = accuracy_score(y_test, y_pred) * 100

    # Support Vector Machine
    SVM_model = SVC(kernel='linear', random_state=0)
    SVM_model.fit(x_train, y_train)
    y_pred = SVM_model.predict(x_test)
    models['Support Vector Machine'] = accuracy_score(y_test, y_pred) * 100

    # Naive Bayes
    NB_model = GaussianNB()
    NB_model.fit(x_train, y_train)
    y_pred = NB_model.predict(x_test)
    models['Naive Bayes'] = accuracy_score(y_test, y_pred) * 100

    # Decision Tree
    DT_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    DT_model.fit(x_train, y_train)
    y_pred = DT_model.predict(x_test)
    models['Decision Tree'] = accuracy_score(y_test, y_pred) * 100

    # Random Forest
    RF_model = RandomForestClassifier(random_state=0)
    RF_model.fit(x_train, y_train)
    y_pred = RF_model.predict(x_test)
    models['Random Forest'] = accuracy_score(y_test, y_pred) * 100

    # Convolutional Neural Network (Simple MLP here, because tabular data)
    CNN_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=x_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    CNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    CNN_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)

    y_pred = CNN_model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    models['Convolutional Neural Network'] = accuracy_score(y_test, y_pred) * 100

    # Prepare data for frontend
    result = [{"algorithm": name, "accuracy": f"{acc:.2f}"} for name, acc in sorted(models.items(), key=lambda item: item[1], reverse=True)]

    return render_template('index.html', results=result)

if __name__ == '__main__':
    app.run(debug=True)
