
import chardet
import pandas as pd #universal usage

#loading the data

# print(data) success

def display_file_encoding():
    with open(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv", 'rb') as file:
        result=chardet.detect(file.read())
        ans=result['encoding']
    return ans


#function for random forest classifier
def pred_rfc():
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

    data=pd.read_csv(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv")

    #defining features and target
    features=[ 'verified', 'friends_count', 'followers_count', 'gender:confidence']
    target='status'

    x=data[features]
    y=data[target]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=42,stratify=y)

    #random forest model
    model=RandomForestClassifier(random_state=42)
    model.fit(x_train,y_train)

    #predictions
    y_pred=model.predict(x_test)

    #model evaluation
    accuracy=accuracy_score(y_test,y_pred)*100
    class_report=classification_report(y_test,y_pred,output_dict=True)
    report_df=pd.DataFrame(class_report).transpose()
    report_html=report_df.to_html(classes="table table-bordered table-striped",float_format="%.2f")
    c_matrix1=confusion_matrix(y_test,y_pred)
    c_matrix = c_matrix1.tolist()
    return accuracy,report_html, c_matrix

def pred_nb():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

    features=['verified','friends_count','followers_count','gender:confidence']
    target='status'
    data=pd.read_csv(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv")

    x=data[features]
    y=data[target]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
    #naive bayes classifier model
    nb_model=GaussianNB()
    nb_model.fit(x_train,y_train)

    #prediction
    ypred=nb_model.predict(x_test)

    #model evaluation
    accuracy=accuracy_score(y_test,ypred)*100
    class_report=classification_report(y_test,ypred,output_dict=True)
    report_df=pd.DataFrame(class_report).transpose()
    report_html=report_df.to_html(classes="table table-bordered table-striped",float_format="%.2f")
    c_matrix1=confusion_matrix(y_test,ypred)
    c_matrix = c_matrix1.tolist()

    return accuracy,report_html, c_matrix

def pred_linear_regression():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error,confusion_matrix,classification_report
    features=['verified','friends_count','followers_count','gender:confidence']
    target='status'
    data=pd.read_csv(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv")

    x=data[features]
    y=data[target]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

    model=LinearRegression()
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    ypred = (y_pred >= 0.5).astype(int)


    mse=mean_squared_error(y_test,ypred)
    c_matrix1=confusion_matrix(y_test,ypred)
    c_matrix=c_matrix1.tolist()
    return mse,c_matrix

def pred_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

    features=['verified','friends_count','followers_count','gender:confidence']
    target='status'
    data=pd.read_csv(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv")

    x=data[features]
    y=data[target]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
    #naive bayes classifier model
    model=LogisticRegression(random_state=46,max_iter=1000)
    model.fit(x_train,y_train)

    #prediction
    ypred=model.predict(x_test)

    #model evaluation
    accuracy=accuracy_score(y_test,ypred)*100
    class_report=classification_report(y_test,ypred,output_dict=True)
    report_df=pd.DataFrame(class_report).transpose()
    report_html=report_df.to_html(classes="table table-bordered table-striped",float_format="%.2f")
    c_matrix1=confusion_matrix(y_test,ypred)
    c_matrix = c_matrix1.tolist()

    return accuracy,report_html, c_matrix

def lstm_approach():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from keras import models, layers, optimizers
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import pandas as pd
    import numpy as np
    import tensorflow as tf

    # Feature and target selection
    features = ['verified', 'friends_count', 'followers_count', 'gender:confidence']
    target = 'status'
    data = pd.read_csv(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv")

    # Preprocessing
    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM (samples, timesteps, features)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split the data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    md = models.Sequential()

    # Add LSTM layer
    md.add(layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Add output layer for binary classification
    md.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    md.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = md.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    accuracy, loss = md.evaluate(X_test, y_test)
    tf.get_logger().setLevel('ERROR')  # Suppress warning logs

    # Predict on test set
    y_pred_continuous = md.predict(X_test)
    y_pred_binary = (y_pred_continuous >= 0.5).astype(int)

    # Example prediction
    sample_input = X_test[0].reshape(1, 1, len(features))  # Reshape for a single sample
    sample_prediction = md.predict(sample_input)
    result = "Real" if sample_prediction[0][0] > 0.5 else "Fake"

    # Confusion matrix and classification report
    c_matrix = confusion_matrix(y_test, y_pred_binary)
    c_report = classification_report(y_test, y_pred_binary, output_dict=True)
    
    # Convert classification report to HTML
    report_df = pd.DataFrame(c_report).transpose()
    report_html = report_df.to_html(classes="table table-bordered table-striped", float_format="%.2f")

    # Accuracy score
    acc = accuracy_score(y_test, y_pred_binary)

    return loss, accuracy, features, result, c_matrix, report_html

loss, accuracy, features, result, c_matrix, report_html=lstm_approach()


def lstm_id(id_num):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from keras import models, layers, optimizers
    import tensorflow as tf

    data = pd.read_csv(r"F:\python\projects\fake_social_media_detection_django\fppd\myapp\twitter_dataset.csv")

    # Example Data Loading (Replace 'path_to_your_data.csv' with your actual data path)
    tf.get_logger().setLevel('ERROR')  # Suppress warning logs
    # Features and target column
    features = ['verified', 'followers_count', 'friends_count', 'gender:confidence', 'profile_yn:confidence']
    X = data[features].values
    y = data['status'].values  # Assuming status is the target column
    target='status'
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Reshape for LSTM (samples, timesteps, features)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build the LSTM model
    md = models.Sequential()

# Add LSTM layer
    md.add(layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))

# Add output layer for binary classification
    md.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
    md.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model without displaying epoch details
    history = md.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Evaluate the model
    loss, accuracy = md.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}')

# --- Prediction for a Specific ID ---
# Define the ID you want to query
    specific_id = id_num  # Replace with the specific ID number

# Find the row for the specific ID
    if 'id' in data.columns:  # Ensure an 'ID' column exists in the dataset
        specific_row = data[data['id'] == specific_id]
        if specific_row.empty:
            pass
        else:
            specific_features = specific_row[features].values

        # Scale and reshape the data for prediction
            specific_features_scaled = scaler.transform(specific_features)
            specific_features_reshaped = specific_features_scaled.reshape((specific_features_scaled.shape[0], 1, specific_features_scaled.shape[1]))

        # Make a prediction
            prediction = md.predict(specific_features_reshaped)
            if prediction[0][0]>0.50:
                result='Real'
            else:
                result="Fake"
    else:
            result='No data found for {specific_id}'

    return result

