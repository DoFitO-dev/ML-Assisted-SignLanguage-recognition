import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = []
labels = []

for i, xi in enumerate(data_dict['data']):
    if len(xi) == len(data_dict['data'][0]):
        data.append(xi)
        labels.append(data_dict['labels'][i])

data = np.array(data)
labels = np.asarray(labels)

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForest model with hyperparameter tuning if needed
model = RandomForestClassifier(n_estimators=100)  # Example: using 100 trees
model.fit(x_train, y_train)

# Make predictions and evaluate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
