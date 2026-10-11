import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_parquet("Training.parquet")

# Convert status to lowercase
data['status'] = data['status'].str.lower()

# Map values properly
data['status'] = data['status'].map({
    'legitimate': 0,
    'phishing': 1
})

# Remove missing values
data = data.dropna()

# Separate features and target
X = data.drop(['url', 'status'], axis=1)
y = data['status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)