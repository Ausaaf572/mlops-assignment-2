import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('data/dataset.csv')
X = df[['feature1', 'feature2']]
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open('models/model.pkl', 'wb'):
    pickle.dump(model, open('models/model.pkl', 'wb'))
print('Model trained and saved to models/model.pkl')
