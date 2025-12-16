import pandas as pd
import pickle
import os

def test_data_loading():
    df = pd.read_csv('data/dataset.csv')
    assert not df.empty, "Dataset should not be empty"
    assert set(['feature1', 'feature2', 'label']).issubset(df.columns)

def test_model_training_and_shape():
    from sklearn.linear_model import LogisticRegression
    df = pd.read_csv('data/dataset.csv')
    X = df[['feature1', 'feature2']]
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)
    assert hasattr(model, 'coef_'), "Model should be trained and have coefficients"
    assert model.coef_.shape[1] == X.shape[1], "Model coefficients should match feature count"

def test_model_file_exists_and_loads():
    assert os.path.exists('models/model.pkl'), "Model file should exist after training"
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert hasattr(model, 'predict'), "Loaded model should have predict method"
