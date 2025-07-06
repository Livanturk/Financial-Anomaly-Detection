import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import Preprocess


p = Preprocess(csv_path = 'data/transaction_dataset.csv')
X_train, X_test, y_train, y_test = p.get_processed_data()

print("Train Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
print("Train Labels Shape:", y_train.shape)