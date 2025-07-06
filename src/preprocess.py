import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Preprocess:
    """
    A class for preprocessing the transaction dataset for anomaly detection.
    Includes safe feature engineering,
    """

    def __init__(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the class and performs a stratified train-test split.
        """
        self.raw_df = pd.read_csv(csv_path)
        self.df = self.raw_df.copy()

        self.df = self.df[self.df['type'].isin(['TRANSFER', 'CASH_OUT'])]

        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['isFraud']
        )

    def _engineer_features(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Balance error features
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

        # One-hot encoding
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

        # Log transform
        df['LogAmount'] = np.log1p(df['amount'])

        # Time features
        df['day'] = df['step'] // 24
        df['hour'] = df['step'] % 24

        # Balance ratios
        df['orig_balance_raio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1)
        df['dest_balance_raio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1)

        # Aggregations from ref_df
        orig_stats = ref_df.groupby('nameOrig').agg({
            'amount': ['count', 'sum'],
            'oldbalanceOrg': 'mean'
        }).reset_index()
        orig_stats.columns = ['nameOrig', 'orig_txn_count', 'orig_total_sent', 'orig_avg_balance']
        df = df.merge(orig_stats, on='nameOrig', how='left')

        dest_stats = ref_df.groupby('nameDest').agg({
            'amount': ['count', 'sum'],
            'oldbalanceDest': 'mean'
        }).reset_index()
        dest_stats.columns = ['nameDest', 'dest_txn_count', 'dest_total_received', 'dest_avg_balance']
        df = df.merge(dest_stats, on='nameDest', how='left')

        df['isOrigRare'] = (df['orig_txn_count'] <= 1).astype(int)
        df['isDestRare'] = (df['dest_txn_count'] <= 1).astype(int)

        threshold = ref_df['amount'].quantile(0.999)
        df['amount_outlier'] = (df['amount'] > threshold).astype(int)

        df.fillna({
            'orig_txn_count': 0,
            'orig_total_sent': 0,
            'orig_avg_balance': 0,
            'dest_txn_count': 0,
            'dest_total_received': 0,
            'dest_avg_balance': 0,
        }, inplace=True)

        return df

    def get_processed_data(self):
        train = self._engineer_features(self.train_df, self.train_df)
        test = self._engineer_features(self.test_df, self.train_df)

        X_train = train.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
        y_train = train['isFraud']
        X_test = test.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
        y_test = test['isFraud']

        return X_train, X_test, y_train, y_test
