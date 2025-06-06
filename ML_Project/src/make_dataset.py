import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """Загрузка исходных данных"""
    return pd.read_csv(data_path)

def preprocess_data(df):
    """Основная предобработка данных"""
    df['teaching_method'] = df['teaching_method'].map({'Standard': 0, 'Experimental': 1})
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['lunch'] = df['lunch'].map({'Does not qualify': 0, 'Qualifies for reduced/free lunch': 1})
    
    df = pd.get_dummies(df, columns=['school_setting', 'school_type'], drop_first=True)
    
    df = df.drop(['school', 'classroom', 'student_id', 'n_student'], axis=1)
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Разделение данных на train/test"""
    X = df.drop('posttest', axis=1)
    y = df['posttest']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(X_train, X_test, y_train, y_test, output_dir):
    """Сохранение обработанных данных"""
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    raw_data_path = "data/raw/test_scores.csv"
    output_dir = "data/processed"
    
    df = load_data(raw_data_path)
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    save_data(X_train, X_test, y_train, y_test, output_dir)
    import os
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)