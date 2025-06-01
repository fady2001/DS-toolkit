import lightgbm as lgb

from utils import rmsle_score, split_time_series_data


def train_model_for_each_family(train_df):
    """
    Train a LightGBM model for each family in the training dataset.
    
    Parameters:
    - train_df: DataFrame containing the training data with 'family' and 'sales' columns.
    - train: DataFrame containing the training data with features and 'family' column.
    
    Returns:
    - models: Dictionary containing trained models for each family.
    """
        
    # train a LightGBM model for each family
    families = train_df['family'].unique()
    models = {}
    for family in families:
        print(f"Training model for family: {family}")
        
        # Filter the training data for the current family
        family_data = train_df[train_df['family'] == family]
        
        # Split the data into features and target variable
        X = family_data.drop(columns=['sales', 'family'])
        y = family_data['sales']
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = split_time_series_data(X, y)
        
        # Define LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Create a LightGBM dataset
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        # Train the model
        model = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=1000, early_stopping_rounds=50)
        
        # Make predictions on the validation set
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        # Calculate RMSLE score
        score = rmsle_score(y_val, y_pred)
        
        print(f"RMSLE score for family {family}: {score:.4f}")
        
        # Store the best model for the family
        models[family] = model
    return models