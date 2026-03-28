import pandas as pd
import numpy as np
import os

def generate_data(num_samples=2000):
    np.random.seed(42)

    # Demographic features
    age = np.random.randint(18, 75, num_samples)
    income = np.random.normal(65000, 25000, num_samples).astype(int)
    income = np.clip(income, 20000, 200000)

    # Engagement features
    engagement_score = np.random.randint(0, 100, num_samples)

    # Transaction-related features
    transaction_freq = np.random.randint(0, 50, num_samples) # Transactions per year
    avg_transaction_value = np.random.normal(150, 80, num_samples).astype(int)
    avg_transaction_value = np.clip(avg_transaction_value, 10, 1000)
    days_since_last_purchase = np.random.randint(1, 365, num_samples)

    # Synthetic logic for target variable (high_value_purchase)
    # A purchase is more likely if they have high income, high engagement, frequent transactions
    likelihood = (
        (income / 200000) * 0.3 + 
        (engagement_score / 100) * 0.4 + 
        (transaction_freq / 50) * 0.2 +
        (avg_transaction_value / 1000) * 0.2 -
        (days_since_last_purchase / 365) * 0.3
    )
    
    # Add noise
    likelihood += np.random.normal(0, 0.1, num_samples)
    
    # Threshold for class 1
    threshold = np.percentile(likelihood, 70) # Top 30% are high-value
    high_value_purchase = (likelihood > threshold).astype(int)

    # Compile dataset
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'engagement_score': engagement_score,
        'transaction_freq': transaction_freq,
        'avg_transaction_value': avg_transaction_value,
        'days_since_last_purchase': days_since_last_purchase,
        'high_value_purchase': high_value_purchase
    })

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_data.csv'), index=False)
    print("Generated customer_data.csv with requested features.")

if __name__ == "__main__":
    generate_data()
