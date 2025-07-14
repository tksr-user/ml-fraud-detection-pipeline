# transform.py - Auto-generated
def create_features(df):
    # Rename 'Class' to 'is_fraud' to align with rest of pipeline
    df = df.rename(columns={"Class": "is_fraud"})

    # Create a log-transformed feature from Amount
    df["amount_log"] = df["Amount"].apply(lambda x: 0 if x <= 0 else round(x**0.5, 2))

    # Drop non-useful columns (e.g., Time) if you want
    df = df.drop(columns=["Time"], errors='ignore')

    return df