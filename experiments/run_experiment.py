# run_experiment.py - Auto-generated
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def run(model, df):
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    acc = accuracy_score(y_test, model.predict(X_test))
    return {"accuracy": round(acc, 4)}