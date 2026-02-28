import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import preprocess

def train():
    print("Loading and preprocessing data...")
    X, y = preprocess()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    print("\nModel Comparison:")
    print("-" * 35)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name}: {acc * 100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print("-" * 35)
    print(f"\n✅ Best Model: {best_name} ({best_accuracy * 100:.2f}%)")

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train()