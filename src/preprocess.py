import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(path="data/matches.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Drop rows where winner is missing (no result / tied matches)
    df = df.dropna(subset=["winner"])

    # Keep only relevant columns
    cols = ["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]
    df = df[cols]

    return df

def engineer_features(df):
    # Did team1 win? This is our target
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    # Did the toss winner choose to bat?
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)

    # Did team1 win the toss?
    df["team1_won_toss"] = (df["toss_winner"] == df["team1"]).astype(int)

    return df

def encode_features(df):
    le_team = LabelEncoder()
    le_venue = LabelEncoder()

    # Fit on all team names (team1 + team2 combined so encoder knows all teams)
    all_teams = pd.concat([df["team1"], df["team2"]])
    le_team.fit(all_teams)

    df["team1_enc"] = le_team.transform(df["team1"])
    df["team2_enc"] = le_team.transform(df["team2"])
    df["venue_enc"] = le_venue.fit_transform(df["venue"])

    # Save encoders for use in the app later
    os.makedirs("models", exist_ok=True)
    joblib.dump(le_team, "models/le_team.pkl")
    joblib.dump(le_venue, "models/le_venue.pkl")

    return df

def get_features_and_target(df):
    features = ["team1_enc", "team2_enc", "venue_enc", "toss_decision_bat", "team1_won_toss"]
    target = "team1_won"
    return df[features], df[target]

def preprocess(path="data/matches.csv"):
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)
    X, y = get_features_and_target(df)
    return X, y