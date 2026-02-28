import joblib
import numpy as np

def load_artifacts():
    model = joblib.load("models/model.pkl")
    le_team = joblib.load("models/le_team.pkl")
    le_venue = joblib.load("models/le_venue.pkl")
    return model, le_team, le_venue

def predict_winner(team1, team2, venue, toss_winner, toss_decision):
    model, le_team, le_venue = load_artifacts()

    # Encode inputs
    team1_enc = le_team.transform([team1])[0]
    team2_enc = le_team.transform([team2])[0]
    venue_enc = le_venue.transform([venue])[0]
    toss_decision_bat = 1 if toss_decision == "bat" else 0
    team1_won_toss = 1 if toss_winner == team1 else 0

    features = np.array([[team1_enc, team2_enc, venue_enc, toss_decision_bat, team1_won_toss]])

    prob = model.predict_proba(features)[0]  # [prob_team2_wins, prob_team1_wins]

    return {
        team1: round(prob[1] * 100, 2),
        team2: round(prob[0] * 100, 2)
    }

def get_all_teams():
    _, le_team, _ = load_artifacts()
    return list(le_team.classes_)

def get_all_venues():
    _, _, le_venue = load_artifacts()
    return list(le_venue.classes_)