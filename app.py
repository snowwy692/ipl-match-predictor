import streamlit as st
from src.predict import predict_winner, get_all_teams, get_all_venues

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered")

st.title("🏏 IPL Match Outcome Predictor")
st.markdown("Predict the winner of an IPL match based on teams, venue and toss details.")
st.divider()

teams = get_all_teams()
venues = get_all_venues()

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams, index=0)

with col2:
    team2 = st.selectbox("Team 2", teams, index=1)

venue = st.selectbox("Venue", venues)

col3, col4 = st.columns(2)

with col3:
    toss_winner = st.selectbox("Toss Winner", [team1, team2])

with col4:
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

st.divider()

if st.button("🔮 Predict Winner", use_container_width=True):
    if team1 == team2:
        st.error("Please select two different teams!")
    else:
        result = predict_winner(team1, team2, venue, toss_winner, toss_decision)

        st.subheader("Win Probability")
        col5, col6 = st.columns(2)

        with col5:
            st.metric(label=team1, value=f"{result[team1]}%")
            st.progress(int(result[team1]))

        with col6:
            st.metric(label=team2, value=f"{result[team2]}%")
            st.progress(int(result[team2]))

        winner = max(result, key=result.get)
        st.success(f"🏆 Predicted Winner: **{winner}**")
