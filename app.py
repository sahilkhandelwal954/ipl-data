import streamlit as st
import pickle
import pandas as pd

# Define teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

# Create layout for user inputs
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, format="%.1f")
with col5:
    wickets = st.number_input('Wickets out', min_value=0)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    crr = score / overs if overs > 0 else 0  # Avoid division by zero
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid division by zero

    # Placeholder values for additional features
    value1 = 0  # Replace with actual logic or calculations
    value2 = 0  # Replace with actual logic or calculations

    # Create the input DataFrame
    input_df = pd.DataFrame({
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr],
        'feature1': [value1],  # Ensure these values are meaningful
        'feature2': [value2]
    })

    # Perform prediction
    try:
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display the results
        st.header(f"{batting_team} - {round(win * 100, 2)}%")
        st.header(f"{bowling_team} - {round(loss * 100, 2)}%")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
