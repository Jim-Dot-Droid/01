import streamlit as st
import pandas as pd
import numpy as np
import os

# File paths
HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
FLAT_FILE = "sol_balance.txt"
FIXED_FILE = "fixed_balance.txt"
MARTI_FILE = "martingale_balance.txt"
MARTI_BET_FILE = "martingale_bet.txt"
MANUAL_FILE = "manual_balance.txt"

# Constants
INITIAL_BALANCE = 0.1
FLAT_BET = 0.01
FIXED_BET = 0.02
MARTI_BASE_BET = 0.01
WINDOW = 20
MIN_UNDERS_FOR_ABOVE = 14

# --- Data helpers ---
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if 'multiplier' in df.columns:
        return df['multiplier'].tolist()
    return df.iloc[:,0].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        return df['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction', 'actual', 'correct'])

def save_result(prediction, actual):
    correct = ((prediction == "Above") and actual > 2.0) or ((prediction == "Under") and actual <= 2.0)
    result_df = load_results()
    result_df.loc[len(result_df)] = [prediction, actual, correct]
    result_df.to_csv(RESULTS_FILE, index=False)
    update_flat_balance(prediction, actual)
    if prediction == "Above":
        update_fixed_balance(actual)
        update_martingale_balance(actual)

# --- Balance handlers ---
def get_flat_balance():
    if os.path.exists(FLAT_FILE):
        with open(FLAT_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def get_fixed_balance():
    if os.path.exists(FIXED_FILE):
        with open(FIXED_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def get_martingale_balance():
    if os.path.exists(MARTI_FILE):
        with open(MARTI_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def get_martingale_bet():
    if os.path.exists(MARTI_BET_FILE):
        with open(MARTI_BET_FILE, "r") as f:
            return float(f.read())
    return MARTI_BASE_BET

def get_manual_balance():
    if os.path.exists(MANUAL_FILE):
        with open(MANUAL_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def update_flat_balance(prediction, actual):
    balance = get_flat_balance()
    if prediction == "Above":
        balance += FLAT_BET if actual > 2.0 else -FLAT_BET
        with open(FLAT_FILE, "w") as f:
            f.write(str(balance))

def update_fixed_balance(actual):
    balance = get_fixed_balance()
    balance += FIXED_BET if actual > 2.0 else -FIXED_BET
    with open(FIXED_FILE, "w") as f:
        f.write(str(balance))

def update_martingale_balance(actual):
    balance = get_martingale_balance()
    bet = get_martingale_bet()

    if actual > 2.0:  # Win
        balance += bet
        bet = MARTI_BASE_BET  # Reset to base
    else:  # Loss
        balance -= bet
        bet *= 2  # Double bet after loss

    with open(MARTI_FILE, "w") as f:
        f.write(str(balance))
    with open(MARTI_BET_FILE, "w") as f:
        f.write(str(bet))

def update_manual_balance(prediction, actual, bet):
    balance = get_manual_balance()
    if (prediction == "Above" and actual > 2.0) or (prediction == "Under" and actual <= 2.0):
        balance += bet
    else:
        balance -= bet
    with open(MANUAL_FILE, "w") as f:
        f.write(str(balance))

def reset_balance():
    for f in [FLAT_FILE, FIXED_FILE, MARTI_FILE, MARTI_BET_FILE, MANUAL_FILE, RESULTS_FILE]:
        if os.path.exists(f):
            os.remove(f)

# --- Logic ---
def normalize_input(value):
    return value / 100 if value > 10 else value

def predict_from_unders(data, threshold=2.0, window=WINDOW, min_unders_for_above=MIN_UNDERS_FOR_ABOVE):
    if len(data) < window:
        return None, None
    recent = np.array(data[-window:])
    under_count = int(np.sum(recent < threshold))
    if under_count >= min_unders_for_above:
        return "Above", under_count
    else:
        return "Under", under_count

# --- Streamlit App ---
def main():
    st.title("Crash Predictor — Under Count Strategy (Last 20) with Martingale & Manual Bets")

    st.sidebar.header("Settings")
    min_unders = st.sidebar.slider("Min unders in last 20 to trigger 'Above' prediction",
                                   min_value=10, max_value=20, value=MIN_UNDERS_FOR_ABOVE, step=1)

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload multipliers CSV", type=["csv"])
        if uploaded:
            st.session_state.history = load_csv(uploaded)
            save_history(st.session_state.history)
            st.success(f"Loaded {len(st.session_state.history)} multipliers.")

    with col2:
        if st.button("Reset all (balances & results)"):
            st.session_state.history = []
            save_history([])
            reset_balance()
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("Reset done.")

    st.subheader("Manual input")
    new_val = st.text_input("Enter multiplier (e.g., 1.87 or 187)")
    if st.button("Add multiplier"):
        try:
            val = float(new_val)
            val = normalize_input(val)
            if "last_prediction" in st.session_state:
                save_result(st.session_state.last_prediction, val)
                del st.session_state.last_prediction
            st.session_state.history.append(val)
            save_history(st.session_state.history)
            st.success(f"Added {val}x")
        except Exception as e:
            st.error("Invalid input")

    if st.session_state.history:
        data = st.session_state.history
        st.write(f"History length: {len(data)}")

        st.subheader("Prediction from Under Count (last 20)")
        prediction, under_count = predict_from_unders(data, min_unders_for_above=min_unders)
        if prediction:
            st.session_state.last_prediction = prediction
            st.write(f"Prediction: **{prediction}** (Under count in last 20 = {under_count})")
        else:
            st.write(f"Not enough data yet (need at least {WINDOW} rounds).")

    else:
        st.write("No history yet. Upload CSV or add multipliers manually.")

    st.subheader("Accuracy Tracker")
    results_df = load_results()
    if not results_df.empty:
        total = len(results_df)
        correct = int(results_df['correct'].sum())
        acc = correct / total if total else 0
        st.metric("Total Predictions", total)
        st.metric("Correct Predictions", correct)
        st.metric("Accuracy Rate", f"{acc:.1%}")
        st.dataframe(results_df[::-1].reset_index(drop=True))
    else:
        st.write("No verified predictions yet.")

    st.subheader("💰 SOL Balance Tracker")
    st.metric("Flat Bet Balance (0.01 SOL per 'Above')", f"{get_flat_balance():.4f} SOL")
    st.metric("Fixed Bet Balance (0.02 SOL only when 'Above')", f"{get_fixed_balance():.4f} SOL")
    st.metric("Martingale Balance", f"{get_martingale_balance():.4f} SOL")
    st.metric("Current Martingale Bet", f"{get_martingale_bet():.4f} SOL")
    st.caption("Martingale doubles bet after loss, resets to base after win. Bets only when 'Above' is predicted and result is recorded.")

    # --- Manual Bet Section (Full Control) ---
    st.subheader("🎲 Manual Bet")
    if st.session_state.history:
        round_index = st.number_input(
            f"Choose round to bet on (1 = first multiplier, {len(st.session_state.history)} = last)",
            min_value=1, max_value=len(st.session_state.history), value=len(st.session_state.history), step=1
        ) - 1

        manual_pred = st.selectbox("Choose prediction", ["Above", "Under"])
        manual_bet = st.number_input("Bet amount (SOL)", min_value=0.0, value=0.01, step=0.01)

        if st.button("Place Manual Bet"):
            actual = st.session_state.history[round_index]
            update_manual_balance(manual_pred, actual, manual_bet)
            st.success(f"Manual bet of {manual_bet} SOL on {manual_pred} recorded (multiplier = {actual} at round {round_index + 1})")

        st.metric("Manual Balance", f"{get_manual_balance():.4f} SOL")
    else:
        st.write("No history yet. Add multipliers manually or upload CSV.")

if __name__ == "__main__":
    main()
