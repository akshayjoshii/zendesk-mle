import streamlit as st
import requests
import pandas as pd

# Configuration
API_BASE_URL = "http://127.0.0.1:9000"
INTENT_ENDPOINT = f"{API_BASE_URL}/intent"
READY_ENDPOINT = f"{API_BASE_URL}/ready"

# Page Configuration
st.set_page_config(
    page_title="AI Intent Classifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Intent Classification Demo\nThis app uses AI to classify your text intent."
    }
)

# Custom CSS
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        /* background-color: #f0f2f6; */ /* Example: Light background */
    }
    /* Dark mode specific background (Streamlit handles theme switching) */
    [data-testid="stAppViewContainer"] > .main {
        /* background-color: #0e1117; */ /* Default Streamlit dark */
    }

    /* Center Title */
    h1 {
        text-align: center;
        /* Example Gradient Title */
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px; /* Add some space below title */
    }

    /* Base Button Styling - Remove specific padding/margin here */
    .stButton>button {
        border-radius: 20px;
        /* padding: 10px 15px; <-- REMOVE base padding */
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        /* margin-top: 45px; <-- REMOVE base margin */
    }

    /* --- Button Sizing and Alignment --- */

    /* Submit Button Style (Targeting button within 2nd main column) */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button {
        border: 2px solid #FF4B4B;
        color: #FF4B4B;
        background-color: transparent;
        /* --- Adjust Vertical Size --- */
        padding-top: 5px !important;    /* Reduce top padding */
        padding-bottom: 5px !important; /* Reduce bottom padding */
        padding-left: 15px;   /* Keep horizontal padding reasonable */
        padding-right: 15px;
        font-size: 0.9rem;    /* Slightly smaller font */
        line-height: 1.4;     /* Adjust line height */
        margin-top: 1px;      /* Adjust top margin for alignment */
    }
    /* Submit Hover/Focus (Keep as before or adjust if needed) */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button:hover {
        background-color: #FF4B4B;
        color: white;
        border-color: #FF4B4B;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.5);
    }

    /* Clear Button Style (Targeting button within 3rd main column) */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) .stButton > button {
        border: 1px solid #888;
        color: #888;
        font-weight: normal;
        background-color: transparent;
        /* --- Adjust Vertical Size (Match Submit) --- */
        padding-top: 5px !important;    /* Reduce top padding */
        padding-bottom: 5px !important; /* Reduce bottom padding */
        padding-left: 15px;
        padding-right: 15px;
        font-size: 0.9rem;    /* Slightly smaller font */
        line-height: 1.4;     /* Adjust line height */
        margin-top: 1px;      /* Adjust top margin for alignment */
    }
    /* Clear Hover (Keep as before or adjust if needed) */
    div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) .stButton > button:hover {
        background-color: #888;
        color: white;
        border-color: #888;
    }

    /* --- Keep Sidebar Button Styling Separate --- */
    /* Ensure sidebar button retains its original padding/size */
     div[data-testid="stSidebar"] .stButton > button {
        border-color: #1E90FF;
        color: #1E90FF;
        font-weight: normal;
        width: auto;
        padding: 5px 15px; /* Original smaller padding */
        margin-top: 5px;
    }
     div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1E90FF;
        color: white;
        border-color: #1E90FF;
    }

    /* --- Keep Sidebar Button Styling Separate --- */

     /* Input Area Styling (Keep as before, maybe remove height if causing issues) */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
        /* height: 56px; */ /* Removing fixed height might help alignment */
    }
    .stTextInput > div > div > input:focus {
        border-color: #FF4B4B;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.3);
    }

    /* Dataframe Styling */
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden; /* Ensures border radius is applied */
    }
    /* Customize header */
    .stDataFrame [data-testid="stDataFrameResizable"] > div > div > div[data-testid="stElementToolbar"] {
         background-color: #f8f9fa; /* Light header */
    }
     /* Dark mode header */
    [data-baseweb="dark-theme"] .stDataFrame [data-testid="stDataFrameResizable"] > div > div > div[data-testid="stElementToolbar"] {
         background-color: #333;
    }


</style>
""", unsafe_allow_html=True)


# Helper funcs
def classify_intent_api(text: str) -> tuple[list | None, str | None]:
    """Calls the backend API and returns results or error message."""
    payload = {"text": text}
    try:
        response = requests.post(INTENT_ENDPOINT, json=payload, timeout=15) # slightly longer timeout
        response.raise_for_status()
        try:
            data = response.json()
            if "intents" in data and isinstance(data["intents"], list):
                return data["intents"], None
            else:
                return None, "API returned unexpected data format."
        except requests.exceptions.JSONDecodeError:
            return None, f"API returned non-JSON response (Status {response.status_code})."
    except requests.exceptions.Timeout:
        return None, "API request timed out. Server might be busy or down."
    except requests.exceptions.ConnectionError:
        return None, f"Connection Error: Could not connect to the API server at {API_BASE_URL}. Is it running?"
    except requests.exceptions.HTTPError as http_err:
        error_label = "API_ERROR"
        error_message = str(http_err)
        try:
            error_data = response.json()
            error_label = error_data.get("label", error_label)
            error_message = error_data.get("message", error_message)
        except requests.exceptions.JSONDecodeError:
            pass
        return None, f"API Error ({response.status_code} - {error_label}): {error_message}"
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None, f"An unexpected application error occurred: {e}"

def check_server_readiness() -> tuple[bool, str]:
    """Checks the /ready endpoint and returns status."""
    try:
        response = requests.get(READY_ENDPOINT, timeout=5) # 5 second timeout for readiness
        if response.status_code == 200 and response.text == '"OK"': # FastAPI returns JSON string "OK"
            return True, "Server is ready! ✅"
        elif response.status_code == 423:
             return False, "Server is not ready (still loading model?). ⏳"
        else:
            return False, f"Server responded with status {response.status_code}. 🤔"
    except requests.exceptions.Timeout:
        return False, "Connection timed out checking server readiness. ⏳"
    except requests.exceptions.ConnectionError:
        return False, f"Connection Error: Could not reach server at {API_BASE_URL}. ❌"
    except Exception as e:
        return False, f"Error checking server readiness: {e} ❌"

# Init Session State
if "intent_query" not in st.session_state:
    st.session_state.intent_query = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0 # Key to help reset text input

# Sidebar
with st.sidebar:
    logo_path = "/workspaces/zendesk-mle/coding_task/docs/Zendesk-Logo.png"
    try:
        st.image(logo_path, use_container_width='auto')
    except FileNotFoundError:
        st.error(f"Logo not found at path: {logo_path}")
    except Exception as e:
        st.error(f"Error loading logo: {e}")

    st.markdown("---")

    st.markdown("### Server Status")
    if st.button("Check API Readiness 📡"):
        ready, message = check_server_readiness()
        if ready:
            st.toast(message, icon="✅") 
        else:
            st.toast(message, icon="⚠️")

    st.markdown("---")
    st.markdown("**About**")
    st.info("Demo app for Zendesk Senior MLE Coding Task")


# main
st.title("AI Intent Classifier")
st.markdown("👋 Welcome! Enter a sentence below to understand its underlying intent")

# inout area
col_input, col_submit, col_clear = st.columns([5, 1, 1], gap="small") 

with col_input:
    user_input = st.text_input(
        "Your query:",
        value=st.session_state.intent_query,
        key=f"text_input_{st.session_state.input_key}",
        placeholder="e.g., 'what flights are available from london to paris?'",
        label_visibility="collapsed"
    )
    st.session_state.intent_query = user_input

with col_submit:
    submit_button = st.button("Submit ✨", use_container_width=True, key="submit")

with col_clear:
    clear_button = st.button("Clear 🗑️", use_container_width=True, key="clear")



# Btn logic
if submit_button:
    query = st.session_state.intent_query
    if not query or not query.strip():
        st.session_state.error_message = '"text" is empty. Please enter a query.'
        st.session_state.results = None
    else:
        st.session_state.error_message = None
        st.session_state.results = None
        with st.spinner("🧠 Thinking... Calling the AI model..."):
            results, error = classify_intent_api(query)
            if error:
                st.session_state.error_message = error
            else:
                st.session_state.results = results

if clear_button:
    st.session_state.intent_query = ""
    st.session_state.results = None
    st.session_state.error_message = None
    st.session_state.input_key += 1
    # need to trigger a rerun after state change for input clearing
    st.rerun()

st.markdown("---")

if st.session_state.error_message:
    st.error(f"🚨 **Error:** {st.session_state.error_message}")

# Results from AI service
if st.session_state.results is not None:
    st.subheader("✨ Top Intent Predictions")
    if not st.session_state.results:
        st.warning("🤔 The API returned no specific intent predictions for this query.")
    else:
        # Convert results to Pandas DataFrame for better display
        try:
            df = pd.DataFrame(st.session_state.results)
            df['confidence_formatted'] = df['confidence'].map('{:.1%}'.format) # Format as percentage
            df_display = df[['label', 'confidence_formatted']].rename(
                columns={"label": "Predicted Intent", "confidence_formatted": "Confidence"}
            )
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Predicted Intent": st.column_config.TextColumn(
                        "Intent",
                        help="The predicted intent label.",
                        width="large"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "Confidence Score",
                        help="The model's confidence score for this intent.",
                        width="medium"
                    )
                }
            )
        except Exception as e:
            st.error(f"Error displaying results: {e}")
            st.json(st.session_state.results) # Show raw JSON as fallback


st.markdown("---")
st.caption("Made with ❤️ by "'<a href="https://akjo.tech" target="_blank">Akshay Joshi</a>', unsafe_allow_html=True)


# Run this using this command:
# python -m streamlit run intent_classifier_streamlit.py