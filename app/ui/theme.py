import streamlit as st

STEEP_CSS = """
<style>
:root {
    --color-ink-black: #17191c;
    --color-paper-white: #ffffff;
    --color-mist-gray: #f2f2f3;
    --color-fog-white: #fafafb;
    --color-slate-gray: #777b86;
    --color-blush-peach: #fbe1d1;
    --color-sienna-brown: #5d2a1a;
    --radius-cards: 24px;
    --radius-buttons: 9999px;
    --radius-inputs: 16px;
}

/* Page canvas */
.stApp { background-color: var(--color-paper-white); }

/* Headings -> serif substitute (Signifier isn't a free web font;
   Source Serif 4 or Georgia are the documented fallbacks) */
h1, h2 {
    font-family: "Source Serif 4", Georgia, serif !important;
    font-weight: 400 !important;
    color: var(--color-ink-black);
    letter-spacing: -0.02em;
}

/* Buttons -> pill */
.stButton > button, .stFormSubmitButton > button {
    border-radius: var(--radius-buttons) !important;
    background-color: var(--color-ink-black) !important;
    color: var(--color-paper-white) !important;
    border: none !important;
    padding: 0.5em 1.5em !important;
}

/* Text inputs -> rounded, mist fill */
.stTextInput > div > div > input {
    border-radius: var(--radius-inputs) !important;
    background-color: var(--color-mist-gray) !important;
    border: none !important;
}

/* Sidebar -> fog band, distinct from canvas */
section[data-testid="stSidebar"] { background-color: var(--color-fog-white); }

/* st.info as the one permitted peach accent surface */
div[data-testid="stAlertContainer"] {
    background-color: var(--color-blush-peach) !important;
    color: var(--color-sienna-brown) !important;
    border-radius: var(--radius-cards) !important;
    border: none !important;
}
</style>
"""

def inject_theme() -> None:
    st.markdown(STEEP_CSS, unsafe_allow_html=True)