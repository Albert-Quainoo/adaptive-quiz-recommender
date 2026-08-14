import streamlit as st

STEEP_CSS = """
<style>
:root {
    --color-ink: #f4f6fb;
    --color-canvas: #0e1117;
    --color-input: #171b24;
    --color-sidebar: #11151d;
    --color-muted: #a1a8b5;
    --color-border: #303746;
    --color-accent-blue: #5b7cfa;
    --radius-buttons: 9999px;
    --radius-inputs: 16px;
}

/* Page canvas */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: var(--color-canvas) !important;
    color: var(--color-ink) !important;
}

header[data-testid="stHeader"] {
    background-color: var(--color-canvas) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"],
.stRadio label p {
    color: var(--color-ink) !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--color-muted) !important;
}

/* Headings -> serif substitute (Signifier isn't a free web font;
   Source Serif 4 or Georgia are the documented fallbacks) */
h1, h2 {
    font-family: "Source Serif 4", Georgia, serif !important;
    font-weight: 400 !important;
    color: var(--color-ink) !important;
    letter-spacing: -0.02em;
}

/* Buttons -> pill */
.stButton > button, .stFormSubmitButton > button {
    border-radius: var(--radius-buttons) !important;
    background-color: var(--color-accent-blue) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.5em 1.5em !important;
}

/* Text inputs -> rounded, mist fill */
.stTextInput > div > div > input {
    border-radius: var(--radius-inputs) !important;
    background-color: var(--color-input) !important;
    color: var(--color-ink) !important;
    border: 1px solid var(--color-border) !important;
}

/* Sidebar -> quiet dark band, distinct from canvas */
section[data-testid="stSidebar"] {
    background-color: var(--color-sidebar) !important;
}

</style>
"""

def inject_theme() -> None:
    st.markdown(STEEP_CSS, unsafe_allow_html=True)
