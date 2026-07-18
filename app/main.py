import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Adaptive Quiz Renderer",
        page_icon=None,
        layout="centered",
    )

    st.title("Adaptive Quiz Renderer")
    st.write("The application environment is configured successfully.")


if __name__ == "__main__":
    main()
