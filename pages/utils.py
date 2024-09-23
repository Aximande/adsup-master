import streamlit as st

def load_custom_css():
    custom_css = """
    <style>
    /* Importer les polices depuis Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Roboto:wght@400;700&display=swap');

    /* Appliquer les polices */
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        color: #FFFFFF;
    }

    /* Titres en Poppins */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }

    /* Arrière-plan */
    body {
        background-color: #01030F;
    }

    /* Couleurs des éléments Streamlit */
    .stApp {
        background-color: #01030F;
    }

    /* Masquer le menu hamburger et le pied de page Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Personnaliser les boutons en forme de gélule */
    .stButton > button {
        background-color: #5281a6;  /* Medium Blue */
        color: white;
        border-radius: 50px;  /* Bouton en forme de gélule */
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #2b556b;  /* Dark Blue */
    }

    /* Personnaliser les sliders */
    .stSlider > div > div > div > div {
        color: #5281a6;  /* Medium Blue */
    }

    /* Personnaliser les en-têtes */
    .css-1aumxhk {
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }

    /* Personnaliser les champs de texte et zones de texte */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #333435;  /* Anthracite */
        color: #FFFFFF;
        border: 1px solid #5281a6;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
    }

    /* Personnaliser les sélecteurs */
    .stSelectbox>div>div>div>select, .stMultiselect>div>div>div>div>input {
        background-color: #333435;  /* Anthracite */
        color: #FFFFFF;
        border: 1px solid #5281a6;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
    }

    /* Personnaliser les expandeurs */
    .streamlit-expanderHeader {
        font-size: 16px;
        color: #FFFFFF;
        font-weight: bold;
    }

    /* Personnaliser la barre de progression */
    .stProgress > div > div > div > div {
        background-color: #5281a6;
    }

    /* Styliser les liens */
    a {
        color: #5281a6;
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Centrer les logos */
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
    }

    /* Styliser les sections (cards) */
    .section {
        padding: 20px;
        border-radius: 8px;
        background-color: #333435;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Styliser les badges */
    .badge {
        display: inline-block;
        margin: 0 10px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
