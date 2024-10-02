import streamlit as st
import os
from utils import load_custom_css

def main():
    # Configuration de la page avec un thème personnalisé via config.toml
    st.set_page_config(
        page_title="Ad's up BU IA",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Charger le CSS personnalisé
    load_custom_css()

    # Afficher le header avec le logo et le titre
    display_header()

    # Contenu principal de la page d'accueil
    st.title("Bienvenue dans la plateforme de test de notre BU IA")
    st.write(
        """
        Cette application vous offre plusieurs outils pour vous assister dans vos tâches quotidiennes. Utilisez la barre latérale pour naviguer entre les différentes fonctionnalités :

        - **Catégorisation de Mots-Clés :** Classez et organisez vos mots-clés de manière efficace.
        - **Générateur d'Images :** Créez des images alimentées par l'IA en fonction de vos prompts.
        - **Analyseur de vidéo YouTube :** Créez des résumés dans n'importe quelle langue de n'importe quelle vidéo youtube afin de booster votre veille

        ---
        Pour toute question, veuillez contacter **[alexandre.lavallee@ads-up.fr](mailto:alexandre.lavallee@ads-up.fr)**.

        Consultez le document compilant tous les cas d'usage : [**Cas d'Usage IA**](https://docs.google.com/spreadsheets/d/1Vb_3WDKX63S_emlM2UsyvJr4KA4j-ofFHiVWwjJ--Y8/edit?usp=sharing)

        Partagez vos besoins via notre formulaire : [**Formulaire de Retour**](https://docs.google.com/forms/d/1mV95eF6SBRVsbWbSWg2xn1FglST6vW3UEZIZEAfL7Ag/prefill)
        """
    )

    # Afficher le pied de page
    display_footer()

def display_header():
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        # Ajouter le logo du client (Ad's up consulting)
        logo_path = os.path.join("adsup-logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=150, use_column_width=False, caption=None)
        else:
            st.warning("Logo Ad's up consulting non trouvé. Assurez-vous que 'adsup-logo.png' est dans le dossier 'images'.")
    with col_title:
        st.markdown("""
        <h1 style='text-align: center; color: #FFFFFF;'>Ad's up BU IA plateforme de test</h1>
        """, unsafe_allow_html=True)

def display_footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    col_footer = st.columns([1])
    with col_footer[0]:
        # Ajouter le logo de ads up
        adsup_logo_path = os.path.join("images", "ads-logo.png")
        if os.path.exists(adsup_logo_path):
            st.image(adsup_logo_path, width=150, use_column_width=False, caption="Développé par Ad's up consulting")
        else:
            st.warning("Logo Ad's up non trouvé. Assurez-vous que 'ads-up-logo.png' est dans le dossier 'images'.")

    # Ajouter les badges GitHub et Twitter
    st.markdown("""
    <div style="text-align: center; margin-top: 10px;">
        <a href="https://github.com/Aximande" target="_blank" class="badge">
            <img src="https://img.shields.io/github/followers/Aximande?label=Suivre%20sur%20GitHub&style=social" alt="Suivre sur GitHub">
        </a>
        <a href="https://twitter.com/adsupfr" target="_blank" class="badge">
            <img src="https://img.shields.io/twitter/follow/adsupfr?style=social" alt="Suivre sur Twitter">
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
