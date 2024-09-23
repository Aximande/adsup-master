import os
import asyncio
import re
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from typing import List, Dict
import pandas as pd
import streamlit as st
import time
from utils import load_custom_css

# Initialize OpenAI client

api_key = st.secrets["OPENAI_API_KEY"]

client = AsyncOpenAI(api_key=api_key)

# Pricing per 1000 tokens
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
    "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015}
}

def calculate_cost(total_prompt_tokens, total_completion_tokens, model_name):
    input_cost_rate = MODEL_PRICING[model_name]["input"]
    output_cost_rate = MODEL_PRICING[model_name]["output"]

    prompt_cost = (total_prompt_tokens / 1000) * input_cost_rate
    completion_cost = (total_completion_tokens / 1000) * output_cost_rate
    total_cost = prompt_cost + completion_cost

    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }

# Define Pydantic data models
class KeywordClassification(BaseModel):
    category: str
    subcategory: str
    explanation: str = Field(description="Brève explication de la classification")

# System prompt for keyword classification
SYSTEM_PROMPT = """
Vous êtes un assistant IA pour une équipe SEO.
Votre rôle est d'analyser les mots-clés et de les classer dans des catégories et sous-catégories appropriées pour aider notre équipe à organiser et prioriser leurs efforts SEO.
Vos tâches :
1. Catégoriser le mot-clé dans la catégorie la plus appropriée de la liste fournie.
2. Attribuer le mot-clé à la sous-catégorie la plus appropriée au sein de la catégorie choisie.
3. Donner une brève explication de votre choix de classification.
Répondez dans le format suivant :
Catégorie : [catégorie choisie]
Sous-catégorie : [sous-catégorie choisie]
Explication : [brève explication]
N'oubliez pas :
- Soyez objectif et basez votre analyse sur le mot-clé fourni et les informations données sur le client.
- L'explication doit être concise mais informative, soulignant pourquoi le mot-clé correspond à la catégorie et sous-catégorie choisies.
- Prenez en compte l'industrie du client, le public cible et les objectifs commerciaux dans votre classification.
"""

def load_categories(file):
    try:
        categories_df = pd.read_csv(file)
        st.write("Colonnes dans le fichier CSV des catégories téléchargé :")
        st.write(categories_df.columns.tolist())

        # Allow the user to select the category and subcategory columns
        category_column = st.selectbox("Sélectionnez la colonne contenant les catégories :", categories_df.columns.tolist())
        subcategory_column = st.selectbox("Sélectionnez la colonne contenant les sous-catégories :", categories_df.columns.tolist())

        categories = {}
        for _, row in categories_df.iterrows():
            category = row[category_column]
            subcategory = row[subcategory_column]
            if category not in categories:
                categories[category] = []
            if pd.notnull(subcategory) and subcategory not in categories[category]:
                categories[category].append(subcategory)
        return categories
    except Exception as e:
        st.error(f"Erreur lors du chargement des catégories : {str(e)}")
        return None

async def classify_keyword(keyword: str, categories: Dict[str, List[str]], client_info: str, model_name: str) -> tuple:
    try:
        categories_str = "\n".join([f"{cat}: {', '.join(subcats)}" for cat, subcats in categories.items()])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Informations sur le client :\n{client_info}\n\nMot-clé : {keyword}\nCatégories et sous-catégories :\n{categories_str}"}
        ]

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        content = response.choices[0].message.content

        # Extract usage information
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Use regex to extract category, subcategory, and explanation
        category_match = re.search(r'Catégorie\s*:\s*(.+)', content)
        subcategory_match = re.search(r'Sous-catégorie\s*:\s*(.+)', content)
        explanation_match = re.search(r'Explication\s*:\s*(.+)', content)

        if not all([category_match, subcategory_match, explanation_match]):
            raise ValueError("Impossible d'analyser correctement la réponse de l'IA")

        category = category_match.group(1).strip()
        subcategory = subcategory_match.group(1).strip()
        explanation = explanation_match.group(1).strip()

        classification = KeywordClassification(category=category, subcategory=subcategory, explanation=explanation)
        return classification, prompt_tokens, completion_tokens, total_tokens
    except Exception as e:
        st.error(f"Erreur lors de la classification du mot-clé '{keyword}' : {str(e)}")
        return None, 0, 0, 0

async def process_keywords(keywords, categories, client_info, model_name, progress_bar, progress_text):
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    for i, keyword in enumerate(keywords, 1):
        classification, prompt_tokens, completion_tokens, tokens = await classify_keyword(keyword, categories, client_info, model_name)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += tokens
        if classification:
            results.append({
                'mot-clé': keyword,
                'catégorie': classification.category,
                'sous-catégorie': classification.subcategory,
                'explication': classification.explanation
            })

        # Update progress
        progress_bar.progress(i / len(keywords))
        progress_text.text(f"Traité {i}/{len(keywords)} mots-clés")

    return results, total_prompt_tokens, total_completion_tokens, total_tokens

def main():
    # Configurer la page
    st.set_page_config(
        page_title="Classification de Mots-clés SEO",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Charger le CSS personnalisé
    load_custom_css()

    # Afficher le logo
    logo_path = "adsup-logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.warning("Logo non trouvé. Assurez-vous que le fichier 'adsup-logo.png' est dans le bon répertoire.")

    st.title("Classification de Mots-clés SEO")

    # Model selection
    model_name = st.selectbox(
        "Sélectionnez le modèle OpenAI",
        options=[
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13"
        ],
        index=0,
        help="Choisissez le modèle OpenAI à utiliser pour la classification. Différents modèles ont des coûts et des capacités différents."
    )

    # Client Information Input
    st.subheader("Informations sur le Client")
    client_info = st.text_area(
        "Entrez des informations importantes sur le client (par exemple, audience, objectifs commerciaux, particularités) ou des instructions spécifiques au projet pour guider le processus de classification :",
        help="Ces informations aideront l'IA à classer les mots-clés de manière plus précise et pertinente pour votre client."
    )

    # Category Input Option
    category_option = st.radio(
        "Comment souhaitez-vous saisir les catégories et sous-catégories ?",
        ("Télécharger un CSV", "Saisir manuellement"),
        help="Choisissez la méthode qui vous convient le mieux pour définir vos catégories et sous-catégories."
    )

    categories = None

    if category_option == "Télécharger un CSV":
        st.subheader("Télécharger le CSV des Catégories et Sous-catégories")
        categories_file = st.file_uploader("Choisissez un fichier CSV pour les Catégories et Sous-catégories", type="csv", help="Le fichier CSV doit contenir deux colonnes : une pour les catégories et une pour les sous-catégories.")

        if categories_file is not None:
            categories = load_categories(categories_file)
            if categories:
                st.success("Catégories et sous-catégories chargées avec succès !")

                # Display the loaded categories and subcategories
                st.subheader("Catégories et Sous-catégories Chargées")
                for category, subcategories in categories.items():
                    st.write(f"**{category}**")
                    for subcategory in subcategories:
                        st.write(f"- {subcategory}")
                    st.write("")  # Add a blank line between categories
            else:
                st.error("Échec du chargement des catégories. Veuillez vérifier votre fichier CSV.")
    else:
        # Manual input for categories and subcategories
        st.subheader("Saisir les Catégories et Sous-catégories")
        num_categories = st.number_input("Nombre de catégories", min_value=1, value=3, help="Définissez le nombre de catégories principales que vous souhaitez utiliser.")
        categories = {}
        for i in range(int(num_categories)):
            cat = st.text_input(f"Catégorie {i+1}", key=f"category_{i}", help=f"Entrez le nom de la catégorie {i+1}")
            if cat:
                subcats = st.text_input(f"Sous-catégories pour {cat} (séparées par des virgules)", key=f"subcategories_{i}", help=f"Entrez les sous-catégories pour {cat}, séparées par des virgules")
                categories[cat] = [subcat.strip() for subcat in subcats.split(',') if subcat.strip()]

    if not categories:
        st.warning("Veuillez fournir des catégories et sous-catégories.")
        return

    # File upload for Keywords CSV
    st.subheader("Télécharger le CSV des Mots-clés")
    keywords_file = st.file_uploader("Choisissez un fichier CSV pour les Mots-clés", type="csv", help="Le fichier CSV doit contenir une colonne avec vos mots-clés à classifier.")

    if keywords_file is not None:
        try:
            df = pd.read_csv(keywords_file)
            st.write("Colonnes dans le fichier CSV des mots-clés téléchargé :")
            st.write(df.columns.tolist())

            # Allow user to select the column with the keywords
            keyword_column = st.selectbox("Sélectionnez la colonne contenant les mots-clés :", df.columns.tolist(), help="Choisissez la colonne de votre CSV qui contient les mots-clés à classifier.")

            num_keywords = len(df)
            st.write(f"Le fichier téléchargé contient {num_keywords} mots-clés")

            # Option to process all keywords or a sample
            process_all = st.checkbox("Traiter tous les mots-clés", value=True, help="Décochez cette case si vous souhaitez traiter seulement un échantillon de vos mots-clés.")
            if not process_all:
                sample_size = st.slider("Sélectionnez la taille de l'échantillon", min_value=1, max_value=num_keywords, value=min(10, num_keywords), help="Choisissez le nombre de mots-clés à traiter dans votre échantillon.")
                df_sample = df.sample(n=sample_size, random_state=42)
                st.write(f"Traitement d'un échantillon de {sample_size} mots-clés")
            else:
                df_sample = df

            keywords = df_sample[keyword_column].astype(str).tolist()

            if st.button("Classifier les Mots-clés", help="Cliquez ici pour lancer le processus de classification des mots-clés."):
                if not client_info:
                    st.warning("Veuillez fournir des informations sur le client pour guider le processus de classification.")
                    return

                st.write(f"Traitement de {len(keywords)} mots-clés en utilisant le modèle {model_name}")
                st.write(f"Cela nécessitera environ {len(keywords)} appels API")

                progress_bar = st.progress(0)
                progress_text = st.empty()

                start_time = time.time()
                results, total_prompt_tokens, total_completion_tokens, total_tokens = asyncio.run(
                    process_keywords(keywords, categories, client_info, model_name, progress_bar, progress_text)
                )
                end_time = time.time()

                # Create results dataframe
                results_df = pd.DataFrame(results)

                # Display results
                st.write("\nRésultats de la Classification :")
                st.dataframe(results_df)

                # Display summary
                st.write("\nRésumé de la Classification :")
                summary_df = results_df.groupby(['catégorie', 'sous-catégorie']).size().unstack(fill_value=0)
                st.write(summary_df)

                # Calculate and display cost
                cost_breakdown = calculate_cost(total_prompt_tokens, total_completion_tokens, model_name)
                st.write(f"\nTotal des tokens de prompt : {total_prompt_tokens}")
                st.write(f"Total des tokens de complétion : {total_completion_tokens}")
                st.write(f"Total des tokens : {total_tokens}")
                st.write(f"Coût des tokens de prompt : {cost_breakdown['prompt_cost']:.6f}€")
                st.write(f"Coût des tokens de complétion : {cost_breakdown['completion_cost']:.6f}€")
                st.write(f"Coût total estimé pour les {len(keywords)} appels API : {cost_breakdown['total_cost']:.6f}€")

                st.write(f"\nNombre total d'appels API effectués : {len(results)}")
                st.write(f"Temps total de traitement : {end_time - start_time:.2f} secondes")

                # Option to download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name="resultats_classification_mots_cles.csv",
                    mime="text/csv",
                    help="Cliquez ici pour télécharger les résultats de la classification au format CSV."
                )

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier de mots-clés téléchargé : {str(e)}")

if __name__ == "__main__":
    main()
