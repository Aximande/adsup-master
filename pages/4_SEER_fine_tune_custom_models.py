import streamlit as st
import replicate
import os
from io import BytesIO
import re

# Initialize Replicate client with API token
REPLICATE_API_TOKEN = "r8_eCOg3M8VkogXEgbMgzRKfrpxw3TuZkS3s8Jen"

if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found. Please set it in the code.")
    st.stop()

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Helper Functions
def validate_model_name(model_name):
    """
    Validates that the model name follows Replicate's naming rules.
    Must follow the 'username/model-name' format.
    """
    # Check if the model name follows the 'username/model-name' format
    if '/' not in model_name:
        st.error("Le nom du modèle doit suivre le format 'username/model-name'.")
        return None

    # Split into username and model name
    owner, name = model_name.split('/', 1)

    # Check for valid characters (lowercase letters, numbers, dashes, underscores, periods)
    if not re.match(r"^[a-z0-9._-]+$", owner) or not re.match(r"^[a-z0-9._-]+$", name):
        st.error("Le nom du modèle ne peut contenir que des lettres minuscules, des chiffres, des tirets (-), des underscores (_), et des points (.).")
        return None

    return owner, name

def start_training(dest_model_name, uploaded_file, trigger_word, steps, autocaption,
                   lora_rank, optimizer, batch_size, learning_rate,
                   wandb_project, wandb_save_interval, caption_dropout_rate,
                   cache_latents_to_disk, wandb_sample_interval, hf_token, hf_repo_id):
    """
    Starts the training process on Replicate with additional parameters.
    """
    # Validate model name and extract owner and model name
    model_name_validation = validate_model_name(dest_model_name)
    if not model_name_validation:
        st.stop()

    owner, name = model_name_validation

    try:
        # Check if model exists
        model = replicate_client.models.get(f"{owner}/{name}")
        st.info(f"Le modèle '{dest_model_name}' existe déjà. Utilisation du modèle existant.")
    except Exception:
        # Create the model if it doesn't exist
        model = replicate_client.models.create(
            owner=owner,
            name=name,
            visibility="private",
            hardware=selected_hardware,  # Set selected hardware
            description=f"Un modèle FLUX.1 fine-tuned avec le mot déclencheur '{trigger_word}'"
        )
        st.success(f"Le modèle '{dest_model_name}' a été créé avec succès.")

    try:
        # Read the uploaded ZIP file
        zip_bytes = uploaded_file.read()
        zip_file = BytesIO(zip_bytes)

        # Start the training
        training = replicate_client.trainings.create(
            version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
            input={
                "input_images": zip_file,
                "trigger_word": trigger_word,
                "steps": steps,
                "autocaption": autocaption,
                "lora_rank": lora_rank,
                "optimizer": optimizer,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "wandb_project": wandb_project,
                "wandb_save_interval": wandb_save_interval,
                "caption_dropout_rate": caption_dropout_rate,
                "cache_latents_to_disk": cache_latents_to_disk,
                "wandb_sample_interval": wandb_sample_interval,
                "hf_token": hf_token if hf_token else "",
                "hf_repo_id": hf_repo_id if hf_repo_id else "",
            },
            destination=f"{owner}/{name}"
        )

        training_url = f"https://replicate.com/p/{training.id}"
        return f"L'entraînement a commencé ! Vous pouvez suivre le progrès [ici]({training_url})."

    except Exception as e:
        st.error(f"Une erreur est survenue pendant l'entraînement : {e}")
        return None

# Streamlit App Layout
st.set_page_config(page_title="FLUX.1 Fine-Tuning & Génération d'Images", layout="wide")

st.title("Outil d'entraînement et génération d'images pour FLUX.1")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choisissez le mode de l'application", ["Fine-tuning du modèle", "Générer des images"])

# Hardware Pricing Information
hardware_options = {
    "cpu": {"price": 0.0001, "description": "CPU (0.36$/hr)"},
    "gpu-a100-large": {"price": 0.0014, "description": "Nvidia A100 (80GB) GPU ($5.04/hr)"},
    "gpu-a40-large": {"price": 0.000725, "description": "Nvidia A40 Large ($2.61/hr)"},
    "gpu-a40-small": {"price": 0.000575, "description": "Nvidia A40 Small ($2.07/hr)"},
    "gpu-t4": {"price": 0.000225, "description": "Nvidia T4 ($0.81/hr)"}
}

# Select hardware with price info
selected_hardware = st.sidebar.selectbox(
    "Choisissez un matériel pour l'entraînement (basé sur 30 minutes de temps d'entraînement)",
    options=hardware_options.keys(),
    format_func=lambda x: f"{hardware_options[x]['description']} (~{hardware_options[x]['price']*30*60:.2f}$ pour 30 minutes)"
)

if app_mode == "Fine-tuning du modèle":
    st.header("Fine-tuning du modèle FLUX.1")

    # Input for destination model name
    dest_model_name = st.text_input(
        "Nom du modèle de destination",
        value="aximande/flux-your-model-name",
        help="Format : 'username/nom-du-modèle'. Remplacez 'aximande' et 'flux-your-model-name' en conséquence."
    )

    # Upload zip file containing training images
    uploaded_file = st.file_uploader("Téléchargez un fichier ZIP contenant les images d'entraînement", type=["zip"])

    # Input for trigger word
    trigger_word = st.text_input(
        "Mot déclencheur",
        value="TOK",
        help="Un mot unique pour associer votre concept fine-tuned. Utilisez ce mot dans vos prompts pour activer le concept."
    )

    # Training steps
    steps = st.number_input(
        "Nombre d'étapes",
        min_value=3,
        max_value=6000,
        value=1000,
        help="Nombre d'étapes d'entraînement. Gamme recommandée : 500-4000."
    )

    st.markdown("---")

    # Autocaption Option
    autocaption = st.checkbox(
        "Activer la génération automatique de légendes",
        value=True,
        help="Générer automatiquement des légendes pour vos images d'entraînement."
    )

    # Toggle for Advanced Options
    if st.checkbox("Afficher les options avancées"):
        st.subheader("Paramètres supplémentaires d'entraînement")

        # LoRA Rank
        lora_rank = st.number_input(
            "LoRA Rank",
            min_value=1,
            max_value=128,
            value=16,
            help="Des rangs plus élevés prennent plus de temps à entraîner mais peuvent capturer des fonctionnalités plus complexes."
        )

        # Optimizer
        optimizer = st.selectbox(
            "Optimiseur",
            options=["adamw8bit", "sgd", "adam"],
            index=0,
            help="Choisissez l'optimiseur pour l'entraînement."
        )

        # Batch Size
        batch_size = st.number_input(
            "Taille du lot",
            min_value=1,
            max_value=16,
            value=1,
            help="Nombre d'échantillons par lot."
        )

        # Learning Rate
        learning_rate = st.number_input(
            "Taux d'apprentissage",
            min_value=0.0001,
            max_value=0.01,
            value=0.0004,
            format="%.6f",
            help="Le taux d'apprentissage pour l'entraînement."
        )

        # WandB Project
        wandb_project = st.text_input(
            "Projet WandB",
            value="flux_train_replicate",
            help="Nom du projet Weights & Biases pour le suivi."
        )

        # WandB Save Interval
        wandb_save_interval = st.number_input(
            "Intervalle de sauvegarde WandB",
            min_value=1,
            max_value=1000,
            value=100,
            help="Intervalle (en étapes) pour sauvegarder des checkpoints sur WandB."
        )

        # Caption Dropout Rate
        caption_dropout_rate = st.slider(
            "Taux d'abandon de légende",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Taux d'abandon des légendes pour éviter le sur-apprentissage."
        )

        # Cache Latents to Disk
        cache_latents_to_disk = st.checkbox(
            "Mettre en cache les latents sur disque",
            value=False,
            help="Mettre en cache les représentations latentes sur disque pour accélérer l'entraînement."
        )

        # WandB Sample Interval
        wandb_sample_interval = st.number_input(
            "Intervalle d'échantillonnage WandB",
            min_value=1,
            max_value=1000,
            value=100,
            help="Intervalle (en étapes) pour échantillonner et journaliser les images sur WandB."
        )

        # Hugging Face Token and Repo ID (Optional)
        hf_token = st.text_input(
            "Token Hugging Face (facultatif)",
            value="",
            help="Si vous souhaitez sauvegarder votre modèle sur Hugging Face, fournissez votre token ici."
        )
        hf_repo_id = st.text_input(
            "ID du dépôt Hugging Face (facultatif)",
            value="",
            help="L'ID du dépôt sur Hugging Face où le modèle sera sauvegardé (par ex. 'username/repo-name')."
        )

    else:
        # Default values for hidden advanced options
        lora_rank = 16
        optimizer = "adamw8bit"
        batch_size = 1
        learning_rate = 0.0004
        wandb_project = "flux_train_replicate"
        wandb_save_interval = 100
        caption_dropout_rate = 0.05
        cache_latents_to_disk = False
        wandb_sample_interval = 100
        hf_token = ""
        hf_repo_id = ""

    st.markdown("---")

    # Start Training Button
    if st.button("Lancer l'entraînement"):
        if not dest_model_name or not uploaded_file or not trigger_word:
            st.error("Veuillez fournir tous les paramètres requis : Nom du modèle, Images d'entraînement et Mot déclencheur.")
        else:
            with st.spinner("Lancement de l'entraînement... Cela peut prendre 20 à 30 minutes."):
                status_message = start_training(
                    dest_model_name=dest_model_name,
                    uploaded_file=uploaded_file,
                    trigger_word=trigger_word,
                    steps=steps,
                    autocaption=autocaption,
                    lora_rank=lora_rank,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    wandb_project=wandb_project,
                    wandb_save_interval=wandb_save_interval,
                    caption_dropout_rate=caption_dropout_rate,
                    cache_latents_to_disk=cache_latents_to_disk,
                    wandb_sample_interval=wandb_sample_interval,
                    hf_token=hf_token,
                    hf_repo_id=hf_repo_id
                )
                if status_message:
                    st.success(status_message)
                else:
                    st.error("L'entraînement n'a pas pu démarrer.")

# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ en utilisant [Streamlit](https://streamlit.io/) et [Replicate](https://replicate.com/).")
