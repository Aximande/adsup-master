import streamlit as st
st.set_page_config(
    page_title="Alfred - Assistant de Génération d'Images",
    layout="wide",
    initial_sidebar_state="expanded",
)
import replicate
import os
import csv
import uuid
import json
import requests
from datetime import datetime
#from dotenv import load_dotenv
from transformers import pipeline
from utils import load_custom_css

def main():
    display_header()
    render_sidebar()

    st.subheader("📝 Entrez votre prompt")
    prompt = st.text_area(
        "Prompt",
        value=st.session_state.prompt_history[-1] if st.session_state.prompt_history else "",
        height=150,
        placeholder="Entrez votre description ici...",
        help="Saisissez une description détaillée pour générer une image."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Enregistrer le prompt"):
            st.session_state.prompt_history.append(prompt)
            st.success("Prompt enregistré.")
    with col2:
        if st.button("✨ Raffiner le prompt"):
            if prompt.strip():
                with st.spinner("Raffinement du prompt en cours..."):
                    enhanced_prompt = enhance_prompt(prompt)
                    st.session_state.prompt_history.append(enhanced_prompt)
                    st.success("Prompt raffiné et enregistré.")
            else:
                st.warning("Veuillez d'abord entrer un prompt.")

    if st.session_state.prompt_history:
        editable_prompt = st.text_area(
            "📝 Modifier le prompt",
            value=st.session_state.prompt_history[-1],
            height=150,
            help="Vous pouvez modifier le prompt actuel."
        )
        if st.button("💾 Enregistrer les modifications"):
            st.session_state.prompt_history.append(editable_prompt)
            st.success("Modifications enregistrées.")
    else:
        st.warning("Aucun prompt n'a été enregistré pour le moment.")

    show_text_section = st.checkbox("Ajouter du texte à l'image", value=False)

    if show_text_section:
        st.subheader("🖋️ Ajouter du Texte à l'Image")
        text_to_add = st.text_input(
            "✏️ Texte à Ajouter",
            value="",
            help="Entrez le texte que vous souhaitez intégrer à l'image. L'IA tentera d'ajouter ce texte à l'image générée."
        )
        st.session_state['text_to_add'] = text_to_add

        typography_options_list = [
            "Bold", "Italic", "Underlined", "Shadow", "3D", "Gradient", "Handwritten",
            "Calligraphy", "Graffiti", "Vintage", "Futuristic", "Neon", "Glow",
            "Comic", "Stencil", "Watercolor", "Chalk", "Marker", "Spray Paint"
        ]
        typography_options = st.multiselect(
            "🔠 Options de Typographie",
            options=typography_options_list,
            help="Sélectionnez les styles typographiques à appliquer au texte ajouté. Plus vous choisissez de styles, plus le rendu sera complexe."
        )
        st.session_state['typography_options'] = typography_options

    if st.session_state.prompt_history:
        prompt_to_use = st.session_state.prompt_history[-1]
        final_prompt = construct_final_prompt(
            prompt_to_use,
            st.session_state.get('text_to_add', ''),
            st.session_state.get('typography_options', [])
        )
        st.subheader("Prompt Final :")
        st.info(final_prompt)
    else:
        final_prompt = None
        st.warning("Veuillez entrer un prompt pour continuer.")

    if st.button("🚀 Générer l'Image"):
        if not final_prompt:
            st.error("❌ Veuillez entrer un prompt valide.")
        elif not st.session_state['model']:
            st.error("❌ Veuillez sélectionner un modèle.")
        else:
            with st.spinner("🖼️ Génération en cours..."):
                try:
                    output = generate_image(final_prompt, st.session_state['model'], st.session_state['settings'])
                    if output:
                        st.success("✅ Image(s) générée(s) avec succès !")
                        if isinstance(output, list):
                            for idx, img_url in enumerate(output):
                                st.image(img_url, use_column_width=True, caption=f"Image {idx+1}")
                        else:
                            st.image(output, use_column_width=True, caption="Image générée")

                        log_interaction(
                            prompt=final_prompt,
                            parameters=st.session_state['settings'],
                            output_url=output,
                            model_name=st.session_state['model'],
                            user_id=st.session_state['user_id']
                        )

                        st.session_state.history.append({
                            "prompt": final_prompt,
                            "images": output if isinstance(output, list) else [output]
                        })
                except Exception as e:
                    st.error(f"❌ Une erreur est survenue : {e}")
                    log_interaction(
                        prompt=final_prompt,
                        parameters=st.session_state['settings'],
                        output_url=f"Error: {e}",
                        model_name=st.session_state['model'],
                        user_id=st.session_state['user_id']
                    )

    if st.button("🔄 Réinitialiser"):
        reset_inputs()
        st.success("✅ Tous les champs ont été réinitialisés.")

    if st.session_state.history:
        st.header("🗂️ Historique des Générations")
        st.markdown("Retrouvez ci-dessous les images que vous avez générées précédemment.")

        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Génération {len(st.session_state.history) - i} : {item['prompt'][:50]}..."):
                st.write(f"**Prompt utilisé :** {item['prompt']}")
                for j, img_url in enumerate(item['images']):
                    st.image(img_url, caption=f"Image {j+1}", use_column_width=True)

        st.markdown("**Note :** L'historique est stocké localement dans votre session et sera perdu si vous rafraîchissez la page.")

# Charger le CSS personnalisé
load_custom_css()

# Replicate API token
#REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]


if not REPLICATE_API_TOKEN:
    st.error("Le token API Replicate n'est pas configuré. Veuillez le définir dans les variables d'environnement.")
    st.stop()

# Initialize the Replicate API client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)


# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())
if 'model' not in st.session_state:
    st.session_state['model'] = 'flux-pro' #set here default model to flux pro
if 'settings' not in st.session_state:
    st.session_state['settings'] = {}

# Initialize Flux Prompt Enhancer
@st.cache_resource
def get_flux_enhancer():
    try:
        enhancer = pipeline("text2text-generation", model="gokaygokay/Flux-Prompt-Enhance")
        return enhancer
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du Flux Prompt Enhancer: {e}")
        return None

enhancer = pipeline("text2text-generation", model=FLUX_ENHANCER_MODEL)

if 'flux_enhancer' not in st.session_state:
    st.session_state.flux_enhancer = get_flux_enhancer()

def load_model_configs(directory="models_configs"):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to your models_configs folder
    models_config_dir = os.path.join(current_dir, directory)

    models = {}
    for filename in os.listdir(models_config_dir):
        if filename.endswith(".json"):
            with open(os.path.join(models_config_dir, filename), "r") as f:
                model_name = filename[:-5]  # Remove .json extension
                config = json.load(f)
                models[model_name] = {
                    "model_version": config.get("model_version"),
                    "trigger_word": config.get("trigger_word"),
                    "description": config.get("description", ""),  # Add this line
                    "input_schema": config.get("input_schema", {})
                }

                # If input_schema is not provided, try to fetch it from the API
                if not models[model_name]["input_schema"] and "model_version" in config:
                    try:
                        api_url = f"https://api.replicate.com/v1/models/{config['model_version']}"
                        headers = {'Authorization': f'Token {REPLICATE_API_TOKEN}'}
                        response = requests.get(api_url, headers=headers)
                        if response.status_code == 200:
                            schema_data = response.json().get("schema", {})
                            models[model_name]["input_schema"] = schema_data.get("input", {})
                    except Exception as e:
                        print(f"Error fetching schema for {model_name}: {e}")

    return models

# Load model configurations
available_models = load_model_configs()

def reset_inputs():
    st.session_state.history = []
    st.session_state.prompt_history = []
    st.session_state['settings'] = {}

def enhance_prompt(prompt):
    if st.session_state.flux_enhancer is not None:
        prefix = "enhance prompt: "
        enhanced = st.session_state.flux_enhancer(prefix + prompt, max_length=256)[0]['generated_text']
        return enhanced.strip()
    else:
        return prompt

def log_interaction(prompt, parameters, output_url, model_name, user_id=None):
    log_file = 'user_interactions.csv'
    file_exists = os.path.isfile(log_file)

    data_row = [
        datetime.utcnow().isoformat(),
        user_id or '',
        model_name,
        prompt,
        json.dumps(parameters),
        output_url if isinstance(output_url, str) else json.dumps(output_url),
        json.dumps(st.session_state.prompt_history)
    ]

    with open(log_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['timestamp', 'user_id', 'model_name', 'prompt', 'parameters', 'output_url', 'prompt_history'])
        writer.writerow(data_row)

def display_header():
    st.title("Alfred - Assistant de Génération d'Images")
    st.write("""
    Bienvenue sur Alfred, votre assistant personnel pour la génération d'images à l'aide de l'intelligence artificielle.
    Créez des images uniques en saisissant un prompt, en sélectionnant un modèle, et en ajustant les paramètres selon vos besoins.
    """)
    st.info("**Conseil :** Pour de meilleurs résultats, décrivez clairement l'image que vous souhaitez générer. Vous pouvez également utiliser le bouton 'Raffiner le prompt' pour améliorer votre description.")

def construct_typography_instruction(options):
    return f" with typography styles: {', '.join(options)}" if options else ""

def construct_final_prompt(base_prompt, text, typography_styles):
    if text.strip():
        typography_instruction = construct_typography_instruction(typography_styles)
        return f"{base_prompt}. Add the text '{text}' written in a legible way on the image{typography_instruction}."
    else:
        return base_prompt

def render_sidebar():
    with st.sidebar:
        st.header("🧠 Sélection du Modèle")
        model_names = list(available_models.keys())
        default_model = st.session_state.get('model', model_names[0])

        model = st.selectbox(
            "Sélectionnez un Modèle",
            options=model_names,
            index=model_names.index(default_model) if default_model in model_names else 0,
            help="Choisissez le modèle d'IA à utiliser pour générer l'image."
        )
        st.session_state['model'] = model

        # Display the model description
        model_info = available_models[model]
        description = model_info.get("description", "")
        if description:
            st.markdown(f"**Description du modèle :** {description}")

        # Display trigger word message if applicable
        trigger_word = model_info.get("trigger_word")
        if trigger_word:
            st.warning(f"Ce modèle nécessite le mot déclencheur : '{trigger_word}'. Assurez-vous de l'inclure dans votre prompt.")
            st.text(f"Exemple: {trigger_word}, a beautiful sunset over the ocean")

        render_main_settings()
        with st.expander("Paramètres Avancés"):
            render_advanced_settings()

def render_parameter(param, schema, settings):
    title = schema.get("title", param.replace("_", " ").capitalize())
    description = schema.get("description", "")
    param_type = schema.get("type")

    if "enum" in schema:
        settings[param] = st.sidebar.selectbox(title, schema["enum"], index=0, help=description)
    elif param_type == "integer":
        min_value = schema.get("minimum", 0)
        max_value = schema.get("maximum", 100)
        default_value = schema.get("default", min_value)
        settings[param] = st.sidebar.slider(title, int(min_value), int(max_value), int(default_value), step=1, help=description)
    elif param_type == "number":
        min_value = schema.get("minimum", 0.0)
        max_value = schema.get("maximum", 1.0)
        default_value = schema.get("default", min_value)
        step = (max_value - min_value) / 100  # Adjust step size based on range
        settings[param] = st.sidebar.slider(title, float(min_value), float(max_value), float(default_value), step=float(step), help=description)
    elif param_type == "boolean":
        settings[param] = st.sidebar.checkbox(title, schema.get("default", False), help=description)
    elif param_type == "string":
        if schema.get("format") == "uri":
            settings[param] = st.sidebar.text_input(title, "", help=description)
        else:
            settings[param] = st.sidebar.text_input(title, "", help=description)

def render_main_settings():
    st.sidebar.subheader("Paramètres Principaux")
    settings = {}
    model_info = available_models[st.session_state['model']]
    main_params = ["aspect_ratio", "output_format", "output_quality", "num_outputs"]

    for param in main_params:
        if param in model_info["input_schema"].get("properties", {}):
            render_parameter(param, model_info["input_schema"]["properties"][param], settings)

    st.session_state['settings'] = settings

def render_advanced_settings():
    settings = st.session_state.get('settings', {})
    model_info = available_models[st.session_state['model']]
    advanced_params = ["seed", "guidance_scale", "num_inference_steps", "lora_scale"]

    for param in advanced_params:
        if param in model_info["input_schema"].get("properties", {}):
            render_parameter(param, model_info["input_schema"]["properties"][param], settings)

    st.session_state['settings'].update(settings)

def generate_image(prompt, model, settings):
    try:
        model_info = available_models[model]
        model_version = model_info["model_version"]
        trigger_word = model_info.get("trigger_word")

        if trigger_word and trigger_word not in prompt:
            prompt = f"{trigger_word}, {prompt}"

        input_params = {"prompt": prompt, **settings}
        input_params = {k: v for k, v in input_params.items() if v is not None}

        output = replicate_client.run(
            model_version,
            input=input_params
        )
        return output
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'image : {e}")
        return None


if __name__ == "__main__":
    main()
