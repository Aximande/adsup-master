import streamlit as st
import replicate
import os
import csv
import uuid
import json
import requests
from datetime import datetime

# Set the Replicate API token
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

# Initialize the Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Default input schema for custom models
default_input_schema = {
    "model": {
        "type": "string",
        "title": "Version du Modèle",
        "description": "Sélectionnez la version du modèle.",
        "enum": ["dev", "schnell"],
        "default": "dev"
    },
    "lora_scale": {
        "type": "number",
        "title": "LoRA Scale",
        "description": "Détermine la force d'application du LoRA principal.",
        "default": 1,
        "minimum": -1,
        "maximum": 2
    },
    "num_outputs": {
        "type": "integer",
        "title": "Nombre d'Images",
        "description": "Nombre d'images à générer.",
        "default": 1,
        "minimum": 1,
        "maximum": 4
    },
    "aspect_ratio": {
        "type": "string",
        "title": "Aspect Ratio",
        "description": "Ratio de l'image générée.",
        "enum": ["1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21", "custom"],
        "default": "1:1"
    },
    "output_format": {
        "type": "string",
        "title": "Format de Sortie",
        "description": "Format des images générées.",
        "enum": ["webp", "jpg", "png"],
        "default": "webp"
    },
    "guidance_scale": {
        "type": "number",
        "title": "Guidance Scale",
        "description": "Scale pour le processus de diffusion. Des valeurs plus faibles peuvent donner des images plus réalistes.",
        "default": 3.5,
        "minimum": 0,
        "maximum": 10
    },
    "output_quality": {
        "type": "integer",
        "title": "Qualité de Sortie",
        "description": "Qualité lors de la sauvegarde des images de sortie, de 0 à 100.",
        "default": 90,
        "minimum": 0,
        "maximum": 100
    },
    "prompt_strength": {
        "type": "number",
        "title": "Puissance du Prompt",
        "description": "Puissance du prompt lors de l'utilisation d'img2img / inpaint.",
        "default": 0.8,
        "minimum": 0,
        "maximum": 1.0
    },
    "extra_lora_scale": {
        "type": "number",
        "title": "Extra LoRA Scale",
        "description": "Détermine la force d'application de l'Extra LoRA.",
        "default": 1,
        "minimum": -1,
        "maximum": 2
    },
    "num_inference_steps": {
        "type": "integer",
        "title": "Nombre d'Étapes d'Inférence",
        "description": "Nombre d'étapes d'inférence. Plus d'étapes peuvent donner des images plus détaillées.",
        "default": 28,
        "minimum": 1,
        "maximum": 50
    },
}

def get_model_versions(model_owner, model_name):
    model_versions = []

    try:
        # List model versions
        model = replicate_client.models.get(f"{model_owner}/{model_name}")
        versions = model.versions.list()

        for version in versions:
            model_versions.append({
                "id": version.id,
                "created_at": str(version.created_at),
                "cog_version": version.cog_version or ""
            })

    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error retrieving model versions: {str(e)}")

    return model_versions

def generate_image(prompt, model_version, settings):
    try:
        input_params = {k: v for k, v in settings.items() if v is not None}
        input_params["prompt"] = prompt

        output = replicate_client.run(
            model_version,
            input=input_params
        )
        return output
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'image : {e}")
        return None

def reset_inputs():
    st.session_state.history = []
    st.session_state.prompt_history = []
    st.session_state['settings'] = {}
    st.session_state['custom_owner'] = ""
    st.session_state['custom_name'] = ""
    st.session_state['custom_versions'] = []
    st.session_state['selected_version'] = ""
    st.session_state['prompt'] = ""
    st.session_state['text_to_add'] = ""
    st.session_state['typography_options'] = []

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
    st.title("Assistant de Génération d'Images avec Modèles Personnalisés")
    st.write("""
    Bienvenue sur cette application dédiée à la génération d'images à l'aide de vos modèles personnalisés.
    Saisissez un prompt, sélectionnez votre modèle personnalisé, et ajustez les paramètres selon vos besoins.
    """)
    st.info("**Conseil :** Pour de meilleurs résultats, décrivez clairement l'image que vous souhaitez générer.")

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
        st.header("🧠 Sélection du Modèle Personnalisé")

        # Custom model input fields
        st.session_state['custom_owner'] = st.text_input(
            "Nom du Propriétaire du Modèle",
            value=st.session_state.get('custom_owner', ''),
            help="Entrez le nom du propriétaire du modèle sur Replicate."
        )
        st.session_state['custom_name'] = st.text_input(
            "Nom du Modèle",
            value=st.session_state.get('custom_name', ''),
            help="Entrez le nom du modèle que vous souhaitez utiliser sur Replicate."
        )

        if st.button("Rechercher les Versions du Modèle"):
            owner = st.session_state.get('custom_owner', '').strip()
            name = st.session_state.get('custom_name', '').strip()

            if not owner or not name:
                st.error("Veuillez fournir à la fois le nom du propriétaire et le nom du modèle.")
            else:
                versions = get_model_versions(owner, name)
                if versions:
                    st.session_state['custom_versions'] = versions
                    version_display = [f"ID: {v['id']} - Créé le: {v['created_at']}" for v in versions]
                    st.session_state['selected_version'] = st.selectbox(
                        "Sélectionnez une Version du Modèle",
                        options=version_display,
                        help="Choisissez la version du modèle que vous souhaitez utiliser."
                    )
                    st.success(f"Modèle `{owner}/{name}` chargé avec {len(versions)} versions disponibles.")
                else:
                    st.session_state['custom_versions'] = []
                    st.session_state['selected_version'] = ""
                    st.error("Aucune version trouvée pour le modèle spécifié.")

        # Render main and advanced settings
        render_main_settings()
        with st.expander("Paramètres Avancés"):
            render_advanced_settings()

def render_parameter(param, schema, settings):
    if not isinstance(schema, dict):
        st.warning(f"Le paramètre '{param}' n'a pas de schéma défini correctement.")
        return

    title = schema.get("title", param.replace("_", " ").capitalize())
    description = schema.get("description", "")
    param_type = schema.get("type")

    if "enum" in schema:
        settings[param] = st.sidebar.selectbox(title, schema["enum"], index=0, help=description)
    elif param_type == "integer":
        min_value = schema.get("minimum", 0)
        max_value = schema.get("maximum", 10000)
        default_value = schema.get("default", min_value)
        settings[param] = st.sidebar.slider(title, int(min_value), int(max_value), int(default_value), step=1, help=description)
    elif param_type == "number":
        min_value = schema.get("minimum", 0.0)
        max_value = schema.get("maximum", 100.0)
        default_value = schema.get("default", min_value)
        step = (max_value - min_value) / 100 if max_value > min_value else 0.1
        settings[param] = st.sidebar.slider(title, float(min_value), float(max_value), float(default_value), step=step, help=description)
    elif param_type == "boolean":
        settings[param] = st.sidebar.checkbox(title, schema.get("default", False), help=description)
    elif param_type == "string":
        if schema.get("format") == "uri":
            settings[param] = st.sidebar.text_input(title, "", help=description)
        else:
            settings[param] = st.sidebar.text_input(title, schema.get("default", ""), help=description)

def render_main_settings():
    st.sidebar.subheader("Paramètres Principaux")
    settings = st.session_state.get('settings', {})

    # Use default input schema
    model_input_schema = default_input_schema

    # Exclude 'prompt' from main parameters since it will be in the main content area
    main_params = ["model", "lora_scale", "num_outputs", "aspect_ratio", "output_format",
                   "guidance_scale", "output_quality", "prompt_strength", "extra_lora_scale", "num_inference_steps"]

    for param in main_params:
        schema_param = model_input_schema.get(param)
        if isinstance(schema_param, dict):
            render_parameter(param, schema_param, settings)
        else:
            st.warning(f"Le paramètre '{param}' n'est pas disponible dans le schéma par défaut.")

    # Update st.session_state['settings']
    st.session_state['settings'] = settings

def render_advanced_settings():
    st.sidebar.subheader("Paramètres Avancés")
    settings = st.session_state.get('settings', {})

    model_input_schema = default_input_schema

    # Exclude parameters already in settings and 'prompt'
    advanced_params = [param for param in model_input_schema.keys() if param not in settings and param != 'prompt']

    for param in advanced_params:
        schema_param = model_input_schema.get(param)
        if isinstance(schema_param, dict):
            render_parameter(param, schema_param, settings)
        else:
            st.warning(f"Le paramètre '{param}' n'est pas disponible dans le schéma par défaut.")

    # Update st.session_state['settings']
    st.session_state['settings'] = settings

def main():
    display_header()
    render_sidebar()

    st.subheader("📝 Entrée du Prompt")
    prompt = st.text_area(
        "Entrez votre prompt ici",
        value=st.session_state.get('prompt', ''),
        height=150,
        placeholder="Décrivez l'image que vous souhaitez générer...",
        help="Saisissez une description détaillée pour générer une image."
    )
    st.session_state['prompt'] = prompt  # Save the prompt in session state

    if st.button("💾 Enregistrer le prompt"):
        st.session_state.prompt_history.append(prompt)
        st.success("Prompt enregistré.")

    if st.session_state.prompt_history:
        editable_prompt = st.text_area(
            "📝 Modifier le prompt",
            value=st.session_state.prompt_history[-1],
            height=150,
            help="Vous pouvez modifier le prompt actuel."
        )
        if st.button("💾 Enregistrer les modifications"):
            st.session_state.prompt_history.append(editable_prompt)
            st.session_state['prompt'] = editable_prompt  # Update the prompt in session state
            st.success("Modifications enregistrées.")
    else:
        st.warning("Aucun prompt n'a été enregistré pour le moment.")

    show_text_section = st.checkbox("Ajouter du texte à l'image", value=False)

    if show_text_section:
        st.subheader("🖋️ Ajouter du Texte à l'Image")
        text_to_add = st.text_input(
            "✏️ Texte à Ajouter",
            value=st.session_state.get('text_to_add', ''),
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

    # Construct the final prompt
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
        elif not st.session_state.get('selected_version'):
            st.error("❌ Veuillez sélectionner une version du modèle personnalisé.")
        else:
            with st.spinner("🖼️ Génération en cours..."):
                try:
                    selected_version_str = st.session_state['selected_version']
                    version_id = selected_version_str.split("ID: ")[1].split(" - Créé le:")[0]
                    owner = st.session_state['custom_owner']
                    name = st.session_state['custom_name']
                    model_version = f"{owner}/{name}:{version_id}"
                    model_name_display = f"{owner}/{name}"

                    # Set the prompt in settings
                    st.session_state['settings']['prompt'] = final_prompt

                    output = generate_image(final_prompt, model_version, st.session_state['settings'])
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
                            model_name=model_name_display,
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
                        model_name=model_name_display if 'model_name_display' in locals() else "Unknown",
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

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())
if 'settings' not in st.session_state:
    st.session_state['settings'] = {}
if 'custom_owner' not in st.session_state:
    st.session_state['custom_owner'] = ""
if 'custom_name' not in st.session_state:
    st.session_state['custom_name'] = ""
if 'custom_versions' not in st.session_state:
    st.session_state['custom_versions'] = []
if 'selected_version' not in st.session_state:
    st.session_state['selected_version'] = ""
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = ""
if 'text_to_add' not in st.session_state:
    st.session_state['text_to_add'] = ""
if 'typography_options' not in st.session_state:
    st.session_state['typography_options'] = []

if __name__ == "__main__":
    main()
