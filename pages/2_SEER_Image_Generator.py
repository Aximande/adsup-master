import streamlit as st
import replicate
import os
from dotenv import load_dotenv
from utils import load_custom_css

# Récupérer le token Replicate depuis les variables de streamlit
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

if not REPLICATE_API_TOKEN:
    st.error("Le token API Replicate n'est pas configuré. Veuillez le définir dans les secrets de l'application Streamlit.")
    st.stop()

# Initialiser le client Replicate
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Configuration de la page avec un thème personnalisé via config.toml
st.set_page_config(
    page_title="Alfred - Le 1er Assistant au service de SEER pour la Génération d'Images",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Charger le CSS personnalisé
load_custom_css()

# Initialiser les variables de session
if 'final_prompt' not in st.session_state:
    st.session_state.final_prompt = ""
if 'enhanced_prompt' not in st.session_state:
    st.session_state.enhanced_prompt = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = "Aucun"
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = ""
if 'text_to_add' not in st.session_state:
    st.session_state.text_to_add = ""
if 'typography_options' not in st.session_state:
    st.session_state.typography_options = []
if 'language' not in st.session_state:
    st.session_state.language = "Français"

# Fonction pour réinitialiser tous les champs
def reset_inputs():
    st.session_state.final_prompt = ""
    st.session_state.enhanced_prompt = ""
    st.session_state.selected_prompt = "Aucun"
    st.session_state.custom_prompt = ""
    st.session_state.text_to_add = ""
    st.session_state.typography_options = []
    st.session_state.language = "Français"

# Fonction pour copier le texte dans le presse-papiers via JavaScript
def copy_to_clipboard(text):
    copy_button = f"""
    <div>
        <input type="text" value="{text}" id="copyText" readonly style="position: absolute; left: -9999px;">
        <button onclick="copyFunction()">📋 Copier le Prompt</button>
        <script>
            function copyFunction() {{
                var copyText = document.getElementById("copyText");
                copyText.select();
                copyText.setSelectionRange(0, 99999); /* For mobile devices */
                document.execCommand("copy");
                alert("Prompt copié dans le presse-papiers!");
            }}
        </script>
    </div>
    """
    st.markdown(copy_button, unsafe_allow_html=True)

# Titre de l'application avec logo et lien
def display_header():
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        # Ajouter le logo du client (Ad's up consulting)
        logo_path = "adsup-logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150, use_column_width=False, caption=None)
        else:
            st.warning("Logo Ad's up consulting non trouvé. Assurez-vous que 'adsup-logo.png' est dans le dossier racine du projet.")
    with col_title:
        st.markdown("""
        <h1 style='text-align: center; color: #FFFFFF;'>Alfred - Le 1er Assistant au service de SEER pour la Génération d'Images</h1>
        <p style='text-align: center; color: #555555;'>
            <a href="https://www.ads-up.fr/" target="_blank">Visitez le site web de Ad's up consulting</a>
        </p>
        """, unsafe_allow_html=True)

    # Section introductive
    st.markdown("""
    <div class="section">
        <p style='font-size: 16px; color: #FFFFFF;'>
            Créez des images uniques avec l'intelligence artificielle via le modèle dernière génération FLUX.PRO de Black Forest Labs, mis à disposition de l'équipe SEER d'Ad's up consulting.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Afficher le header
display_header()

# Création de deux colonnes pour le contenu principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Création du Prompt")

    # Sélecteur de langue
    language = st.radio("🌐 Sélectionnez la langue du prompt", ["Français", "Anglais"], index=0)
    st.session_state.language = language

    # Librairie de Prompts
    st.subheader("Choisissez un Prompt ou Entrez le Vôtre")

    predefined_prompts_fr = [
        "Un paysage magnifique avec des montagnes en arrière-plan et un lac serein reflétant le ciel.",
        "Une ville futuriste la nuit avec des néons lumineux et des voitures volantes.",
        "Un astronaute flottant dans l'espace avec une vue sur la Terre derrière lui.",
        "Un portrait rapproché d'une personne avec un maquillage artistique et une coiffe élaborée."
    ]

    predefined_prompts_en = [
        "A beautiful landscape with mountains in the background and a serene lake reflecting the sky.",
        "A futuristic city at night with bright neon lights and flying cars.",
        "An astronaut floating in space with a view of Earth behind him.",
        "A close-up portrait of a person with artistic makeup and an elaborate headdress."
    ]

    predefined_prompts = predefined_prompts_fr if language == "Français" else predefined_prompts_en

    selected_prompt = st.selectbox(
        "📚 Prompts Prédéfinis",
        ["Aucun"] + predefined_prompts,
        index=0,
        help="Sélectionnez un prompt prédéfini pour générer une image correspondant à ce thème."
    )

    # Champ de saisie libre
    custom_prompt = st.text_area(
        "✍️ Ou entrez votre propre prompt",
        height=100,
        placeholder="Entrez votre description ici...",
        help="Saisissez une description détaillée pour générer une image personnalisée."
    )

    # Logique pour déterminer le prompt final
    if selected_prompt != "Aucun" and custom_prompt.strip() != "":
        st.warning("⚠️ Veuillez choisir soit un prompt prédéfini, soit entrer votre propre prompt, pas les deux.")
        st.session_state.final_prompt = ""
    elif selected_prompt != "Aucun":
        st.session_state.final_prompt = selected_prompt
    elif custom_prompt.strip() != "":
        st.session_state.final_prompt = custom_prompt
    else:
        st.session_state.final_prompt = ""

    # Section pour ajouter du texte écrit avec typographie
    st.subheader("🖋️ Ajouter du Texte à l'Image")

    # Option pour le texte à ajouter
    text_to_add = st.text_input(
        "✏️ Texte à Ajouter",
        value="",
        help="Entrez le texte que vous souhaitez ajouter à l'image."
    )

    with st.expander("🎨 Options de Typographie", expanded=False):
        # Typographie Options (Personnalisables)
        typography_options = [
            "Random",
            "Bold",
            "Italic",
            "Underlined",
            "Shadow",
            "3D",
            "Gradient",
            "Handwritten",
            "Calligraphy",
            "Graffiti",
            "Vintage",
            "Futuristic",
            "Neon",
            "Glow",
            "Comic",
            "Stencil",
            "Watercolor",
            "Chalk",
            "Marker",
            "Spray Paint"
        ]

        selected_typography_options = st.multiselect(
            "🔠 Options de Typographie",
            typography_options,
            default=[],
            help="Sélectionnez les options de typographie à appliquer au texte ajouté."
        )

    # Fonction pour construire les instructions de typographie
    def construct_typography_instruction(options):
        if options:
            return f" with typography styles: {', '.join(options)}"
        else:
            return ""

    # Construire l'instruction de typographie
    typography_instruction = construct_typography_instruction(selected_typography_options)

    # Section pour valider le prompt final
    st.subheader("📝 Valider le Prompt Final")

    # Construire le prompt final avec les instructions de typographie
    if st.session_state.final_prompt:
        base_prompt = st.session_state.final_prompt
        if text_to_add.strip():
            if typography_instruction:
                final_prompt_display_with_text = f"{base_prompt}. Add the text '{text_to_add}' written in a legible way on the image{typography_instruction}."
            else:
                final_prompt_display_with_text = f"{base_prompt}. Add the text '{text_to_add}' written in a legible way on the image."
        else:
            final_prompt_display_with_text = base_prompt

        # Champ modifiable pour le prompt final
        final_prompt_editable = st.text_area(
            "🖋️ Prompt Final",
            value=final_prompt_display_with_text,
            height=100,
            help="Modifiez le prompt final ici avant la génération de l'image."
        )

        # Bouton pour copier le prompt
        copy_to_clipboard(final_prompt_editable)
    else:
        st.warning("⚠️ Veuillez sélectionner ou entrer un prompt pour continuer.")
        final_prompt_editable = ""

with col2:
    st.header("⚙️ Paramètres de Génération ⚙️")

    # Modèle à utiliser (fixé à 'black-forest-labs/flux-pro')
    selected_model = "black-forest-labs/flux-pro"

    st.write(f"🧠 Modèle sélectionné: Flux Pro ({selected_model})")

    # Paramètres spécifiques du modèle
    steps = st.slider(
        "🔄 Nombre d'étapes (steps)",
        min_value=1,
        max_value=50,
        value=25,
        step=1,
        help="Nombre d'étapes de diffusion."
    )

    guidance = st.slider(
        "📈 Guidance",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Contrôle l'équilibre entre l'adhésion au prompt et la qualité/diversité de l'image."
    )

    interval = st.slider(
        "⏳ Interval",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Ajuste la variance dans les sorties possibles."
    )

    safety_tolerance = st.slider(
        "🛡️ Safety Tolerance",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Tolérance de sécurité, 1 est le plus strict et 5 le plus permissif."
    )

    # Aspect Ratio
    aspect_ratio = st.selectbox(
        "📐 Aspect Ratio",
        ["1:1", "16:9", "4:3", "custom"],
        index=0,
        help="Ratio d'aspect pour l'image générée."
    )

    # Dimensions personnalisées
    if aspect_ratio == "custom":
        width = st.number_input(
            "📏 Largeur",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Largeur de l'image générée."
        )
        height = st.number_input(
            "📐 Hauteur",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Hauteur de l'image générée."
        )
    else:
        # Définitions des dimensions selon le ratio choisi
        width = None
        height = None

    # Format de l'image de sortie
    output_format = st.selectbox(
        "🖼️ Format de l'image de sortie",
        ["webp", "jpg", "png"],
        index=0,
        help="Format des images générées."
    )

    output_quality = st.slider(
        "🎚️ Qualité de l'image (output_quality)",
        min_value=0,
        max_value=100,
        value=80,
        step=1,
        help="Qualité lors de la sauvegarde des images de sortie, de 0 à 100."
    )

    # Seed optionnelle
    seed_input = st.text_input(
        "🌱 Seed (optionnel)",
        value="",
        help="Seed aléatoire. Définissez-le pour une génération reproductible."
    )
    seed = int(seed_input) if seed_input.strip().isdigit() else None

    # Bouton de génération
    if st.button("🚀 Générer l'Image"):
        if not final_prompt_editable.strip():
            st.error("❌ Veuillez entrer un prompt valide.")
        else:
            with st.spinner("🖼️ Génération en cours..."):
                try:
                    # Préparer les paramètres d'entrée
                    input_params = {
                        "prompt": final_prompt_editable,
                        "steps": steps,
                        "guidance": guidance,
                        "interval": interval,
                        "aspect_ratio": aspect_ratio,
                        "output_format": output_format,
                        "output_quality": output_quality,
                        "safety_tolerance": safety_tolerance,
                    }

                    if seed is not None:
                        input_params["seed"] = seed

                    if aspect_ratio == "custom":
                        if width and height:
                            input_params["width"] = width
                            input_params["height"] = height
                        else:
                            st.error("❌ Veuillez spécifier la largeur et la hauteur pour un aspect ratio personnalisé.")
                            st.stop()

                    # Appeler l'API Replicate
                    output = replicate_client.run(
                        selected_model,
                        input=input_params
                    )

                    # Afficher l'image générée
                    if isinstance(output, str):
                        st.success("✅ Image générée avec succès !")
                        st.image(output, use_column_width=True, caption="🖼️ Image générée")
                        # Ajouter à l'historique
                        st.session_state.history.append({
                            "prompt": final_prompt_editable,
                            "images": [output]
                        })
                    else:
                        st.error("❌ Format de réponse inattendu.")
                except replicate.exceptions.ReplicateError as e:
                    st.error(f"❌ Erreur Replicate : {e}")
                except Exception as e:
                    st.error(f"❌ Une erreur est survenue : {e}")

    # Bouton de réinitialisation
    if st.button("🔄 Réinitialiser"):
        reset_inputs()
        st.success("✅ Tous les champs ont été réinitialisés.")
