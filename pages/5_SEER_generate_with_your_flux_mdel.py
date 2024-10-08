# File: pages/2_Generate.py

import streamlit as st
import replicate
import os
import csv
import uuid
import json
from datetime import datetime
from transformers import pipeline

# Initialize Replicate client with API token
REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Function to get public models owned by a specific user
def get_public_models(username="aximande"):
    try:
        public_models = []
        for page in replicate.paginate(replicate_client.models.list):
            for model in page:
                if model.owner == username:
                    public_models.append(model)
        model_list = [f"{model.owner}/{model.name}" for model in public_models]
        return model_list
    except Exception as e:
        st.error(f"Error fetching public models: {e}")
        return []

# Function to load model configurations (private and public)
def load_model_configs(directory="models_configs"):
    import os
    import json

    models = {}

    # Load private models from the local configs
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r") as f:
                    model_name = filename[:-5]  # Remove .json extension
                    config = json.load(f)
                    models[model_name] = {
                        "model_version": config.get("model_version"),
                        "trigger_word": config.get("trigger_word"),
                        "description": config.get("description", ""),
                        "input_schema": config.get("input_schema", {})
                    }

    # Fetch public models from Replicate
    public_model_names = get_public_models(username="aximande")
    for model_full_name in public_model_names:
        try:
            model = replicate_client.models.get(model_full_name)
            latest_version = model.versions.list()[0]
            models[model_full_name] = {
                "model_version": latest_version.id,
                "description": model.description or "",
                "input_schema": latest_version.openapi_schema['components']['schemas']['Input']
            }
        except Exception as e:
            st.warning(f"Could not load public model {model_full_name}: {e}")

    return models

# Load model configurations
available_models = load_model_configs()

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())
if 'model' not in st.session_state:
    st.session_state['model'] = list(available_models.keys())[0] if available_models else None  # Set default model
if 'settings' not in st.session_state:
    st.session_state['settings'] = {}

# Initialize Flux Prompt Enhancer
@st.cache_resource
def get_flux_enhancer():
    try:
        enhancer = pipeline("text2text-generation", model="gokaygokay/Flux-Prompt-Enhance")
        return enhancer
    except Exception as e:
        st.error(f"Error initializing Flux Prompt Enhancer: {e}")
        return None

if 'flux_enhancer' not in st.session_state:
    st.session_state.flux_enhancer = get_flux_enhancer()

def main():
    display_header()
    render_sidebar()

    st.subheader("📝 Enter your prompt")
    prompt = st.text_area(
        "Prompt",
        value=st.session_state.prompt_history[-1] if st.session_state.prompt_history else "",
        height=150,
        placeholder="Enter your description here...",
        help="Provide a detailed description to generate an image."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Prompt"):
            st.session_state.prompt_history.append(prompt)
            st.success("Prompt saved.")
    with col2:
        if st.button("✨ Enhance Prompt"):
            if prompt.strip():
                with st.spinner("Enhancing prompt..."):
                    enhanced_prompt = enhance_prompt(prompt)
                    st.session_state.prompt_history.append(enhanced_prompt)
                    st.success("Prompt enhanced and saved.")
            else:
                st.warning("Please enter a prompt first.")

    if st.session_state.prompt_history:
        editable_prompt = st.text_area(
            "📝 Edit Prompt",
            value=st.session_state.prompt_history[-1],
            height=150,
            help="You can edit the current prompt."
        )
        if st.button("💾 Save Changes"):
            st.session_state.prompt_history.append(editable_prompt)
            st.success("Changes saved.")
    else:
        st.warning("No prompt has been saved yet.")

    show_text_section = st.checkbox("Add text to the image", value=False)

    if show_text_section:
        st.subheader("🖋️ Add Text to the Image")
        text_to_add = st.text_input(
            "✏️ Text to Add",
            value="",
            help="Enter the text you want to integrate into the image. The AI will attempt to add this text to the generated image."
        )
        st.session_state['text_to_add'] = text_to_add

        typography_options_list = [
            "Bold", "Italic", "Underlined", "Shadow", "3D", "Gradient", "Handwritten",
            "Calligraphy", "Graffiti", "Vintage", "Futuristic", "Neon", "Glow",
            "Comic", "Stencil", "Watercolor", "Chalk", "Marker", "Spray Paint"
        ]
        typography_options = st.multiselect(
            "🔠 Typography Options",
            options=typography_options_list,
            help="Select typographic styles to apply to the added text. The more styles you choose, the more complex the rendering."
        )
        st.session_state['typography_options'] = typography_options

    if st.session_state.prompt_history:
        prompt_to_use = st.session_state.prompt_history[-1]
        final_prompt = construct_final_prompt(
            prompt_to_use,
            st.session_state.get('text_to_add', ''),
            st.session_state.get('typography_options', [])
        )
        st.subheader("Final Prompt:")
        st.info(final_prompt)
    else:
        final_prompt = None
        st.warning("Please enter a prompt to continue.")

    if st.button("🚀 Generate Image"):
        if not final_prompt:
            st.error("❌ Please enter a valid prompt.")
        elif not st.session_state['model']:
            st.error("❌ Please select a model.")
        else:
            with st.spinner("🖼️ Generating image..."):
                try:
                    output = generate_image(final_prompt, st.session_state['model'], st.session_state['settings'])
                    if output:
                        st.success("✅ Image(s) generated successfully!")
                        if isinstance(output, list):
                            for idx, img_url in enumerate(output):
                                st.image(img_url, use_column_width=True, caption=f"Image {idx+1}")
                        else:
                            st.image(output, use_column_width=True, caption="Generated Image")

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
                    st.error(f"❌ An error occurred: {e}")
                    log_interaction(
                        prompt=final_prompt,
                        parameters=st.session_state['settings'],
                        output_url=f"Error: {e}",
                        model_name=st.session_state['model'],
                        user_id=st.session_state['user_id']
                    )

    if st.button("🔄 Reset"):
        reset_inputs()
        st.success("✅ All fields have been reset.")

    if st.session_state.history:
        st.header("🗂️ Generation History")
        st.markdown("Find below the images you have previously generated.")

        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Generation {len(st.session_state.history) - i} : {item['prompt'][:50]}..."):
                st.write(f"**Prompt used:** {item['prompt']}")
                for j, img_url in enumerate(item['images']):
                    st.image(img_url, caption=f"Image {j+1}", use_column_width=True)

        st.markdown("**Note:** The history is stored locally in your session and will be lost if you refresh the page.")

def display_header():
    st.title("Image Generation Assistant")
    st.write("""
    Welcome to the Image Generation Assistant. Create unique images by entering a prompt, selecting a model, and adjusting parameters according to your needs.
    """)
    st.info("**Tip:** For best results, clearly describe the image you want to generate. You can also use the 'Enhance Prompt' button to improve your description.")

def render_sidebar():
    with st.sidebar:
        st.header("🧠 Model Selection")
        model_names = list(available_models.keys())
        default_model = st.session_state.get('model', model_names[0] if model_names else None)

        if not model_names:
            st.error("No models available. Please ensure you have models configured locally or created on Replicate.")
            st.stop()

        model = st.selectbox(
            "Select a Model",
            options=model_names,
            index=model_names.index(default_model) if default_model in model_names else 0,
            help="Choose the AI model to use for generating the image."
        )
        st.session_state['model'] = model

        # Display the model description
        model_info = available_models[model]
        description = model_info.get("description", "")
        if description:
            st.markdown(f"**Model Description:** {description}")

        # Display trigger word message if applicable
        trigger_word = model_info.get("trigger_word")
        if trigger_word:
            st.warning(f"This model requires the trigger word: '{trigger_word}'. Make sure to include it in your prompt.")
            st.text(f"Example: {trigger_word}, a beautiful sunset over the ocean")

        render_main_settings()
        with st.expander("Advanced Settings"):
            render_advanced_settings()

def render_main_settings():
    st.sidebar.subheader("Main Parameters")
    settings = {}
    model_info = available_models[st.session_state['model']]
    input_schema = model_info["input_schema"]
    properties = input_schema.get("properties", {})
    required_params = input_schema.get("required", [])

    # Define main parameters to display (customize as needed)
    main_params = ["prompt_strength", "num_outputs", "image_dimensions", "style_preset"]

    for param in main_params:
        if param in properties:
            render_parameter(param, properties[param], settings)

    st.session_state['settings'] = settings

def render_advanced_settings():
    settings = st.session_state.get('settings', {})
    model_info = available_models[st.session_state['model']]
    input_schema = model_info["input_schema"]
    properties = input_schema.get("properties", {})
    required_params = input_schema.get("required", [])

    # Define advanced parameters to display (customize as needed)
    advanced_params = [param for param in properties if param not in st.session_state['settings']]

    for param in advanced_params:
        render_parameter(param, properties[param], settings)

    st.session_state['settings'].update(settings)

def render_parameter(param, schema, settings):
    title = schema.get("title", param.replace("_", " ").capitalize())
    description = schema.get("description", "")
    param_type = schema.get("type")

    if "enum" in schema:
        settings[param] = st.sidebar.selectbox(title, schema["enum"], help=description)
    elif param_type == "integer":
        min_value = schema.get("minimum", 0)
        max_value = schema.get("maximum", 100)
        default_value = schema.get("default", min_value)
        settings[param] = st.sidebar.slider(
            title,
            int(min_value),
            int(max_value),
            int(default_value),
            step=1,
            help=description
        )
    elif param_type == "number":
        min_value = schema.get("minimum", 0.0)
        max_value = schema.get("maximum", 1.0)
        default_value = schema.get("default", min_value)
        step = (max_value - min_value) / 100  # Adjust step size based on range
        settings[param] = st.sidebar.slider(
            title,
            float(min_value),
            float(max_value),
            float(default_value),
            step=float(step),
            help=description
        )
    elif param_type == "boolean":
        settings[param] = st.sidebar.checkbox(title, schema.get("default", False), help=description)
    elif param_type == "string":
        if schema.get("format") == "uri":
            settings[param] = st.sidebar.text_input(title, "", help=description)
        else:
            settings[param] = st.sidebar.text_input(title, "", help=description)

def reset_inputs():
    st.session_state.history = []
    st.session_state.prompt_history = []
    st.session_state['settings'] = {}

def enhance_prompt(prompt):
    if st.session_state.flux_enhancer is not None:
        try:
            prefix = "enhance prompt: "
            enhanced = st.session_state.flux_enhancer(prefix + prompt, max_length=256)[0]['generated_text']
            return enhanced.strip()
        except Exception as e:
            st.error(f"Error enhancing prompt: {e}")
            return prompt
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

def construct_typography_instruction(options):
    return f" with typography styles: {', '.join(options)}" if options else ""

def construct_final_prompt(base_prompt, text, typography_styles):
    if text.strip():
        typography_instruction = construct_typography_instruction(typography_styles)
        return f"{base_prompt}. Add the text '{text}' written in a legible way on the image{typography_instruction}."
    else:
        return base_prompt

def generate_image(prompt, model, settings):
    try:
        model_info = available_models[model]
        model_version = model_info["model_version"]
        trigger_word = model_info.get("trigger_word")

        if trigger_word and trigger_word not in prompt:
            prompt = f"{trigger_word}, {prompt}"

        input_params = {**settings}
        # Ensure 'prompt' is included in the input parameters
        input_params['prompt'] = prompt
        input_params = {k: v for k, v in input_params.items() if v is not None}

        output = replicate_client.run(
            model_version,
            input=input_params
        )
        return output
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

if __name__ == "__main__":
    main()
