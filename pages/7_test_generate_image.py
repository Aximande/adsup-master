import streamlit as st
import replicate
import os
import csv
import uuid
import json
from datetime import datetime
from transformers import pipeline

REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@st.cache_resource
def get_flux_enhancer():
    try:
        enhancer = pipeline("text2text-generation", model="gokaygokay/Flux-Prompt-Enhance")
        return enhancer
    except Exception as e:
        st.error(f"Error initializing Flux Prompt Enhancer: {e}")
        return None

def load_model_configs(directory="models_configs"):
    models = {}

    # Load private models from local configs
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r") as f:
                    model_name = filename[:-5]
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

def get_public_models(username="aximande"):
    try:
        public_models = []
        for page in replicate.paginate(replicate_client.models.list):
            for model in page:
                if model.owner == username:
                    public_models.append(model)
        return [f"{model.owner}/{model.name}" for model in public_models]
    except Exception as e:
        st.error(f"Error fetching public models: {e}")
        return []

def enhance_prompt(prompt, enhancer):
    if enhancer is not None:
        try:
            prefix = "enhance prompt: "
            enhanced = enhancer(prefix + prompt, max_length=256)[0]['generated_text']
            return enhanced.strip()
        except Exception as e:
            st.error(f"Error enhancing prompt: {e}")
    return prompt

def generate_image(prompt, model, settings):
    try:
        model_info = available_models[model]
        model_version = model_info["model_version"]
        trigger_word = model_info.get("trigger_word")

        if trigger_word and trigger_word not in prompt:
            prompt = f"{trigger_word}, {prompt}"

        input_params = {**settings, 'prompt': prompt}
        input_params = {k: v for k, v in input_params.items() if v is not None}

        output = replicate_client.run(
            model_version,
            input=input_params
        )
        return output
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

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

def image_generation_page():
    st.title("Image Generation Assistant")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())
    if 'settings' not in st.session_state:
        st.session_state['settings'] = {}

    # Load model configurations
    available_models = load_model_configs()

    # Sidebar for model selection and settings
    st.sidebar.header("🧠 Model Selection")
    model_names = list(available_models.keys())
    selected_model = st.sidebar.selectbox("Select a Model", options=model_names)
    st.session_state['model'] = selected_model

    model_info = available_models[selected_model]
    st.sidebar.markdown(f"**Model Description:** {model_info.get('description', '')}")

    trigger_word = model_info.get("trigger_word")
    if trigger_word:
        st.sidebar.warning(f"This model requires the trigger word: '{trigger_word}'. Include it in your prompt.")

    # Main parameters
    st.sidebar.subheader("Main Parameters")
    settings = {}
    input_schema = model_info["input_schema"]
    properties = input_schema.get("properties", {})

    main_params = ["prompt_strength", "num_outputs", "image_dimensions", "style_preset"]
    for param in main_params:
        if param in properties:
            settings[param] = st.sidebar.slider(
                param.replace("_", " ").capitalize(),
                min_value=properties[param].get("minimum", 0),
                max_value=properties[param].get("maximum", 100),
                value=properties[param].get("default", 50)
            )

    # Advanced parameters
    with st.sidebar.expander("Advanced Settings"):
        for param, schema in properties.items():
            if param not in main_params:
                if schema["type"] == "number":
                    settings[param] = st.slider(
                        param.replace("_", " ").capitalize(),
                        min_value=schema.get("minimum", 0.0),
                        max_value=schema.get("maximum", 1.0),
                        value=schema.get("default", 0.5),
                        step=0.01
                    )
                elif schema["type"] == "integer":
                    settings[param] = st.number_input(
                        param.replace("_", " ").capitalize(),
                        min_value=schema.get("minimum", 0),
                        max_value=schema.get("maximum", 100),
                        value=schema.get("default", 50)
                    )
                elif schema["type"] == "boolean":
                    settings[param] = st.checkbox(param.replace("_", " ").capitalize(), value=schema.get("default", False))
                elif schema["type"] == "string" and "enum" in schema:
                    settings[param] = st.selectbox(param.replace("_", " ").capitalize(), options=schema["enum"])

    st.session_state['settings'] = settings

    # Prompt input and enhancement
    st.subheader("📝 Enter your prompt")
    prompt = st.text_area("Prompt", height=150, placeholder="Enter your description here...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Prompt"):
            st.session_state.prompt_history.append(prompt)
            st.success("Prompt saved.")
    with col2:
        if st.button("✨ Enhance Prompt"):
            if prompt.strip():
                with st.spinner("Enhancing prompt..."):
                    flux_enhancer = get_flux_enhancer()
                    enhanced_prompt = enhance_prompt(prompt, flux_enhancer)
                    st.session_state.prompt_history.append(enhanced_prompt)
                    st.success("Prompt enhanced and saved.")
            else:
                st.warning("Please enter a prompt first.")

    # Display and edit current prompt
    if st.session_state.prompt_history:
        current_prompt = st.text_area("📝 Edit Prompt", value=st.session_state.prompt_history[-1], height=150)
        if st.button("💾 Save Changes"):
            st.session_state.prompt_history.append(current_prompt)
            st.success("Changes saved.")
    else:
        st.warning("No prompt has been saved yet.")

    # Image generation
    if st.button("🚀 Generate Image"):
        if not current_prompt:
            st.error("❌ Please enter a valid prompt.")
        else:
            with st.spinner("🖼️ Generating image..."):
                output = generate_image(current_prompt, selected_model, st.session_state['settings'])
                if output:
                    st.success("✅ Image(s) generated successfully!")
                    if isinstance(output, list):
                        for idx, img_url in enumerate(output):
                            st.image(img_url, use_column_width=True, caption=f"Image {idx+1}")
                    else:
                        st.image(output, use_column_width=True, caption="Generated Image")

                    log_interaction(
                        prompt=current_prompt,
                        parameters=st.session_state['settings'],
                        output_url=output,
                        model_name=selected_model,
                        user_id=st.session_state['user_id']
                    )

                    st.session_state.history.append({
                        "prompt": current_prompt,
                        "images": output if isinstance(output, list) else [output]
                    })

    # Display generation history
    if st.session_state.history:
        st.header("🗂️ Generation History")
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Generation {len(st.session_state.history) - i}: {item['prompt'][:50]}..."):
                st.write(f"**Prompt used:** {item['prompt']}")
                for j, img_url in enumerate(item['images']):
                    st.image(img_url, caption=f"Image {j+1}", use_column_width=True)

        st.info("The history is stored locally in your session and will be lost if you refresh the page.")

if __name__ == "__main__":
    image_generation_page()
