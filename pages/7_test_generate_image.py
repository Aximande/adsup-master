import streamlit as st
import replicate
import os
import json

REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def get_public_models(username="aximande"):
    try:
        public_models = []
        for page in replicate.paginate(replicate_client.models.list):
            for model in page:
                if model.owner == username:
                    public_models.append(model)
        return {f"{model.owner}/{model.name}": model for model in public_models}
    except Exception as e:
        st.error(f"Error fetching public models: {e}")
        return {}

def load_generic_models(directory="models_configs"):
    models = {}
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r") as f:
                    model_name = filename[:-5]
                    config = json.load(f)
                    models[model_name] = config
    return models

def image_generation_page():
    st.title("Image Generation Assistant")

    # Step 1: Choose model type
    model_type = st.radio("Choose model type:",
                          ["Fine-tuned models by SEER", "Generic models"])

    # Step 2: Load and display models based on selection
    if model_type == "Fine-tuned models by SEER":
        available_models = get_public_models()
        if not available_models:
            st.warning("No fine-tuned models available. Please check your connection or try again later.")
    else:
        available_models = load_generic_models()
        if not available_models:
            st.warning("No generic models found. Please check your models_configs directory.")

    # Step 3: Select a model
    if available_models:
        selected_model = st.selectbox("Select a Model", options=list(available_models.keys()))

        # Display model information
        model_info = available_models[selected_model]
        st.write(f"**Model Description:** {model_info.get('description', 'No description available.')}")

        # Here you would continue with the rest of your image generation logic
        # For now, let's just display a placeholder
        st.text_area("Enter your prompt", placeholder="Describe the image you want to generate...")
        st.button("Generate Image", disabled=True)  # Disabled for now

    else:
        st.error("No models available. Please check your configuration and try again.")

if __name__ == "__main__":
    image_generation_page()
