import streamlit as st
import replicate
import os

REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@st.cache_data(ttl=300)
def get_available_models(owner="aximande"):
    all_models = replicate_client.models.list()
    owner_models = [model for model in all_models if model.owner == owner]
    return [f"{model.owner}/{model.name}:{model.latest_version.id}" for model in owner_models]

def image_generation_page():
    st.header("Generate Images with Fine-Tuned FLUX Models")

    # Get available models
    available_models = get_available_models()

    if not available_models:
        st.warning("No models found for the specified owner. Please check your Replicate account.")
        return

    # Select model
    selected_model = st.selectbox("Select a model", available_models)

    # Input fields
    prompt = st.text_area("Enter your prompt")
    negative_prompt = st.text_area("Enter negative prompt (optional)")

    # Model parameters
    model = st.selectbox("Model version", ["dev", "main", "newest"])
    lora_scale = st.slider("LoRA scale", 0.0, 1.0, 1.0, 0.01)
    num_outputs = st.slider("Number of images to generate", 1, 4, 1)
    aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "4:3", "3:4", "16:9", "9:16"])
    output_format = st.selectbox("Output format", ["webp", "png"])
    guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 3.5, 0.1)
    output_quality = st.slider("Output quality", 60, 100, 90)
    prompt_strength = st.slider("Prompt strength", 0.0, 1.0, 0.8, 0.01)
    extra_lora_scale = st.slider("Extra LoRA scale", 0.0, 1.0, 1.0, 0.01)
    num_inference_steps = st.slider("Number of inference steps", 1, 150, 28)

    if st.button("Generate Images"):
        if prompt:
            with st.spinner("Generating images..."):
                input_params = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model": model,
                    "lora_scale": lora_scale,
                    "num_outputs": num_outputs,
                    "aspect_ratio": aspect_ratio,
                    "output_format": output_format,
                    "guidance_scale": guidance_scale,
                    "output_quality": output_quality,
                    "prompt_strength": prompt_strength,
                    "extra_lora_scale": extra_lora_scale,
                    "num_inference_steps": num_inference_steps
                }

                try:
                    output = replicate_client.run(
                        selected_model,
                        input=input_params
                    )

                    # Display generated images
                    if isinstance(output, list):
                        for i, image_url in enumerate(output):
                            st.image(image_url, caption=f"Generated Image {i+1}")
                    else:
                        st.image(output, caption="Generated Image")
                except Exception as e:
                    st.error(f"An error occurred while generating the image: {str(e)}")
        else:
            st.error("Please enter a prompt.")

if __name__ == "__main__":
    image_generation_page()
    