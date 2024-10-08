import streamlit as st
import replicate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_public_models(username="aximande"):
    try:
        public_models = []
        for page in replicate.paginate(replicate.Client().models.list):
            for model in page:
                if model.owner == username:
                    public_models.append(model)
                    logging.info(f"Found public model: {model.owner}/{model.name}")

        if not public_models:
            logging.warning(f"No public models found for user {username}")

        model_dict = {f"{model.owner}/{model.name}": {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "owner": model.owner,
            "visibility": model.visibility,
            "latest_version": model.latest_version
        } for model in public_models}

        return model_dict
    except Exception as e:
        logging.error(f"Error fetching public models: {e}")
        return {}

# Test the function
st.write("Fetching public models...")
public_models = get_public_models()
st.write(f"Found {len(public_models)} public models:")
for model_name, model_info in public_models.items():
    st.write(f"- {model_name}")
    st.json(model_info)
