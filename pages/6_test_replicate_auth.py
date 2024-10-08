import requests

def test_authentication(api_token):
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    try:
        response = requests.get("https://api.replicate.com/v1/account", headers=headers)
        if response.status_code == 200:
            account_info = response.json()
            st.success(f"✅ Authenticated successfully!\n**Account Type:** {account_info.get('type', 'N/A').capitalize()}\n**Username:** {account_info.get('username', 'N/A')}")
            return True
        elif response.status_code == 401:
            st.error("🔴 Authentication failed: Invalid API token.")
            return False
        else:
            st.error(f"🔴 Authentication failed with status code {response.status_code}: {response.text}")
            return False
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during authentication: {e}")
        return False
