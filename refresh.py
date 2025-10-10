import requests
import secret
import base64

class Refresh:
    def __init__(self):
        self.refresh_token = secret.refresh_token
        self.client_id = secret.client_id
        self.client_secret = secret.client_secret

    def refresh(self):
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": "Basic " + base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }

        r = requests.post(url, headers=headers, data=data)
        if r.status_code != 200:
            raise Exception(f"Failed to refresh token: {r.text}")
        access_token = r.json()["access_token"]
        return access_token
