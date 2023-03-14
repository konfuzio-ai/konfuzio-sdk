"""Test code examples for using the token in the documentation."""
import requests
from konfuzio_sdk import KONFUZIO_TOKEN
from konfuzio_sdk.api import konfuzio_session

url = "https://app.konfuzio.com/api/v3/auth/"

payload = {"username": "example@example.org", "password": "examplepassword"}

response = requests.post(url, json=payload)
assert response.json()['non_field_errors'] == ['Unable to log in with provided credentials.']

headers = {"Authorization": "Token bf20d992c0960876157b53745cdd86fad95e6ff4"}

response = requests.get(url, headers=headers)
assert response.json()['detail'] == 'Invalid token.'

# if you ran konfuzio_sdk init, you can run konfuzio_session() without explicitly specifying the token
session = konfuzio_session(KONFUZIO_TOKEN)
url = "https://app.konfuzio.com/api/v3/projects/"
response = session.get(url)

print(response.json())
assert response.json()['results'][0]['name'] == 'Konfuzio Main Model'
