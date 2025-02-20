import requests
from rich import print as rprint
from rich.panel import Panel

url = "http://127.0.0.1:8000/chat"
session_id = '1'
summary = 'The user asked you about Ebla and you answered about its history'
while True:
    message = input('You: ')
    data = {"session_id": session_id,"message": message,"summary": summary}
    response = requests.post(url, json=data)
    summary = response.json().get('summary',summary)
    rprint(Panel('AI: ' + response.json().get('message')))

