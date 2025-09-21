import requests

url = "https://56fd4fde8117.ngrok-free.app/analyze_image"
files = {"image": open("image.png", "rb")}
data = {"conversation": ""}
response = requests.post(url, files=files, data=data)
print(response.json())