import requests


predict_url = "http://localhost:8080/predict"

request = {
    "url": "http://bit.ly/mlbookcamp-pants"
}

response = requests.post(predict_url, json=request)
result = response.json()

print(f"Top prediction result: {result['top_class']} ({result['top_probability']:.2%})")
print(f"")