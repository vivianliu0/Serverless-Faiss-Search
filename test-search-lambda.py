import requests

data = {"query": "What is the solution of separable closed queueing networks?"}
retval = requests.post('https://sz6p2apapj.execute-api.us-east-2.amazonaws.com/Prod/search', json=data)
print(retval.text)
