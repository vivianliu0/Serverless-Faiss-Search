# Serverless-Faiss-Search

## Requirements
- Python 3.8
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)

## Setup

```bash
# first download the pretrained model, "facebook/dpr-question_encoder-multiset-base"
python save-model.py

# build the application
sam build

# create a new ECR repository
aws ecr create-repository --repository-name search-lambda --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE

# deploy the application with a guided interactive mode
sam deploy -g
# name: search-lambda, choose the correct region
```

To trigger lambda, go to search-lambda in AWS console, click on API Gateway icon, copy API endpoint value and replace the placeholder url in test-search-lambda.py (the POST request endpoint). Save and run:

```bash
python test-search-lambda.py
```

And the output should be like `CACM-2445`
