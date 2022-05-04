# Serverless-Faiss-Search
This repository contains a lambda that searches a small faiss index. Below is a short setup to build the application and deploy it to AWS.

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

## Next Steps
The next step of the project would be to scale up the example to an entire DPR index. 
An example of a index that could be used from the pyserini repository can be found [here](https://github.com/castorini/pyserini/blob/master/pyserini/prebuilt_index_info.py#L1369)
Since DPR indices are quite large, the index will need to be partitioned into smaller chunks (perhaps 2-5GB each) and searched in parallel. The results from
all of the partitions would be gathered together and the top results would be returned to the user. An integration test of how an index can be partitioned can be found
[here](https://github.com/castorini/pyserini/pull/1074)


