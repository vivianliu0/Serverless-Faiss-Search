FROM public.ecr.aws/lambda/python:3.8

RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install faiss-cpu transformers==4.6.0

# Copy function code
COPY app.py /var/task/
COPY temp_index/ /var/task/temp_index/
COPY model/ /var/task/model

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]
