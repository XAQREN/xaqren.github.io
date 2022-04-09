---
layout: post
title: How to containerize huggingface transformers inference API on your own. 
---

Before going into the details I want to make a quick summary of the post and what you can expect without reading the entire article.

*Since transformer models are widely used for various machine learning tasks nowadays, we will use a pretrained transformers model
from huggingface and build an inference API and deliver as a Docker image.*

*If you are looking to deploy your own transformer model make sure to upload the model weights to the huggingface repository and you are good to go.*

**Pre-requisites**

* Huggingface account (For uploading the model weights if you want to use a custom model)
* Docker 
* Python

**Tools and libraries**

* torch==1.10.2
* transformers==4.18.0
* Flask==2.0.2 

*for this demo we will be doing a text-classification task (sentiment-analysis)*

The default model for sentiment analysis in huggingface pipeline api is "distilbert-base-uncased-finetuned-sst-2-english".
Although this model have some controversies due to its biased predictions, but for our case its a good starting point.

Lets start by creating a minimal flask application.

*included a sample code so that anyone can customize to their needs.*

`app.py`

~~~
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_restx import Resource, Api


app = Flask(__name__)
api = Api(app)
#Loading the default model for sentiment 
print("Model is loading ...")
classifier = pipeline('sentiment-analysis')
print("Model loading complete")
@api.route("/sentiment", methods=['GET', 'POST'])
class SentmentAnalyser(Resource):
    def get(self):
        return jsonify({"message":"Welcome to sentiment analysier"})
    def post(self):
        text = request.get_json()['text']
        sentiment = classifier(text)
        return jsonify(sentiment)


if __name__ == "__main__":
    app.run(debug=False)
~~~

`Dockerfile`

~~~
#Author:sreehari
FROM python:3.8-slim-buster

WORKDIR /sentment_new

COPY requirements.txt .
RUN : \
    && python3 -m venv /venv \
    && pip install --default-timeout=1000 -r requirements.txt 

COPY . .

EXPOSE 5001
RUN python3 -c "from transformers import pipeline; pipeline('sentiment-analysis')"

CMD ["gunicorn" "-w" "4" "-b" "127.0.0.1:4000" "sentment_new:app"] 
~~~

