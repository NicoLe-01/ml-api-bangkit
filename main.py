# README
# Hello everyone, in here I (Kaenova | Bangkit Mentor ML-20) 
# will give you some headstart on createing ML API. 
# Please read every lines and comments carefully. 
# 
# I give you a headstart on text based input and image based input API. 
# To run this server, don't forget to install all the libraries in the
# requirements.txt simply by "pip install -r requirements.txt" 
# and then use "python main.py" to run it
# 
# For ML:
# Please prepare your model either in .h5 or saved model format.
# Put your model in the same folder as this main.py file.
# You will load your model down the line into this code. 
# There are 2 option I give you, either your model image based input 
# or text based input. You need to finish functions "def predict_text" or "def predict_image"
# 
# For CC:
# You can check the endpoint that ML being used, eiter it's /predict_text or 
# /predict_image. For /predict_text you need a JSON {"text": "your text"},
# and for /predict_image you need to send an multipart-form with a "uploaded_file" 
# field. you can see this api documentation when running this server and go into /docs
# I also prepared the Dockerfile so you can easily modify and create a container iamge
# The default port is 8080, but you can inject PORT environement variable.
# 
# If you want to have consultation with me
# just chat me through Discord (kaenova#2859) and arrange the consultation time
#
# Share your capstone application with me! 🥳
# Instagram @kaenovama
# Twitter @kaenovama
# LinkedIn /in/kaenova

## Start your code here! ##

import os
import uvicorn
import traceback
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd


from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array
from typing import List

# Initialize Model
# If you already put yout model in the same folder as this main.py
# You can load .h5 model or any model below this line

# If you use h5 type uncomment line below
model = tf.keras.models.load_model('./models/model_v3.h5')
# If you use saved model type uncomment line below
# model = tf.saved_model.load("./my_model_folder")

# Load Tokenizer
tokenizer = joblib.load("./tokenizer/tokenizer2.pkl")

# Load cosine similarity
cosine_matrix = pd.read_csv('./csv/destination_similarities.csv', header=0, index_col=0)
cosine_matrix = np.array(cosine_matrix)

# Load cleaned dataset
cleaned_df = pd.read_csv('./csv/destination_cleaned.csv')

app = FastAPI()

max_length = 120
trunc_type = 'post'
padding_type = 'post'

dic_label = {
    0: "benteng kuto besak",
    1: "pulau kemaro",
    2: "sungai musi",
    3: "warung terapung",
    4: "riverside restaurant",
    5: "sentral kampung pempek",
    6: "taman purbakala",
    7: "museum monpera",
    8: "museum sultan mahmud II",
    9: "jembatan ampera",
    10: "wisata alam punti kayu",
    11: "taman kambang iwak besak",
    12: "bird park jaka baring",
    13: "jaka baring sport city",
    14: "palembang indah mall",
}


# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# If your model need text input use this endpoint!
class RequestText(BaseModel):
    text:str

class RequestList(BaseModel):
    preferences: List[str]

# Define a Pydantic model for the response
class PredictResponse(BaseModel):
    top_labels: List[str]

@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # In here you will get text sent by the user
        text = req.text
        print("Uploaded text:", text)
        
        # Step 1: (Optional) Do your text preprocessing
        sequence = tokenizer.texts_to_sequences(text)
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=120, truncating=trunc_type, padding=padding_type)

        
        # Step 2: Prepare your data to your model
        predictions = model.predict(padded_sequence)

        # Step 3: Predict the data
        # result = model.predict(...)
        top_n = 3
        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
        print(top_indices)
        
        # Get the corresponding labels
        top_labels = [dic_label.get(index, "Unknown") for index in top_indices]
        print(top_labels)
        
        return PredictResponse(top_labels=top_labels)
        
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

@app.post("/recomendations_destination")
def predict_text(req: RequestList, response: Response):
    try:
        # In here you will get text sent by the user
        text = req.preferences
        print("Uploaded preferences:", text)
        
        if len(text) == 0:
            result = cleaned_df.sort_values(by='rating', ascending=False)
            result = list(result['nama'][:3])
            print(result)
        else:
            filtered_df = cleaned_df[cleaned_df['jenis'].isin(text)]
            
            if filtered_df.empty:
                return []  # If no destinations match the user preferences, return an empty list

            indices = filtered_df.index.tolist()

            idx = indices[0]
            sim_scores = list(enumerate(cosine_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:4]  # Get the top 3 similar destinations (excluding itself)
            destination_indices = [i[0] for i in sim_scores]
            
            result = cleaned_df['nama'].iloc[destination_indices].tolist()
        
        return PredictResponse(top_labels=result)
        
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"



# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8081)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)