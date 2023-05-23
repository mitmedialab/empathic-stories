import sys
sys.path.append("./")
sys.path.append("../")

from firebase_admin import db
import firebase_admin
import random
import json
from datetime import datetime
import faiss 
from flask import Flask, abort, request
from flask_cors import CORS
import logging
import threading
from multiprocessing import Lock
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from passwords import open_ai_api_key
from numpy import dot
from numpy.linalg import norm
import numpy as np
import torch
from trainer import EmpathicSimilarityModel
from transformers import AutoTokenizer

openai.api_key = open_ai_api_key

model_SBERT = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
model_BART = EmpathicSimilarityModel.load_from_checkpoint("./models/EmpathicStoriesBART.ckpt", model="BART", pooling="MEAN", bin=False, learning_rate = 5e-6)

tokenizer_BART = AutoTokenizer.from_pretrained("facebook/bart-base")
tokenizer_SBERT =  model_SBERT.tokenizer

df_clean = pd.read_csv("./STORIES (user study).csv")
df_clean =  df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]

embeddings_sbert = np.array([list(eval(_)) for _ in df_clean["embedding (SBERT, no finetuning)"]], dtype=np.float32)
d = len(embeddings_sbert[0])
index_sbert = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
faiss.normalize_L2(embeddings_sbert)
index_sbert.add(embeddings_sbert)

embeddings_bart = np.array([list(eval(_)) for _ in df_clean["embedding (BART, finetuning, MSE, MEAN, no_binning)"]], dtype=np.float32)
d = len(embeddings_bart[0])
index_bart = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
faiss.normalize_L2(embeddings_bart)
index_bart.add(embeddings_bart)

lock = Lock()
app = Flask(__name__)
CORS(app)
logging.getLogger('flask_cors').level = logging.DEBUG

sem = threading.Semaphore()

c = firebase_admin.credentials.Certificate("./credentials.json")
default_app = firebase_admin.initialize_app(c, {
    'databaseURL': "https://empathic-stories-default-rtdb.firebaseio.com/"
})

PILOT_STUDY = True

def get_cosine_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def retrieve_top(embedding, model = "SBERT", k=len(df_clean), embedding_dim = 768):
    embedding = embedding.reshape(1, embedding_dim)
    faiss.normalize_L2(embedding)

    if model == "SBERT":
        D, I = index_sbert.search(embedding, k) 
    else:
        D, I = index_bart.search(embedding, k)
    return D, I

@app.route('/')
def hello_world():
    return "Hello World"


@app.route('/participantIDInput/', methods=["GET", "POST"])
def get_participant_id():
    """Get current session number for participant"""
    try:
        participantIDInput = request.json['participantIDInput']
        print(f'The value of my id is {participantIDInput}')
        ref = db.reference(participantIDInput)
        currentSession = db.reference(participantIDInput + "/currentSession").get()

        if currentSession is None:
            if PILOT_STUDY:
                db.reference(participantIDInput + "/currentSession").set(3) 
            else:
                print("PARTICIPANT SESSION")
                db.reference(participantIDInput + "/currentSession").set(1) 
        elif currentSession not in [1, 2, 3]:
            abort(404)   
        return "success"
    except:
        abort(404)  


def get_stories_from_model(mystory, top_k = 1):
    prompt = f"""Story: 
    {mystory}
    Write a story from your own life that the narrator would empathize with. Do not refer to the narrator explicitly.
    """
    mystory = mystory.replace("\n", "")

    with torch.no_grad():
        embedding_sbert = model_SBERT.encode(mystory)
        embedding_bart = model_BART(mystory)
        embedding_bart = embedding_bart.detach().numpy()
        r1 = df_clean["story_formatted"].iloc[retrieve_top(embedding_bart, model="BART")[1][0][0]]
        r2 = df_clean["story_formatted"].iloc[retrieve_top(embedding_sbert)[1][0][0]]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    r3 = response["choices"][0]["message"]["content"].replace("\n\n", "\n")
    results = {"condition1": r1, "condition2": r2, "condition3": r3}
    return results


@app.route('/sessionDone/', methods=["GET", "POST"])
def sessionDone():
    try:
        id = request.json['participantIDInput']
        currentSession = db.reference(id + "/currentSession").get()
        dict = {'showParticipantID': id, 'showSessionNum': currentSession}
        return json.dumps(dict)
    except:
        abort(404)  
            

@app.route('/getPrompt/', methods=["GET", "POST"])
def getPrompt():
    """Get initial writing prompt for user + retrieve 3 stories from 3 models + save to firebase"""
    try:
        id = request.json['participantIDInput']
        currentSession = db.reference(id + "/currentSession").get()
        dict = {'showParticipantID': id, 'showSessionNum': currentSession}
        session = db.reference(id + '/s00' + str(currentSession))
        session.child("startTime").set(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        
        return json.dumps(dict)
    except:
        abort(404)  


@app.route('/submitMyStory/', methods=["GET", "POST"])
def submitMyStory():
    """Save their story in firebase"""
    try:
        id = request.json['participantIDInput']
        currentSession = db.reference(id + "/currentSession").get()
        if currentSession == 1 or PILOT_STUDY: 
            gender = request.json['gender']
            age = request.json['age']
            race = request.json['race']
            empathyLevel = request.json['empathyLevel']
            demographic = {"gender": gender, "age": age,
                            "race": race, "empathyLevel": empathyLevel}

        valence = request.json['valence']
        arousal = request.json['arousal']
        reflection = {"valence": valence, "arousal": arousal}
        mystory = request.json['mystory']
        fullDate = request.json['fullDate']
        mystoryTopic = request.json['mystoryTopic']
        mainEvent = request.json['mainEvent']
        narratorEmotions = request.json['narratorEmotions']
        moral = request.json['moral']

        mystoryQuestions = {"mainEvent": mainEvent,
                        "narratorEmotions": narratorEmotions, "moral": moral, "fullDate": fullDate}
        ref = db.reference(id)
        session = db.reference(id + '/s00' + str(currentSession))

        session.child("mystory").set(mystory)
        session.child("mystoryTopic").set(mystoryTopic)
        session.child("mystoryQuestions").set(mystoryQuestions)
        session.child("reflection").set(reflection)
        session.child("submitStoryTime").set(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        if currentSession == 1 or PILOT_STUDY:
            session.child("demographic").set(demographic) 


        ## make call to model and save firebase mapping
        stories = get_stories_from_model(mystory)

        session.child("storyMap").set(stories)
        # remove duplicates and randomize, save to firebase with what model it came from, send the 1-4 stories back to frontend to display

        dict = {
            'mystory': mystory
        }

        # check for duplicates
        unique_stories = list(set(stories.values()))
        random.shuffle(unique_stories)
        dict['numOfStories'] = len(unique_stories)
        for i in range(len(unique_stories)):
            dict["story" + str(i + 1)] = unique_stories[i]
        session.child("randomizedStories").set(dict)
    
        return json.dumps(dict)
    except:
        abort(404)  


@app.route('/submitSurveyQuestions/', methods=["GET", "POST"])
def submitSurveyQuestions():
    try:
        id = request.json['participantIDInput']
        currentSession = db.reference(id + "/currentSession").get()
        print(currentSession)
        if currentSession == 3:
            print("i will take the new value of part 6 now")
            empathyWithAI = request.json['empathyWithAI']

        survey1_answers = request.json['survey1_answers']
        survey2_answers = request.json['survey2_answers']
        survey3_answers = request.json['survey3_answers']
        mostEmpathizedOrder = request.json['mostEmpathizedOrder']
        
        feedback = request.json['feedback']

        session = db.reference(id + '/s00' + str(currentSession))
        session.child("endTime").set(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        if currentSession == 1:
            session1 = db.reference(id + '/s001')
            session1.child("feedback").set(feedback)
            session1.child("survey1_answers").set(survey1_answers)
            session1.child("survey2_answers").set(survey2_answers)
            session1.child("survey3_answers").set(survey3_answers)
            session1.child("mostEmpathizedOrder").set(mostEmpathizedOrder)
            db.reference(id + "/currentSession").set(2)

        elif currentSession == 2:
            session2 = db.reference(id + '/s002')
            session2.child("feedback").set(feedback)
            session2.child("survey1_answers").set(survey1_answers)
            session2.child("survey2_answers").set(survey2_answers)
            session2.child("survey3_answers").set(survey3_answers)
            session2.child("mostEmpathizedOrder").set(mostEmpathizedOrder)
            db.reference(id + "/currentSession").set(3)

        elif currentSession == 3:
            print("i will submit everything now")
            session3 = db.reference(id + '/s003')
            session3.child("feedback").set(feedback)
            session3.child("survey1_answers").set(survey1_answers)
            session3.child("survey2_answers").set(survey2_answers)
            session3.child("survey3_answers").set(survey3_answers)
            session3.child("mostEmpathizedOrder").set(mostEmpathizedOrder)
            session3.child("empathyWithAI").set(empathyWithAI)
            db.reference(id + "/currentSession").set(4)
        return 'Data submitted successfully!'
    except:
        abort(404)  

################################### START SERVER ###################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5193, debug=1)
