### General imports ###
from __future__ import division
from turtle import position  # (unused, safe to remove later)
import numpy as np
import pandas as pd
import time
import re
import os
import pickle
from collections import Counter
import altair as alt

### Flask imports ###
from flask import Flask, render_template, session, request, redirect, flash, Response
import requests

### Score / Audio utils ###
from pydub.silence import split_on_silence
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import pydub
import speech_recognition as sr

# ✅ PDF text extraction (replaces Tika)
import fitz  # PyMuPDF

# 🔒 Recommend moving secrets to env vars
# GOOGLE_KEY = os.environ.get("GOOGLE_KEY", "")
GOOGLE_KEY = "PASTE_YOUR_KEY_HERE_IF_YOU_MUST"
lang = "pl"

### Audio imports ###
from library.speech_emotion_recognition import *

### Text imports ###
from library.text_emotion_recognition import *
from library.text_preprocessor import *
from nltk import *
from werkzeug.utils import secure_filename
import tempfile


# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'


### HOME ###
@app.route('/')
def home():
    return render_template('home.html')


### INDEX ###
@app.route('/platform', methods=['POST', 'GET'])
def platform():
    return render_template('platform.html')


### ABOUT ###
@app.route('/about')
def about():
    return render_template('about.html')


### BLOGS ###
@app.route('/blogs')
def blogs():
    return render_template('blogs.html')


@app.route('/blog_content')
def blogs_content():
    return render_template('blogs_content.html')


################################################################### Interview Scorer #####################################################################
global name
global duration
global job_position
global text


# Score Home page
@app.route('/score', methods=['POST', 'GET'])
def score():
    return render_template('scorehome.html')


######### Interview text #############
@app.route('/interview_text', methods=['POST', 'GET'])
def interview_text():
    global duration
    global name
    global job_position
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        job_position = request.form['position']
        duration = int(request.form['duration'])
    return render_template('interview_text.html')


def get_personality(text):
    try:
        pred = predict().run(text, model_name="Personality_traits_NN")
        return pred
    except KeyError:
        return None


def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts


def preprocess_text(text):
    preprocessed_texts = NLTKPreprocessor().transform([text])
    return preprocessed_texts


######### Interview audio #############
@app.route('/interview', methods=['POST', 'GET'])
def interview():
    global text
    print(request.form['answer'])
    text = request.form['answer']

    flash("After pressing the button above, you will get " + str(duration) + " sec to answer the question.")
    return render_template('interview.html', name=name, display_button=False, color='#C7EEFF')


# Audio Recording (Interview)
@app.route('/audio_recording_interview', methods=("POST", "GET"))
def audio_recording_interview():
    global duration
    global text

    SER = speechEmotionRecognition()

    rec_duration = duration  # seconds
    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    flash("Recording over! Evaluate your answer on basis of emotions you expressed.")

    ############################### Conversion to text #########################
    sound = "tmp/voice_recording.wav"
    r = sr.Recognizer()

    with sr.AudioFile(sound) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            text += r.recognize_google(audio)
            print(text)
        except Exception as e:
            print(e)

    return render_template('interview.html', display_button=True, name=name, text=text, color='#00ffad')


# Interview Analysis
@app.route('/interview_analysis', methods=("POST", "GET"))
def interview_analysis():
    global text
    global name
    global job_position

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    SER = speechEmotionRecognition(model_sub_dir)

    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')

    # ✅ RECENT CHANGE: Use non-overlapping windows for short recordings
    sample_rate = 16000
    chunk_size = 49100  # ~3.07 seconds @16kHz
    emotions, timestamp = SER.predict_emotion_from_file(
        rec_sub_dir,
        chunk_step=chunk_size,   # no overlap
        chunk_size=chunk_size,
        sample_rate=sample_rate
    )

    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    major_emotion = max(set(emotions), key=emotions.count)

    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db', 'audio_emotions_dist.txt'), sep=',')

    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")

    major_emotion_other = df_other.EMOTION.mode()[0]

    emotion_dist_other = [
        int(100 * len(df_other[df_other.EMOTION == emotion]) / len(df_other))
        for emotion in SER._emotion.values()
    ]

    df_other_out = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other_out.to_csv(os.path.join('static/js/db', 'audio_emotions_dist_other.txt'), sep=',')

    ############################ for text #############################
    print(text)

    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    df_text = pd.read_csv('static/js/db/text.txt', sep=",")

    new_row = pd.DataFrame([probas], columns=traits)
    df_new = pd.concat([df_text, new_row], ignore_index=True)
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)

    perso = {
        'Extraversion': probas[0],
        'Neuroticism': probas[1],
        'Agreeableness': probas[2],
        'Conscientiousness': probas[3],
        'Openness': probas[4]
    }

    df_text_perso = pd.DataFrame.from_dict(perso, orient='index').reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)

    means = {
        'Extraversion': np.mean(df_new['Extraversion']),
        'Neuroticism': np.mean(df_new['Neuroticism']),
        'Agreeableness': np.mean(df_new['Agreeableness']),
        'Conscientiousness': np.mean(df_new['Conscientiousness']),
        'Openness': np.mean(df_new['Openness'])
    }

    probas_others = [
        np.mean(df_new['Extraversion']),
        np.mean(df_new['Neuroticism']),
        np.mean(df_new['Agreeableness']),
        np.mean(df_new['Conscientiousness']),
        np.mean(df_new['Openness'])
    ]
    probas_others = [int(e * 100) for e in probas_others]

    df_mean = pd.DataFrame.from_dict(means, orient='index').reset_index()
    df_mean.columns = ['Trait', 'Value']
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)

    trait_others = df_mean.loc[df_mean['Value'].idxmax(), 'Trait']

    probas = [int(e * 100) for e in probas]

    session['probas'] = probas
    session['text_info'] = {"common_words": [], "num_words": []}

    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ\n")
        for line in counts:
            d.write(line + "," + str(counts[line]) + "\n")

    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts:
            d.write(line + "," + str(counts[line]) + "\n")

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', on_bad_lines='skip')
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', on_bad_lines='skip')
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    time.sleep(0.5)

    ################ SCORE CALCULATION ##########################
    text_model = pickle.load(open('Models/text_score.sav', 'rb'))
    text_data = probas
    t_score = text_model.predict([text_data])[0]
    print(t_score)

    audio_model = pickle.load(open('Models/audio_score.sav', 'rb'))
    audio_data = emotion_dist
    a_score = audio_model.predict([audio_data])[0]
    print(a_score)

    score = (73.755 * a_score + 26.2445 * t_score) / 100
    score = round(score, 2)
    print(score)

    return render_template(
        'score_analysis.html',
        a_emo=major_emotion,
        a_prob=emotion_dist,
        t_text=text,
        t_traits=probas,
        t_trait=trait,
        t_num_words=num_words,
        t_common_words=common_words_perso,
        name=name,
        position=job_position,
        score=score
    )


######################################################################## AUDIO INTERVIEW ##################################################################

@app.route('/audio', methods=['POST', 'GET'])
def audio_index():
    flash("After pressing the button above, you will get 15sec to answer the question.")
    return render_template('audio.html', display_button=False, color='#C7EEFF')


@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():
    SER = speechEmotionRecognition()

    rec_duration = 16
    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    flash("Recording over! Evaluate your answer on basis of emotions you expressed.")
    return render_template('audio.html', display_button=True, color='#00ffad')


@app.route('/audio_analysis', methods=("POST", "GET"))
def audio_analysis():
    model_sub_dir = os.path.join('Models', 'audio.hdf5')
    SER = speechEmotionRecognition(model_sub_dir)

    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')

    # ✅ RECENT CHANGE: non-overlapping windows
    sample_rate = 16000
    chunk_size = 49100
    emotions, timestamp = SER.predict_emotion_from_file(
        rec_sub_dir,
        chunk_step=chunk_size,
        chunk_size=chunk_size,
        sample_rate=sample_rate
    )

    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    major_emotion = max(set(emotions), key=emotions.count)

    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db', 'audio_emotions_dist.txt'), sep=',')

    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")

    major_emotion_other = df_other.EMOTION.mode()[0]

    emotion_dist_other = [
        int(100 * len(df_other[df_other.EMOTION == emotion]) / len(df_other))
        for emotion in SER._emotion.values()
    ]

    df_other_out = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other_out.to_csv(os.path.join('static/js/db', 'audio_emotions_dist_other.txt'), sep=',')

    time.sleep(0.5)

    return render_template(
        'audio_analysis.html',
        emo=major_emotion,
        emo_other=major_emotion_other,
        prob=emotion_dist,
        prob_other=emotion_dist_other
    )


############################ TEXT INTERVIEW ##############################
global df_text
tempdirectory = tempfile.gettempdir()


@app.route('/text', methods=['POST', 'GET'])
def text():
    return render_template('text.html')


@app.route('/text_analysis', methods=['POST'])
def text_analysis():
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    df_text = pd.read_csv('static/js/db/text.txt', sep=",")

    new_row = pd.DataFrame([probas], columns=traits)
    df_new = pd.concat([df_text, new_row], ignore_index=True)
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)

    perso = {
        'Extraversion': probas[0],
        'Neuroticism': probas[1],
        'Agreeableness': probas[2],
        'Conscientiousness': probas[3],
        'Openness': probas[4]
    }

    df_text_perso = pd.DataFrame.from_dict(perso, orient='index').reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)

    means = {
        'Extraversion': np.mean(df_new['Extraversion']),
        'Neuroticism': np.mean(df_new['Neuroticism']),
        'Agreeableness': np.mean(df_new['Agreeableness']),
        'Conscientiousness': np.mean(df_new['Conscientiousness']),
        'Openness': np.mean(df_new['Openness'])
    }

    probas_others = [
        np.mean(df_new['Extraversion']),
        np.mean(df_new['Neuroticism']),
        np.mean(df_new['Agreeableness']),
        np.mean(df_new['Conscientiousness']),
        np.mean(df_new['Openness'])
    ]
    probas_others = [int(e * 100) for e in probas_others]

    df_mean = pd.DataFrame.from_dict(means, orient='index').reset_index()
    df_mean.columns = ['Trait', 'Value']
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)

    trait_others = df_mean.loc[df_mean['Value'].idxmax(), 'Trait']

    probas = [int(e * 100) for e in probas]

    session['probas'] = probas
    session['text_info'] = {"common_words": [], "num_words": []}

    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ\n")
        for line in counts:
            d.write(line + "," + str(counts[line]) + "\n")

    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts:
            d.write(line + "," + str(counts[line]) + "\n")

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', on_bad_lines='skip')
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', on_bad_lines='skip')
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template(
        'text_analysis.html',
        traits=probas,
        trait=trait,
        trait_others=trait_others,
        probas_others=probas_others,
        num_words=num_words,
        common_words=common_words_perso,
        common_words_others=common_words_others
    )


ALLOWED_EXTENSIONS = set(['pdf'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/text_input', methods=['POST'])
def text_pdf():
    # ✅ Validate file exists
    if 'file' not in request.files:
        flash("No file part in request.")
        return redirect('/text')

    f = request.files['file']
    if f.filename == '':
        flash("No file selected.")
        return redirect('/text')

    if not allowed_file(f.filename):
        flash("Only PDF files are allowed.")
        return redirect('/text')

    # ✅ Save into tmp folder
    os.makedirs("tmp", exist_ok=True)
    filename = secure_filename(f.filename)
    filepath = os.path.join("tmp", filename)
    f.save(filepath)

    # ✅ Extract text from PDF using PyMuPDF (no Java needed)
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        flash(f"Could not read PDF: {e}")
        return redirect('/text')

    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    df_text = pd.read_csv('static/js/db/text.txt', sep=",")

    new_row = pd.DataFrame([probas], columns=traits)
    df_new = pd.concat([df_text, new_row], ignore_index=True)
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)

    perso = {
        'Extraversion': probas[0],
        'Neuroticism': probas[1],
        'Agreeableness': probas[2],
        'Conscientiousness': probas[3],
        'Openness': probas[4]
    }

    df_text_perso = pd.DataFrame.from_dict(perso, orient='index').reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)

    means = {
        'Extraversion': np.mean(df_new['Extraversion']),
        'Neuroticism': np.mean(df_new['Neuroticism']),
        'Agreeableness': np.mean(df_new['Agreeableness']),
        'Conscientiousness': np.mean(df_new['Conscientiousness']),
        'Openness': np.mean(df_new['Openness'])
    }

    probas_others = [
        np.mean(df_new['Extraversion']),
        np.mean(df_new['Neuroticism']),
        np.mean(df_new['Agreeableness']),
        np.mean(df_new['Conscientiousness']),
        np.mean(df_new['Openness'])
    ]
    probas_others = [int(e * 100) for e in probas_others]

    df_mean = pd.DataFrame.from_dict(means, orient='index').reset_index()
    df_mean.columns = ['Trait', 'Value']
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)

    trait_others = df_mean.loc[df_mean['Value'].idxmax(), 'Trait']

    probas = [int(e * 100) for e in probas]

    session['probas'] = probas
    session['text_info'] = {"common_words": [], "num_words": []}

    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ\n")
        for line in counts:
            d.write(line + "," + str(counts[line]) + "\n")

    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts:
            d.write(line + "," + str(counts[line]) + "\n")

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', on_bad_lines='skip')
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', on_bad_lines='skip')
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template(
        'text_analysis.html',
        traits=probas,
        trait=trait,
        trait_others=trait_others,
        probas_others=probas_others,
        num_words=num_words,
        common_words=common_words_perso,
        common_words_others=common_words_others
    )


### RUN APP ###
if __name__ == '__main__':
    app.run(debug=True)
