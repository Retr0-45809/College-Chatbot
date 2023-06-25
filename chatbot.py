import nltk
import numpy as np
import json  
import random
import spacy
import torch
import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast

bert_model = BertForQuestionAnswering.from_pretrained("bert-model")
tokenizer = BertTokenizerFast.from_pretrained("bert-model")

app = Flask(__name__)
CORS(app)

def load_contexts(n_contexts):
    contexts = []
    for i in range(n_contexts):
        with open(os.path.join('context', f'context_{i}.txt'), encoding='utf-8') as f:
            contexts.append(f.read())
    return contexts

context_files = [f"context/context_{i}.txt" for i in range(10)]

def bert_answer(msg, context_files):
    answer_found = False
    answer = None
    highest_accuracy = -float('inf')
    for context_file in context_files:
        with open(context_file, "r", encoding='utf-8') as f:
            context = f.read()
        inputs = tokenizer(msg, context, padding=True, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = bert_model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)   # Get the most likely beginning of the answer 
        answer_end = torch.argmax(answer_end_scores)  # Get the most likely end of the answer
        if answer_start >= answer_end:
            accuracy = -float('inf')
        else:
            tokens = inputs['input_ids'][0][answer_start:answer_end + 1]
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
            accuracy = answer_end_scores[0][answer_end] 
            print(f"Answer found in {context_file}: {answer}, accuracy: {accuracy}")
            answer_found = True
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            res = answer
            print(f"New highest accuracy: {highest_accuracy}")
    if not answer_found:
        res = "Sorry, I am unable to answer your question."
        print(res)
    return res

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getChatBotResponse') 
def chat():
    msg = request.args.get('msg')
    if msg.lower() in ["hi there", "how are you", "is anyone there?", "hey", "hola", "hello", "good day", "namaste", "hi"]:
        response = "Hi there! How can I assist you today?"
    elif msg.lower() in ["bye", "see you later", "goodbye", "get lost", "till next time", "bbye"]:
        response = "Bye Bye!"
    elif msg.lower() in ["thanks", "thank you", "that's helpful", "awesome, thanks", "thanks for helping me"]:
        response = "You're welcome! Happy to help."
    elif msg.lower() in ['help', 'how can you help']:
        response = "I'm here to assist you. How can I help?"
    else:    
        response = bert_answer(msg, context_files)
    print("Bot response: ",response)
    return jsonify({ "type": "text", "text": response })
    
@app.errorhandler(400)
def bad_request(error):
    return "Bad request: " + str(error), 400

@app.errorhandler(500)
def internal_server_error(error):
    return "Internal server error: " + str(error), 500

@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.error("Unhandled exception: %s", (error), exc_info=True)
    return "Internal server error: " + str(error), 500

@app.route('/static/images/robot.png')
def robot_img():
    return app.send_static_file('images/robot.png')

@app.route('/static/images/user.png')
def user_img():
    return app.send_static_file('images/user.png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)