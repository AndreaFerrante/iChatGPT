import os
from flask_cors import CORS
from models.openaikeys import openai_main_key
from models.openaiassistant import OpenAIAssistant
from flask import Flask, render_template, request, jsonify, url_for, redirect
from models.embedder import get_pdf_dataframe_embeddings, search_a_query_in_docs_with_faiss


#################################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_main_key)
app             = Flask(__name__)

UPLOAD_FOLDER               = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#################################################################


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/get")
def get_response():

    userText   = request.args.get('msg')
    bot_answer = openAIAssistant.ask_gpt(user_query=userText)

    return bot_answer


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return 'File successfully uploaded'
    return 'File upload failed'


if __name__ == "__main__":
    app.run(debug=True, port=5000)





