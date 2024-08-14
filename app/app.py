import os
from flask_cors import CORS
from models.openaikeys import openai_key
from models.openaiassistant import OpenAIAssistant
from flask import Flask, render_template, request, jsonify, url_for, redirect
from models.embedder import get_pdf_dataframe_embeddings, search_a_query_in_docs_with_faiss


#################################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_key)
pdf_files       = None
app             = Flask(__name__)
#################################################################


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/get")
def get_response():

    userText   = request.args.get('msg')
    bot_answer = openAIAssistant.ask_gpt(user_query=userText)

    if pdf_files is not None:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    return bot_answer


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    pdf_files      = request.files.getlist('file')
    all_embeddings = {}

    # for file in files:
    #
    #     if file.filename == '':
    #         return jsonify({'error': 'No selected file'}), 400
    #
    #     if file and file.filename.endswith('.pdf'):
    #         file_stream = io.BytesIO(file.read())
    #         texts       = extract_text_from_pdf(file_stream)
    #         embeddings  = generate_embeddings(texts)
    #         all_embeddings[file.filename] = embeddings

    return jsonify(all_embeddings), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)





