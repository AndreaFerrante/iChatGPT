import os
import uuid
import faiss
import numpy as np
from flask_cors import CORS
from PyPDF2 import PdfReader
from models.openaikeys import openai_key
from werkzeug.utils import secure_filename
from models.openaiassistant import OpenAIAssistant
from models.utils import create_folder_if_not_exist
from flask import Flask, render_template, request, jsonify, url_for, redirect
from models.embedder import get_pdf_dataframe_embeddings, search_a_query_in_docs_with_faiss


#########################################################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_key)
app             = Flask(__name__)
index_id        = None # Variable used to understand if the user has dropped PDFs...
create_folder_if_not_exist('uploads/')
#########################################################################################


def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)
    texts = []

    for page in reader.pages:
        texts.append(page.extract_text())

    return texts


def generate_embeddings(text_to_embed):
    return openAIAssistant.get_embeddings_from_openai(text_to_embed=text_to_embed)


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]
    index      = faiss.IndexFlatL2(dimension)
    index.add(embeddings.numpy())

    return index


def query_documents(request, userText):

    data       = request.json
    index_id   = data.get('index_id')

    if not index_id or not userText:
        return jsonify({'error': 'Invalid request'}), 400

    index      = faiss.read_index(f'{index_id}.index')
    # embeddings = np.load(f'{index_id}_embeddings.npy')

    query_embedding = openAIAssistant.get_embeddings_from_openai(text_to_embed=userText)
    _, I = index.search(query_embedding.numpy(), 1)

    with open(f'{index_id}_mapping.txt', 'r') as f:
        mappings = f.readlines()

    result = mappings[I[0][0]].strip().split(':')

    return jsonify({'filename': result[0], 'page_number': int(result[1]) + 1})


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/get", methods=['POST', 'GET'])
def get_response():

    userText   = request.args.get('msg')



    if True:
        bot_answer = openAIAssistant.ask_gpt(user_query=userText)
    elif False:
        query_documents

    return bot_answer


@app.route('/upload', methods=['POST'])
def upload_pdf():

    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files         = request.files.getlist('files')
    texts         = []
    file_mappings = []

    for file in files:

        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            pdf_texts = extract_text_from_pdf(filepath)
            texts.extend(pdf_texts)
            file_mappings.extend([(filename, i) for i in range(len(pdf_texts))])

    embeddings = generate_embeddings(texts)
    index      = create_faiss_index(embeddings)

    # Save index and mappings
    index_id = str(uuid.uuid4())
    np.save(f'{index_id}_embeddings.npy', embeddings.numpy())
    faiss.write_index(index, f'{index_id}.index')
    with open(f'{index_id}_mapping.txt', 'w') as f:
        for mapping in file_mappings:
            f.write(f"{mapping[0]}:{mapping[1]}\n")

    return jsonify({'index_id': index_id})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

