import os
import uuid
import faiss
import numpy as np
from flask_cors import CORS
from PyPDF2 import PdfReader
from models.openaikeys import openai_key
from werkzeug.utils import secure_filename
from models.openaiassistant import OpenAIAssistant
from models.utils import create_folder_if_not_exist, is_folder_empty
from flask import Flask, render_template, request, jsonify


#########################################################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_key)
app             = Flask(__name__)
create_folder_if_not_exist('_uploads/')
create_folder_if_not_exist('_index_embeddings/')
#########################################################################################


def extract_text_from_pdf(pdf_path) -> list:

    reader = PdfReader(pdf_path)
    texts = list()

    for page in reader.pages:
        texts.append(str(page.extract_text()))

    return texts


def generate_embeddings(texts_to_embed:list=None) -> list:

    if texts_to_embed is None or not isinstance(texts_to_embed, list):
        raise Exception(f'Pass to the function a "LIST" of texts to be "EMBEDDED".')

    try:

        embeds = list()
        for text_to_embed in texts_to_embed:
            embeds.append( np.array(openAIAssistant.get_embeddings_from_openai(text_to_embed=str(text_to_embed))).squeeze() )

        return embeds

    except Exception as ex:
        raise Exception(f'While embeddings, there was this Exception: {ex}')


def create_faiss_index(embeddings: np.array):

    try:

        dimension = embeddings.shape[1]
        index      = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    except Exception as ex:
        raise Exception(f'While creating Faiss Index, this occured: \n\n {ex}')


def query_documents(request, userText):

    data       = request.json
    index_id   = data.get('index_id')

    if not index_id or not userText:
        return jsonify({'error': 'Invalid request'}), 400

    index      = faiss.read_index(f'./_index_embeddings/{index_id}.index')
    embeddings = np.load(f'./_index_embeddings/{index_id}_embeddings.npy')

    query_embedding = np.array(openAIAssistant.get_embeddings_from_openai(text_to_embed=userText)).squeeze()
    _, I = index.search(query_embedding, 1)

    with open(f'./_index_embeddings/{index_id}_mapping.txt', 'r') as f:
        mappings = f.readlines()

    result = mappings[I[0][0]].strip().split(':')

    return jsonify({'filename': result[0], 'page_number': int(result[1]) + 1})


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():

    if 'files' not in request.files:
        return jsonify({'error': f'No file part, here the request: {request}'}), 400

    files         = request.files.getlist('files')
    texts         = list()
    file_mappings = list()

    for file in files:

        # Read only PDFs for the moment ...
        if file and file.filename.endswith('.pdf'):

            filename = secure_filename(file.filename)
            filepath = os.path.join('_uploads/', filename)
            file.save(filepath)

            pdf_texts = extract_text_from_pdf(filepath)
            texts.extend(pdf_texts)
            file_mappings.extend([(filename, i) for i in range(len(pdf_texts))])

    # Create embeddings
    if os.path.exists('./_index_embeddings/embeddings.txt'):
        embeddings = np.load('./_index_embeddings/embeddings.npy')
    else:
        embeddings = generate_embeddings(texts)
        embeddings = np.array(embeddings)
        np.save('./_index_embeddings/embeddings.npy', embeddings)

    index = create_faiss_index(embeddings=embeddings)

    # Save index and mappings
    index_id = str(uuid.uuid4())
    np.save(f'./_index_embeddings/{index_id}_embeddings.npy', embeddings)
    faiss.write_index(index, f'./_index_embeddings/{index_id}.index')

    with open(f'./_index_embeddings/{index_id}_mapping.txt', 'w') as f:
        for mapping in file_mappings:
            f.write(f"./index_embeddings/{mapping[0]}:{mapping[1]}\n")

    return jsonify({'index_id': index_id})


@app.route("/get", methods=['POST', 'GET'])
def get_response():

    userText = request.args.get('msg')
    is_empty = is_folder_empty( '_uploads/' ) and is_folder_empty( '_index_embeddings/' )

    if is_empty:
        bot_answer = openAIAssistant.ask_gpt(user_query=userText)
    else:
        bot_answer = query_documents(request, userText)

    return str(bot_answer)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

