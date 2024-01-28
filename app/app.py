from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_dropzone import Dropzone
from models.embedder import get_pdf_dataframe_embeddings, search_a_query_in_docs_with_faiss
from models.openai import get_embeddings_openai, ask_gpt
from models.servicefactory import *


def getResponse():
    return 'Hello there !'


def get_chatbot_response(user_request):

    user_request_embedded = get_embeddings_openai(text_to_embed = user_request)
    similarities          = ''
    response              = ask_gpt(question   = user_request,
                                     content    = similarities)

    return response


app               = Flask(__name__)
app.static_folder = 'static'
basedir           = os.path.abspath(os.path.dirname(__file__))
pdf_dataframe     = None
final, npe        = None, None
CORS(app)


app.config.update(
    UPLOADED_PATH                   = os.path.join(basedir, 'uploads'),
    DROPZONE_UPLOAD_ON_CLICK        = True,
    DROPZONE_MAX_FILE_SIZE          = 1024,
    DROPZONE_MAX_FILES              = 30,
    DROPZONE_TIMEOUT                = 5 * 60 * 1000,
    DROPZONE_ALLOWED_FILE_TYPE      = 'app',
    DROPZONE_REDIRECT_VIEW          = 'completed'  # Set redirect view once the upload is finished !
)


########################
dropzone = Dropzone(app)
########################


@app.route('/upload', methods=['POST', 'GET'])
def upload():

    # Process all the uploaded PDFs to make them as txt...
    delete_all_files( os.path.join(basedir, 'uploads') )

    if request.method == 'POST':
        for key, f in request.files.items():
            if key.startswith('file'):
                f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))

    return render_template('upload.html')


@app.route("/get")
def get_response():

    userText = request.args.get('msg')
    ######################################################
    bot_answer = getResponse() # <<<-------------------
    ######################################################
    bot_answer = bot_answer.replace('.', '.\n')
    bot_answer = bot_answer.replace('!', '!\n')
    return bot_answer


@app.route('/api/process_pdf', methods=['POST', 'GET'])
def process_pdf():

    # Process all the PDFs upladed in the folder ...
    if pdf_dataframe is None:
        print(f'XXXXXXXXXXXXXXXXXX {os.chdir("..")}')
        final, npe = get_pdf_dataframe_embeddings(all_pdf_in_path = None,
                                                  path_to_embed   = os.getcwd() + '/pdfs/',
                                                  use_openai      = True)
        return jsonify({'message': 'all pdf are already have been processed !'}), 200

    else:
        return jsonify({'message': 'all pdf are already processed'}), 200


@app.route('/api/get_response', methods=['GET'])
def get_response_from_openai():

    userText = request.args.get('msg')
    ######################################################
    bot_answer = getResponse() # <<<-------------------
    ######################################################
    bot_answer = bot_answer.replace('.', '.\n')
    bot_answer = bot_answer.replace('!', '!\n')
    return bot_answer


if __name__ == "__main__":
    app.run(debug=False, port=5000)



