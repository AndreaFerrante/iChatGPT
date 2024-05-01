from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from models.embedder import get_pdf_dataframe_embeddings, search_a_query_in_docs_with_faiss
from app.models.openaiassistant import get_embeddings_openai, ask_gpt


def get_chatbot_response(user_request):
    return ask_gpt(user_query = user_request)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get")
def get_response():

    userText = request.args.get('msg')

    ######################################################
    bot_answer = get_chatbot_response(userText)
    bot_answer = bot_answer.replace('.', '.\n')
    bot_answer = bot_answer.replace('!', '!\n')
    ######################################################

    return bot_answer


if __name__ == "__main__":
    app.run(debug=True, port=5000)






