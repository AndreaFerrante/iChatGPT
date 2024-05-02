from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from models.openaikeys import openai_key
from models.embedder import get_pdf_dataframe_embeddings, search_a_query_in_docs_with_faiss
from models.openaiassistant import OpenAIAssistant


############################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_key)
app             = Flask(__name__)
############################################################


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/get")
def get_response():

    userText   = request.args.get('msg')
    bot_answer = openAIAssistant.ask_gpt(user_query=userText)

    return bot_answer


if __name__ == "__main__":
    app.run(debug=True, port=5000)





