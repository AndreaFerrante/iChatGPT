# iChatGPT

This is a Python [Flask](https://flask.palletsprojects.com/en/3.0.x/)-based chatbot that uses the OpenAI APIs to chat and send/retrieve messages. 
This chatbot is faster than ChatGPT and, if you have OpenAI APIs running, you must only insert your private API Key inside the Python file named **__openaikeys__.py** contained inside app/models folder.

Before running the app, please go into folder *app/models/* and add into the file *__openaikeys__* your own **OpenAI private key**. Then, simply rename the same fiel into **openaikeys.py**, only.
As a reference, here OpenAI pricings and current models:

>- **OpenAI Models**: [here the models](https://platform.openai.com/docs/models)
>- **OpenAI Pricing**: [here the princing](https://openai.com/pricing)
>- **OpenAI API**: [here OpenAI API](https://platform.openai.com/docs/quickstart){target="_blank"}

It's a *Flask based application* so running the **app.py** file will make the code give you back the following interface:

![robo_chatter](https://github.com/AndreaFerrante/iChatGPT/assets/19763070/bc8afda1-c603-4a22-a77e-885c4fab8123)
