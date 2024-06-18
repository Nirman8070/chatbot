# Chatbot
An AI Chatbot using Python

## Requirements (libraries)
1. TensorFlow

## VsCode SetUp
1. Run ```pip install --upgrade tensorflow``` to install ```Tensorflow```
2. Run ```pip install nltk``` to install ```nltk```
3. Run ```pip install tkinter``` to install ```tkinter```
4. To access your bot on localhost, go to ```http://127.0.0.1:5000/ ``` 


### Execution
To run this Bot, first run the ```train.py``` file to train the model. This will generate a file named ```chatbot_model.h5```
This is the model which will be used by the Flask REST API to easily give feedback without the need to retrain.
After running ```train.py```, next run the ```app.py``` to initialize and start the bot.
To add more terms and vocabulary to the bot, modify the ```intents.json``` file and add your personalized words and retrain the model again.