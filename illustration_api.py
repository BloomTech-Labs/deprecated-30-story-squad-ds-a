from fastapi import FastAPI
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Labs30-StorySquad-DS-Team A",
    description="An API for the illustration score",
    version="0.1",
    docs_url="/"
)

# this api is meant to help transfer the illustration similarity scoring model
# when set up properly, there should be a file called 'transfer_model.h5' in the same scope as this illustration_api file
# first, you'll want to download the data that the neural network will use, which can be found at this link: https://drive.google.com/drive/folders/1rWbjhPRoGj-kwvESVUWhAigfecsN6XDo?usp=sharing
# then open the Google Colaboratory notebook that can be found here: https://colab.research.google.com/drive/1J66ylaqZfZQzCiOmRYHJ4mWt7Jmh7y_B?usp=sharing
# get the data folder to the "Files" sidebar, run all the cells properly, and take the newly downloaded h5 from your Colaboratory workflow to the story-squad-ds-a main folder
# lastly, uncomment the line below this one

#model = load_model('transfer_model.h5')

if __name__ == "__main__":
    uvicorn.run(app)