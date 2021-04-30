# Mood Classifier
A small Django app that, when given a text blob about your day, tries to predict if is was a good day or a bad day. The app uses a pretrained NLP model called BERT to execute binary classification of text blobs. The model is trained on 25,000 movie reviews on IMDB, courtesy of Standford. The goal of this app is to showcase the use of Django, pandas, Tensorflow NLP and SQL.

### Requirements

Run `pip install -r requirements.txt` from the Django project folder.

Next, you're going to want to train your model from the core/train_model_button page. Please input a batch size here. The recommended size is 30000, but that could take an extremely long time, so I recommend something more like 32 for a quick proof of concept review.

After training is complete, please navigate to the /core/ url where you can find the mood classification form. The controller returns a positive or negative remark based on the user's description of their day. The controller also writes the user's input, timespamps it and saves the infered mood as an integer to a SQLite server. I prefer postgres generally, but from the perspective of quick setup this is a better option in my opinion.


Please note: I would love to have included the model here but the size is over 100MB.
