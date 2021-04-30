from django.shortcuts import render
import mood_classifier
from .helpers import *
import transformers
from mood_classifier_app.forms import UserEntryForm
from mood_classifier_app.models import UserEntry
import datetime

DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'
ACCURACY_THRESHOLD = .80
LABELS = ["Looks like your day isn't going so well, Sorry about that :(", "Glad to hear you're having a good day!"]
MOOD_MODEL_DIR = './models/mood_model/'

#this takes forever, but the this is the suggested batch size given stanford's IMDB dataset.
# feel free to cange for faster testing.
BATCH_SIZE = 30000

# main form view here
def index(request):
    if request.method == "POST":
        user_entry_form = UserEntryForm(request.POST)
        if user_entry_form.is_valid():
            day_description = user_entry_form.cleaned_data['user_input_text']
            #guess the mood
            from transformers import TFBertForSequenceClassification
            #load trained model from memory so we're not training every time.
            model = TFBertForSequenceClassification.from_pretrained(MOOD_MODEL_DIR)
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            mood = list(getMood(day_description, model, tokenizer))[0]
            print(mood)
            context = {
                'mood': LABELS[mood],
                'user_entry_form': UserEntryForm()
            }
            # save user input for further learning or other purposes
            date = datetime.datetime.now()
            bar = UserEntry.objects.create(
                user_input_text=day_description,
                entry_date=date,
                mood_classification=mood
            )
    else:
        context = {
            'user_entry_form': UserEntryForm()
        }
    return render(request, 'mood_classifier/index.html', context)

# this is a view and a function that I made for convienvience to train the model.
def train_model(request):
    if request.method == "POST":
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        print('thinks we should train model.')
        #IMDB movie reviews
        dataset = get_dataset()
        #save dataset to file memory
        clean_dataset(dataset)

        #training and testing datasets here
        train = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/train', batch_size=BATCH_SIZE, validation_split=0.2,
            subset='training', seed=123)

        test = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/train', batch_size=BATCH_SIZE, validation_split=0.2,
            subset='validation', seed=123)

        #convert to pandas dataframes
        train = convert_dataset_to_dataframe(train)
        test = convert_dataset_to_dataframe(test)

        #convert data to tf datasets
        train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

        train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
        train_data = train_data.shuffle(100).batch(32).repeat(2)

        validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
        validation_data = validation_data.batch(32)

        #fine tune it!
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

        callbacks = stopAt75PercentAccurate()

        model.fit(train_data, epochs=2, validation_data=validation_data)

        #save the model weights to memory to be used later
        model.save_pretrained(MOOD_MODEL_DIR)

    context = {}
    return render(request, 'mood_classifier/train_model_button.html', context)
