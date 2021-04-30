from django.shortcuts import render
import mood_classifier
from .helpers import *
import transformers


DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'
ACCURACY_THRESHOLD = .80
LABELS = ['Negative','Positive']
MOOD_MODEL_DIR = './models/mood_model/'

# main form view here
def index(request):
    if request.method == "POST":
        #guess the mood
        from transformers import TFBertForSequenceClassification
        #load trained model from memory so we're not training every time.
        model = TFBertForSequenceClassification.from_pretrained(MOOD_MODEL_DIR)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        mood = list(getMood(request.POST['input_text'], model, tokenizer))[0]
        print(mood)
        context = {
            'mood': LABELS[mood]
        }
    else:
        context = {}
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
            'aclImdb/train', batch_size=1000, validation_split=0.2,
            subset='training', seed=123)

        test = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/train', batch_size=1000, validation_split=0.2,
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
