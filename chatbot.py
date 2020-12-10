  
import json
import nltk
import numpy
import random
from tensorflow.python.framework import ops
import tflearn
# Required to save model and data structures
import pickle

# Required for NLP
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Load intents.json file to data
with open('intents.json') as file:
    data = json.load(file)

# If a previosuly saved model is present load it 
try:
    # Retrieve previously saved model and its data structures
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except: 
    # Store words from each sentence 
    words = []
    # Contains tags 
    labels = []
    pattern_words = []
    pattern_label = []

    # loop through each patterns in the intent
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize each word in the sentence 
            word = nltk.word_tokenize(pattern)
            # Add tokenized words to words list
            words.extend(word)
            pattern_words.append(word)
            pattern_label.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Convert each word to lower case and Stem 
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # Remove duplicate words and sort them
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # Needed for creating training data 
    training = []
    output = []

    # Empty array for output
    output_empty = [0] * len(labels)

    for x, p_words in enumerate(pattern_words):
        # Initialize bag of words
        bag = []

        # Tokenize words present in pattern_words
        wrds = [stemmer.stem(w.lower()) for w in p_words]

        # If the wrds (pattern words) are present in the
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(pattern_label[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Convert to numpy array for input_data
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    # reset underlying graph data
    ops.reset_default_graph()

    # Build Neural network 
    net = tflearn.input_data(shape=[None, len(training[0])])
    # Two hidden layers
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    # Output layer
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    # Define  model 
    model = tflearn.DNN(net)
    # Train and save model
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')

try:
    model.load('./model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')

# Create a bag of words of user's input
def bag_of_words(s, words):
    bag = [0] * len(words)

    user_words = nltk.word_tokenize(s)
    user_words = [stemmer.stem(word.lower()) for word in user_words]

    for uw in user_words:
        for i, w in enumerate(words):
            if w == uw:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (Enter quit to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        results = model.predict([bag_of_words(user_input, words)])[0]
        # Index of maximum value from label probabilities 
        results_index = numpy.argmax(results)
        # Tag/Label with highest probability
        tag = labels[results_index]

        if (results[results_index] > 0.7):
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I'm not sure about that. Try again.")

chat()