import re
import nltk
import keras
import requests
import customtkinter
import threading
import matplotlib
import pandas as pd
import numpy as np
from newspaper import Article
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from cleantext import clean
from tkinter import *
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

data = pd.read_csv('IMDB.csv')

# Global Variables
model = ""
word_tokenizer = lemmatizer = []
train_sentences = test_sentences = train_labels = test_labels = ""
vocab_size = 3000
oov_tok = ''
embedding_dim = 100
max_length = 200
padding_type = 'post'
trunc_type = 'post'
tokenizer = train_sequences = train_padded = test_sequences = test_padded = []

api_key = "cf004587bce74263ac18c81b20785a2e"

url_list = list()
dataframes = list()
html_page = str()
clean_test_data = str()
positive_sentiments = list()
negative_sentiments = list()


def clean_text(string):
    # HTML Tags
    string = re.sub(r"<[^<]+?>", '', string)
    # Non-word characters
    string = re.sub(r"[^\w\s]", '', string)
    # Digits
    string = re.sub(r"\d", '', string)
    string = string.lower()
    return string


def lemmatize_text(text):
    global word_tokenizer, lemmatizer
    string = ""
    for w in word_tokenizer.tokenize(text):
        string = string + lemmatizer.lemmatize(w) + " "
    return string


def pre_process():
    global word_tokenizer, lemmatizer, train_sentences, test_sentences, train_labels, test_labels

    data['review'] = data['review'].apply(lambda cw: clean_text(cw))
    stop_words = set(stopwords.words('english'))
    data['review'] = data['review'].apply(lambda l: ' '.join([word for word in l.split() if word not in stop_words]))

    word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    data['review'] = data.review.apply(lemmatize_text)

    reviews = data['review'].values
    labels = data['sentiment'].values
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels,
                                                                                  stratify=encoded_labels)


def model_hyperparameters():
    global tokenizer, train_sequences, train_padded, test_sequences, test_padded

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)


def initialize_model():
    global model
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())


def lstm_train():
    pre_process()
    model_hyperparameters()
    initialize_model()
    model.fit(train_padded, train_labels, epochs=5, verbose=1, validation_split=0.25)
    model.save("lstm.keras")
    prediction = model.predict(test_padded)
    pred_labels = []
    for i in prediction:
        if i >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    print("Accuracy of prediction on test set : ", accuracy_score(test_labels, pred_labels))


def lstm_test():
    sequences = tokenizer.texts_to_sequences(clean_test_data)
    padded = pad_sequences(sequences, padding='post', maxlen=max_length)
    prediction = model.predict(padded)

    for i in prediction:
        if i >= 0.5:
            positive_sentiments.append(1)
        else:
            negative_sentiments.append(0)


def schedule_thread(task):
    threading.Thread(target=task).start()


def get_user_data(key, size):
    parameters = {
        'q': str(key),
        'pageSize': int(size),
        'apiKey': api_key
    }

    return parameters


def get_url_requests(key, size):
    url = "https://newsapi.org/v2/everything?"

    query_parameters = get_user_data(key, size)
    response = requests.get(url, params=query_parameters)

    response_json = response.json()

    for i in response_json['articles']:
        print("\n" + i['url'])
        url_list.append(i['url'])


def parse_and_clean_data(current_url):
    global clean_test_data

    article = Article(current_url, language="en")
    article.download()
    article.parse()
    article.nlp()

    # Clean Text
    clean_test_data = clean(text=article.text, fix_unicode=True, to_ascii=True, lower=True, no_line_breaks=False,
                            no_urls=True,
                            no_emails=False, no_phone_numbers=False, no_numbers=True, no_digits=True,
                            no_currency_symbols=True,
                            no_punct=False, replace_with_punct=" ", replace_with_url="URL", replace_with_email="Email",
                            replace_with_phone_number=" ", replace_with_number=" ", replace_with_digit=" ",
                            replace_with_currency_symbol=" ", lang="en")

    # Convert from string to list of sentences
    clean_test_data = sent_tokenize(clean_test_data)

    for sent in clean_test_data:
        if len(sent) < 3:
            clean_test_data.remove(sent)


def scrape_and_predict_data():

    for count, i in enumerate(url_list):

        url = i

        schedule_thread(parse_and_clean_data(url))

        schedule_thread(lstm_test())

        num_positive_sentiments = len(positive_sentiments)
        num_negative_sentiments = len(negative_sentiments)
        total_sentiments = num_positive_sentiments + num_negative_sentiments
        negative_sentiments_percentage = (num_negative_sentiments / total_sentiments) * 100
        positive_sentiments_percentage = (num_positive_sentiments / total_sentiments) * 100

        if num_positive_sentiments > num_negative_sentiments:
            article_status = 'Positive'
            percentage = round(positive_sentiments_percentage, 2)
        elif num_negative_sentiments > num_positive_sentiments:
            article_status = 'Negative'
            percentage = round(negative_sentiments_percentage, 2)
        else:
            article_status = 'Neutral'
            percentage = 50.0

        print(f'\nPositive Sentiment Count = {num_positive_sentiments}')
        print(f'Negative Sentiment Count = {num_negative_sentiments}')
        print(f'Article {count + 1} Status = {article_status} | Percentage = {percentage}')

        article_data = {
            'Article': count + 1,
            'Positive': num_positive_sentiments,
            'Negative': num_negative_sentiments,
            'Status': article_status,
            'Percentage': percentage
        }

        dataframes.append(article_data)

    tab_view.add("Results")
    display_results(dataframes)
    tab_view.set("Results")

    dataframes.clear()
    url_list.clear()


def compute(key, size):
    schedule_thread(get_url_requests(key, size))
    schedule_thread(scrape_and_predict_data)


def display_results(result_data):
    # Results Tab Plot Widget
    figure = Figure(figsize=(6, 4), dpi=100)
    figure_canvas = FigureCanvasTkAgg(figure, tab_view.tab("Results"))
    NavigationToolbar2Tk(figure_canvas, tab_view.tab("Results"))

    ax = figure.add_subplot()

    label = list()
    plot_data = list()

    for count, i in enumerate(result_data):
        percentage_data = i['Percentage']
        plot_data.append(percentage_data)
        article_no = i['Article']
        article_status = i['Status']
        label_data = "Article : " + str(article_no) + "\nStatus : " + article_status + "\nPercentage : " + str(
            percentage_data)
        label.append(label_data)

    wedges, texts = ax.pie(plot_data, wedgeprops=dict(width=0.5), startangle=-40)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        plot_y = np.sin(np.deg2rad(ang))
        plot_x = np.cos(np.deg2rad(ang))
        horizontal_alignment = {-1: "right", 1: "left"}[int(np.sign(plot_x))]
        connection_style = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connection_style})
        ax.annotate(label[i], xy=(plot_x, plot_y), xytext=(1.35 * np.sign(plot_x), 1.4 * plot_y),
                    horizontalalignment=horizontal_alignment, **kw)

    figure_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def process_input():
    if tab_view.tab("Results"):
        tab_view.delete("Results")
    compute(key_text.get("0.0", "end"), page_text.get("0.0", "end"))


if __name__ == "__main__":

    lstm_train()

    model_hyperparameters()

    matplotlib.use('TkAgg')

    customtkinter.set_appearance_mode("system")
    customtkinter.set_default_color_theme("green")

    app = customtkinter.CTk()
    app.title("News Media Analytics")
    app.iconbitmap("Images/News.ico")

    # Centering Window
    width = 1000
    height = 700
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    app.geometry('%dx%d+%d+%d' % (width, height, x, y))
    app.resizable(False, False)

    app_font = customtkinter.CTkFont(family="Cascadia Code", size=20)
    app_image = customtkinter.CTkImage(light_image=Image.open("Images/News.png"),
                                       dark_image=Image.open("Images/News.png"),
                                       size=(128, 128))
    app_image_label = customtkinter.CTkLabel(master=app, image=app_image, text='')
    app_image_label.place(relx=0.5, rely=0.1, anchor=CENTER)

    # Tab View
    tab_view = customtkinter.CTkTabview(master=app, width=800, height=500, corner_radius=20)
    tab_view.add("Input")
    tab_view.add("Results")
    tab_view.set("Input")

    tab_view.place(relx=0.5, rely=0.6, anchor=CENTER)

    # Input Tab Widgets
    key_label = customtkinter.CTkLabel(master=tab_view.tab("Input"), text="Enter Key", font=app_font,
                                       text_color='dodgerblue2')
    key_label.place(relx=0.25, rely=0.2, anchor=CENTER)
    key_text = customtkinter.CTkTextbox(master=tab_view.tab("Input"), width=300, height=10, corner_radius=5,
                                        font=app_font,
                                        text_color='cyan')
    key_text.place(relx=0.65, rely=0.2, anchor=CENTER)

    page_label = customtkinter.CTkLabel(master=tab_view.tab("Input"), text="Enter No. of Articles", font=app_font,
                                        text_color='dodgerblue2')
    page_label.place(relx=0.25, rely=0.5, anchor=CENTER)
    page_text = customtkinter.CTkTextbox(master=tab_view.tab("Input"), width=300, height=10, corner_radius=5,
                                         font=app_font,
                                         text_color='cyan')
    page_text.place(relx=0.65, rely=0.5, anchor=CENTER)

    search_image = customtkinter.CTkImage(light_image=Image.open("Images/Search.png"),
                                          dark_image=Image.open("Images/Search.png"),
                                          size=(30, 30))

    search_button = customtkinter.CTkButton(master=tab_view.tab("Input"), image=search_image, text="Search",
                                            font=app_font,
                                            hover=True, command=process_input)
    search_button.place(relx=0.5, rely=0.85, anchor=CENTER)

    app.mainloop()
