import tensorflow as tf
from collections import Counter
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd  # 1
import streamlit as st

selected = st.sidebar.selectbox("Select the section", [
                                'Introduction', 'Dataset', 'Exploration', 'Verification', 'Web', 'Graph'])

siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()
modelTraining = st.container()

if selected == 'Introduction':
    with siteHeader:
        st.title('DETECTION OF PHISHING WEBSITES!')
        st.title('GROUP MEMBERS')
        st.markdown('Brian Onyango')
        st.markdown('Anne Nyambura')
        st.markdown('Victoria Mbaka')
        st.markdown('Felistas Njoroge')
        st.markdown('Matilda Kadzo')
        st.text("")
        st.title('INTRODUCTION')
        st.markdown("The expectation is that this project will give us better insight on phishing, \ni.e how to distinguish phishing websites from legitimate websites by selecting the best algorithm and have it embedded\nin browsers as an extension that detects the phishing sites.\nThrough this, we will be able to prevent and educate internet users on the deceptive ways of phishers through URLs and\nthus reduce the rate of financial theft from users and organizations online.")
elif selected == 'Dataset':
    with dataExploration:
        st.header('Dataset: ')
        st.markdown(
            'In this project, a dataset containing information for Phishing sites was collected from Kaggle data for Phishing Site URLs')
        st.markdown(
            'Link: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls?select=phishing_site_urls.csv')

        # Reading our dataset
        df = pd.read_csv('phishing_site_urls.csv')  # 2
        st.write(df.sample(10))
elif selected == 'Exploration':
    with newFeatures:
        st.header('Describing and Exploring Data')
        st.text('Data contains 549,346 entries.There are two columns.')
        st.markdown('Label column is prediction col which has 2 categories:')
        st.markdown(
            'Good - which means the URL does not contain ‘bait’ and therefore is not a Phishing Site')
        st.markdown(
            'Bad - which means the URL contains ‘bait’ therefore is a Phishing Site.')
        st.text("")
        st.markdown('Cross-Industry Standard Process For Data mining(CRISP-DM) will be used for conducting this research Link:  https://docs.google.com/document/d/11qRGmqJynQMOJ8AlHsy0rl6NLIVPnnr44c6XzwHeN90/edit')
        st.text("")
        st.markdown(
            'JIRA Kanban board to manage and track the different tasks involved in this project. Link: ')
        st.text("")
        st.markdown('TensorFlow to view the Neural Network’s creation')
        st.text("")
        st.markdown('Streamlit for deployment')
        st.text("")
        st.markdown('A GitHub repository. Link: ')
        st.text("")
        st.markdown('Presentation slides for the project Link:')
elif selected == 'Verification':
    with modelTraining:
        st.header('Verifying Data Quality')
        st.text('The data set does not require much cleaning. Detailed cleaning may be done during data preparation')
        st.header('Data Preparation')
        st.markdown('These are the steps followed in preparing the data')
        st.subheader('Loading Data')
        st.markdown(
            'Loaded the dataset from the CSV and then created a python notebook from it.')
        st.subheader('Cleaning Data')
        st.markdown('The data cleaning involved several steps;')
        st.markdown('Missing Values -  The dataset has no missing values.')
        st.markdown(
            'Duplicates - The dataset was found to have 42145 duplicate values which were dropped')
        st.markdown(
            'Column names - All the columns are named appropriately and in a homogenous manner')

        st.subheader('Data Types')
        st.markdown(
            'The  dataset contains categorical variables: url and labels')
        st.subheader('Assumptions')
        st.markdown('The data provided is correct and up to date')

elif selected == 'Web':
    df = pd.read_csv('phishing_site_urls.csv')
    tokenizer = Tokenizer(oov_token="<OOV>")
    split = round(len(df)*0.8)
    train_reviews = df['URL'][:split]
    train_label = df['Label'][:split]
    test_reviews = df['URL'][split:]
    test_label = df['Label'][split:]
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []
    for row in train_reviews:
        training_sentences.append(str(row))
    for row in train_label:
        training_labels.append(row)
    for row in test_reviews:
        testing_sentences.append(str(row))
    for row in test_label:
        testing_labels.append(row)
    vocab_size = 400000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = '<OOV>'
    padding_type = 'post'
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sentences, maxlen=max_length)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    counter = Counter(training_labels_final)
    smt = SMOTE()
    X_train_sm, y_train_sm = smt.fit_resample(padded, training_labels_final)
    num_epochs = 20
    history = model.fit(X_train_sm, y_train_sm, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels_final))
    reccomendations = ("""Protections How to Protect Your Computer Below are some key steps to protecting your computer from intrusion:\n\n Keep Your Firewall Turned On: A firewall helps protect your computer from hackers who might try to gain access to crash it,\n
    delete information, or even steal passwords or other sensitive information. Software firewalls are widely recommended for single\n
    computers. The software is prepackaged on some operating systems or can be purchased for individual computers.\n
    For multiple networked computers, hardware routers typically provide firewall protection.\n\n
    Install or Update Your Antivirus Software: Antivirus software is designed to prevent malicious software programs from embedding on\n
    your computer. If it detects malicious code, like a virus or a worm, it works to disarm or remove it. Viruses can infect computers\n
    without users’ knowledge. Most types of antivirus software can be set up to update automatically.\n\n
    Install or Update Your Antispyware Technology: Spyware is just what it sounds like—software that is surreptitiously installed on your\n
    computer to let others peer into your activities on the computer. Some spyware collects information about you without your consent or\n
    produces unwanted pop-up ads on your web browser. Some operating systems offer free spyware protection, and inexpensive software is\n
    readily available for download on the Internet or at your local computer store. Be wary of ads on the Internet offering downloadable\n
    antispyware—in some cases these products may be fake and may actually contain spyware or other malicious code.\n
    It’s like buying groceries—shop where you trust.\n\n
    Keep Your Operating System Up to Date: Computer operating systems are periodically updated to stay in tune with technology requirements\n
    and to fix security holes. Be sure to install the updates to ensure your computer has the latest protection.\n\n
    Be Careful What You Download: Carelessly downloading e-mail attachments can circumvent even the most vigilant anti-virus software.\n
    Never open an e-mail attachment from someone you don’t know, and be wary of forwarded attachments from people you do know.\n
    They may have unwittingly advanced malicious code.\n\n
    Turn Off Your Computer: With the growth of high-speed Internet connections, many opt to leave their computers on and ready for action.\n
    The downside is that being “always on” renders computers more susceptible. Beyond firewall protection, which is designed to fend off\n
    unwanted attacks, turning the computer off effectively severs an attacker’s connection—be it spyware or a botnet that employs your\n
    computer’s resources to reach out to other unwitting users.""")
    print("Insert the Link")
    data = input("Paste the Link Here\t")
    print('Prediction Started ...')
    t0 = time.perf_counter()
    data = str(data)
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen=max_length)
    score = model.predict(data).round(0).astype('int')
    t1 = time.perf_counter() - t0
    print('Prediction Completed\nTime taken', t1, 'sec')
    if score == 0:
        print("The URL is probaly a phising URL. Kindly read through the following Reccomendations\n", reccomendations)
    else:
        print("The website is secure, kindly click the link to proceed")
        print(data)
    classes = np.argmax(score, axis=1)
    print(classes)


# elif selected == 'Graph':
#     df = pd.read_csv('phishing_site_urls.csv')  # 2

#     def countPlot():
#         fig = plt.figure(figsize=(3, 2))
#         sb.countplot(x="Label", data=df)
#         plt.title('Bar graph showing distribution of urls')
#         return st.pyplot(fig)
# countPlot()

# image = Image.open('bad_sites.png')

# st.image(image, caption='Visualization of the words used in bad sites')

# image = Image.open('good_sites.png')

# st.image(image, caption='Visualization of the words used in good sites')
