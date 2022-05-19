# Loading the libraries
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold,StratifiedKFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,f1_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from scipy.stats import bartlett, chi2, loguniform
import os
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import bartlett
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import time
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.pipeline import make_pipeline 
from PIL import Image
from bs4 import BeautifulSoup
import networkx as nx 
import pickle
from PIL import Image
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
sb.set(rc={'figure.figsize':(30, 5)})
from warnings import filterwarnings
filterwarnings('ignore')
df = pd.read_csv(r"C:\Users\brian\Documents\phishing_site_urls.csv")

# Streamlit
import streamlit as st
selected = st.sidebar.selectbox("Select the section", [
                                'Introduction', 'Dataset', 'Exploration' ,'Verification','Graph','Prediction','Reccomendation'])

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
        st.markdown('The data set does not require much cleaning. Detailed cleaning may be done during data preparation')
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
elif selected == 'Prediction':
    st.header("Training")
    st.write("training in progres ...")
    df[['Label']] = df[['Label']].apply(LabelEncoder().fit_transform)
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
    vocab_size = 20000
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
    model.compile(optimizer='adam', metrics=['accuracy'],loss='binary_crossentropy')
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    num_epochs = 1
    history = model.fit(padded,training_labels_final, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels_final))
    st.write("Congratulations,Training is completed")
    st.header("Prediction")
    data = st.text_input("Insert the URL to test here")
    link = data
    st.write('Prediction Started ...')
    t0= time.perf_counter()
    data = str(data)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(data)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(data)
    padded_data = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    score =model.predict(padded_data).round(0).astype('int')
    score =np.average(score)
    t1 = time.perf_counter() - t0
    st.write('Prediction Completed\nTime taken',t1 ,'sec')
    if score <= 0.2:
        st.write("The URL is probaly a phising URL. Kindly read through the reccomendations")
    else:
        st.write("The website is secure, kindly click the link to proceed")
        st.write(link)
        st.write("Thank You!")
        
elif selected == 'Graph':

    def countPlot():
        fig = plt.figure(figsize=(3, 2))
        sb.countplot(x="Label", data=df)
        plt.title('Bar graph showing distribution of urls')
        return st.pyplot(fig)
    countPlot()
elif selected == 'Reccomendation':
    st.header("How to Protect Your Computer")
    st.subheader("Below are some key steps to protecting your computer from intrusion:")
    st.markdown("""Keep Your Firewall Turned On: A firewall helps protect your computer from hackers who might try to gain access to crash it,
            delete information, or even steal passwords or other sensitive information. Software firewalls are widely recommended for single
            computers. The software is prepackaged on some operating systems or can be purchased for individual computers.
            For multiple networked computers, hardware routers typically provide firewall protection.\n\n""")
    st.markdown("""Install or Update Your Antivirus Software: Antivirus software is designed to prevent malicious software programs from embedding on
            your computer. If it detects malicious code, like a virus or a worm, it works to disarm or remove it. Viruses can infect computers
            without users’ knowledge. Most types of antivirus software can be set up to update automatically.\n\n""")
    st.markdown("""Install or Update Your Antispyware Technology: Spyware is just what it sounds like—software that is surreptitiously installed on your
            computer to let others peer into your activities on the computer. Some spyware collects information about you without your consent or
            produces unwanted pop-up ads on your web browser. Some operating systems offer free spyware protection, and inexpensive software is
            readily available for download on the Internet or at your local computer store. Be wary of ads on the Internet offering downloadable
            antispyware—in some cases these products may be fake and may actually contain spyware or other malicious code.
            It’s like buying groceries—shop where you trust.\n\n""")
    st.markdown("""Keep Your Operating System Up to Date: Computer operating systems are periodically updated to stay in tune with technology requirements
            and to fix security holes. Be sure to install the updates to ensure your computer has the latest protection.\n\n""")
    st.markdown("""Be Careful What You Download: Carelessly downloading e-mail attachments can circumvent even the most vigilant anti-virus software.
            Never open an e-mail attachment from someone you don’t know, and be wary of forwarded attachments from people you do know.
            They may have unwittingly advanced malicious code.\n\n""")
    st.markdown("""Turn Off Your Computer: With the growth of high-speed Internet connections, many opt to leave their computers on and ready for action.
            The downside is that being “always on” renders computers more susceptible. Beyond firewall protection, which is designed to fend off
            unwanted attacks, turning the computer off effectively severs an attacker’s connection—be it spyware or a botnet that employs your
            computer’s resources to reach out to other unwitting users.""")
