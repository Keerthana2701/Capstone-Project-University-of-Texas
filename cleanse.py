import pandas as pd
import numpy as np
from word2number import w2n

import random
import pickle
import nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')


accidents_level_dict = {"accident_level":{"Bees/Venomous Animals":1,"Blocking and isolation of energies":1,"Fall":1.7272727272727273,"Hazards during Electrical installations":2,"Injuries/Cuts":1.3888888888888888,"Manual Tools":1.55,"NA":4,"Others":1.5086206896551724,"Power lock":4.666666666666667,"Pressed":1.7083333333333333,"Pressurized Systems/ Chemical Substances":1.1875,"Projections of fragments":1.7307692307692308,"Suspended Loads":1.8333333333333333,"Vehicles and Mobile Equipment":1.8888888888888888},"potential_accident_level":{"Bees/Venomous Animals":1.1538461538461537,"Blocking and isolation of energies":2.6666666666666665,"Fall":3.1363636363636362,"Hazards during Electrical installations":4.333333333333333,"Injuries/Cuts":2.6666666666666665,"Manual Tools":2.75,"NA":5,"Others":3.103448275862069,"Power lock":5,"Pressed":3,"Pressurized Systems/ Chemical Substances":3.03125,"Projections of fragments":3.8461538461538463,"Suspended Loads":3.6666666666666665,"Vehicles and Mobile Equipment":4}}

df_conversion = pd.read_excel("./other_files/conversions.xlsx", sheet_name=0)
df_conversion = df_conversion[["ori", "convert"]]
df_conversion = df_conversion[~df_conversion.convert.isna()] # Filter rows where not NA

df_chem = pd.read_excel("./other_files/conversions.xlsx", sheet_name=1)
df_chem = df_chem[["chemical_formula", "synonyms"]]
df_chem = df_chem[~df_chem.synonyms.isna()] # Filter rows where not NA

stop_word_list = list(set(stopwords.words('english'))) # Load standard stop words
stopword_pattern = re.compile(r'\b(' + r'|'.join(stop_word_list) + r')\b\s*') # Compile regex to remove stopwords

chemical_names = df_chem['synonyms'].str.lower().to_list() # Get a list of all chemical names
chemical_symbols = df_chem['chemical_formula'].dropna().astype('str').str.lower().to_list() # Get a list of all chemical symbols
df_conversion_dict = {i['ori']:i['convert'] for i in df_conversion.to_dict('records')}

class Chatbot():
    def __init__(self, model, tokenizer, classes, thresh, name='Albert'):
        self.name = name
        self.flag = True
        self.model = model
        self.thresh = thresh
        self.tokenizer = tokenizer
        self.words = pickle.load(open('./other_files/words.pkl', 'rb'))
        self.classes = classes
        self.WORDS_ZEROS = {word: 0 for word in self.words}
    
    def preprocess_text(self, raw_text):
        cleaned_text = raw_text
        df = pd.DataFrame({'description': [raw_text]})
        df['desc'] = df['description'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True).str.replace('mx', 'm x ').str.replace(stopword_pattern, ' ', regex=True) # Replace 10cmx20cm to 10cm x 20cm
        df['desc'] = df['desc'].str.split(' ')

        # Convert words to numbers
        def convert_numbers(x):
            new_x = []
            for word in x:
                try:
                    word = w2n.word_to_num(word.lower())
                except Exception as e:
                    word = word.lower()
                if len(str(word)) > 0: # If the length of the word is zero, then don't return
                    new_x.append(str(word))
            return ' '.join(new_x)

        df['desc'] = df['desc'].apply(convert_numbers)

        # High frequency occuring bigram removal
        bigram_stop_pattern = "(i?)(left hand|left foot|left leg|left arm|right hand|right leg|right foot|right arm|time accident|causing the injury|caused the injury|caused injury|causing injury)+"
        df["desc"] = df["desc"].str.replace(bigram_stop_pattern, '', regex=True)

        # Convert any dimensions like 10cm x 20cm x 30cm to xdimension
        dimension_pattern = re.compile(r"(i?)((([\d.]+)\s{0,}(cm|m)\s{0,})+(\s{0,}[x|X]\s{0,}\b|\b))+")
        df["desc"] = df["desc"].str.replace(dimension_pattern, ' xdimension ', regex=True)

        # Convert weights to xweight
        weight_pattern = re.compile(r"(?i)([\d.]+)\s{0,}(lb(s\b|\b)|pound(s\b|\b)|ton(s\b|\b)|kilogram(s\b|\b)|gram(s\b|\b)|ounce(s\b|\b)|oz(s\b|\b)|g(s\b|\b)|gr(s\b|\b)|kg(s\b|\b)|tn(s\b|\b))")
        df["desc"] = df["desc"].str.replace(weight_pattern, ' xweight ', regex=True)

        # Convert lenghts/heights to xlength
        lenght_pattern = re.compile(r"(?i)([\d.]+)\s{0,}(millimeter(s\b|\b)|centimeter(s\b|\b)|meter(s\b|\b)|kilometer(s\b|\b)|mm(s\b|\b)|cm(s\b|\b)|m(s\b|\b)|km(s\b|\b)|mt(s\b|\b)|inch|inches|yard(s\b|\b))")
        df["desc"] = df["desc"].str.replace(lenght_pattern, ' xlength ', regex=True)

        df['desc'] = df['desc'].str.split(' ')

        lemma = WordNetLemmatizer()
        def convert_words(word): # Word corrections in case of typing mistakes
            if word in df_conversion_dict:
                return df_conversion_dict[word]
            else:
                return word

        def convert_chemicals(word): # Convert chemical names/symbols to 'chemical'
            if word in chemical_names or word in chemical_symbols:
                return 'chemical'
            else:
                return word

        def stop_word_removal(word): # Remove any stopwords
            if word not in stop_word_list:
                return word
            else:
                return ''

        def data_preprocessing(word): # Chain all functions together to perform the data preprocessing
            word = word.lower()
            word = convert_words(word)
            word = convert_chemicals(word)
            word = lemma.lemmatize(word, 'v')
            if word.isnumeric():
                return ''
            return word

        df['desc'] = df['desc'].apply(lambda x: [data_preprocessing(i) for i in x if len(data_preprocessing(i)) > 0])
        df['len'] = df['desc'].apply(len)

        df['desc_split'] = df['desc']
        df['desc'] = df['desc'].str.join(' ')

        seq = self.tokenizer.texts_to_sequences(df['desc'])
        padded = pad_sequences(seq, maxlen=250)
        return padded

    def predict_class(self, sentence):
        proc = self.preprocess_text(sentence)
        res = self.model.predict(proc)[0]
        print(res)
        predictions_dict = [{"intent" : self.classes[cl], "probability" : prob} for cl, prob in enumerate(res) if prob > self.thresh]
        return predictions_dict

    def chatbot_response(self, msg):
        predictions = self.predict_class(msg)
        if len(predictions) == 0:
            tag = random.choice(["Sorry, I didn't get that.", "Sorry! I don't have an answer for that."])
            mean_accident_level = 0
            mean_potential_accident_level = 0
        else:
            tag = predictions[0]['intent']
            mean_accident_level = round(accidents_level_dict['accident_level'][tag], 2)
            mean_potential_accident_level = round(accidents_level_dict['potential_accident_level'][tag], 2)
            tag = "Critical Risk falls under the category: " + predictions[0]['intent']
        return tag, mean_accident_level, mean_potential_accident_level

