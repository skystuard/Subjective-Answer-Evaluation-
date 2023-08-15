import re, math
import string
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import requests
import gensim.downloader as api
import gensim.models 


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')
WORD = re.compile(r'\w+')


def remove_punctuation(text):
    # Inputs string and returns string with Punctuation marks removed
    
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree
    
def process(text_sentences):
    # Inputs an array of sentences and returns an array of words with
    # stop words and punctuation removed 

    filtered_text = []
    for sentence in text_sentences:
        sentence_mod = remove_punctuation(sentence)
        for word in sentence_mod.split():
            if word not in stopwords:
                filtered_text.append(word)
    return filtered_text

def lemmatize(filtered_text,lemma=False):
    # Inputs array of words and lemmatizes if lemma is set to True
    # Doesnt do anything if lemma is set to False
    # Returns array of words
     
    if(lemma==True):
        wordnet_lemmatizer = WordNetLemmatizer()
        words_final = [wordnet_lemmatizer.lemmatize(word) for word in filtered_text]
    else:
        words_final = filtered_text
    return words_final
    
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def givKeywordsValue(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    cosine = round(get_cosine(vector1, vector2),3)*100
    return cosine

def grammar_check(text):
    # Inputs a string and returns 0 if grammar is very bad and 1 otherwise
    
    req = requests.get("https://api.textgears.com/check.php?text=" + text + "&key=JmcxHCCPZ7jfXLF6")
    no_of_errors = len(req.json()['errors'])
    if(no_of_errors>=5):
        return 0
    else:
        return 1
   
def cosine2(text1,text2):
    # Inputs two strings and calculates the cosine similarity (Bert embeddings)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence1 = text1.split(".")
    sentence2 = text2.split(".")
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_score2 = util.cos_sim(embeddings1, embeddings2).item()
    return cosine_score2
    
def word_distance_mover(text1_words,text2_words):
    # Inputs an array of words and returns the Word Mover Distance
    
    # No separate download -but 10 mins runnig time
    wv = api.load('word2vec-google-news-300')
    
    # If model is downloaded and placed in models folder,comment 98 and uncomment below
    #wv = gensim.models.KeyedVectors.load_word2vec_format("Group_51_models/GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    wmd = wv.wmdistance(text1_words,text2_words)
    return wmd
