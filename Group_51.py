import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from Group_51_helpers import *



def classifier(feature):

    # Since dataset is very small,model is trained everytime code is executed
    # Alternatively,the trained model could have been stored and loaded
    
    regressor = joblib.load("Group_51_models/custom_trained_random_forest1.joblib")
    
    y_pred = regressor.predict(feature)
    return y_pred

def subjective_eval():
    # Function for evaluating subjective answers
    
    print("\n\n**************   Evaluate Subjective Answers  *************\n\n")
    model_ans = input("Enter model answer : ")
    print("\n")
    student_ans = input("Enter student answer : ")
    print("\n")

    # Converting to lowercase 

    model_ans = model_ans.lower()
    student_ans = student_ans.lower()
    
    # Tokenizing - model_ans_sen is an array of sentences
    
    model_ans_sen = sent_tokenize(model_ans)
    student_ans_sen = sent_tokenize(student_ans)
    
    # Processing - Removing Punctuation and Stopwords
    # model_ans_filtered is an array of words without stopwords

    model_ans_filtered = process(model_ans_sen)
    student_ans_filtered = process(student_ans_sen)
    
    # Lemmatization - model_ans_words is array of words after lemmatizing
    
    model_ans_words = lemmatize(model_ans_filtered)
    student_ans_words = lemmatize(student_ans_filtered)
    
    # model_ans_final - one giant sentence with relevant words after proceesing
    model_ans_finalsen = " ".join(model_ans_words)
    student_ans_finalsen = " ".join(student_ans_words)
    
    # Similarity Functions !!!
    
    cosine_score1 = givKeywordsValue(text1=model_ans_finalsen, text2=student_ans_finalsen)
    cosine_score2 = cosine2(model_ans_finalsen,student_ans_finalsen)
    wmd = word_distance_mover(model_ans_words,student_ans_words)
    
    grammar = grammar_check(student_ans)
    
    
    # Use the ML model (Random forest for classification)
    features = np.zeros(3,)
    features[0] = cosine_score1/100
    features[1] = round(cosine_score2,3)
    features[2] = round(wmd,3)
    features.resize(1,3)
    class_label = classifier(features)
    label = class_label.item()
    if(grammar==0 and label>2):
        label = label-1
    
    print("The student's answer fetched a ",label)

          
if __name__ == '__main__':
    subjective_eval()


