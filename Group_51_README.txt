only script to run for testing - Group_51.py

nltk.download commands in helper_functions.py only need to be run on first execution

Running time of subjective_evaluator is a bit too much for the first execution because of the Sentence transformer model

** line 97 of Group_51_helpers.py **

api.load - takes 11 minutes to run - not recommended

Alternate - download the 'word2vec-google-news-300' model (1.6 GB) and place into the models folder. Now uncomment the api.load line and uncomment line 98. Download link found below :

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g

For easier usage and prevent dependency issues, use the following notebook to test
https://colab.research.google.com/drive/1qu5bKzhH6vZR_Um5U6cPGj2BaoNomty5?usp=sharing
(make sure custom_trained_random_forest1.joblib is uploaded to your drive)