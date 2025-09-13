import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import os
import pandas as pd
os.system("python -m spacy download en_core_web_sm")
import en_core_web_sm
import collections
import csv
import spacy
from sklearn.metrics.pairwise import cosine_similarity


# Task 1 (3 marks)
def stats_pos(csv_file_path):
    """Return the normalized frequency of all appeared part of speech in the questions and answers
    (namely the `sentence text` column) in the given csv file, respectively. Each of the resulting 
    two lists must be sorted alphabetically according to tags.
    Example:
    >>> stats_pos('dev_test.csv')
    output would look like [(ADV, 0.1), (NOUN, 0.21), ...], [(ADJ, 0.08), (ADV, 0.22), ...]
    """

    # Read CSV file and extract questions and answers
    unique_questions = []
    unique_answers = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            if question not in unique_questions:
                unique_questions.append(question)
            answer = row['sentence text']
            if answer not in unique_answers:
                unique_answers.append(answer)
    # Concatenate questions and answers
    questions_text = ' '.join(unique_questions)
    answers_text = ' '.join(unique_answers)
    
    #Using NLTK's sentence tokeniser before NLTK's word tokeniser. Code from practical week 7     
    questions_sent_tokenize = [word_tokenize(s) for s in sent_tokenize(questions_text)]
    answers_sent_tokenize = [word_tokenize(s) for s in sent_tokenize(answers_text)]
    
    #Using NLTK's part of speech tagger: pos_tag_sents and the "Universal" tagset.
    # code from practical week7
    questions_tagged_sents = nltk.pos_tag_sents(questions_sent_tokenize, tagset="universal")
    answers_tagged_sents = nltk.pos_tag_sents(answers_sent_tokenize, tagset="universal")
    
    #creating an empty list to get count the the pos tags using counter function
    
    questions_pos = []
    for s in questions_tagged_sents:
        for w in s:
            questions_pos.append(w[1])
    questions_counter = collections.Counter(questions_pos)
    
    #add the values of each pos tag to get the total count
    total_questions_counter= sum(questions_counter.values())
    
    #creating an empty dictionary to add pos tags with their normalized frequencies
    normalized_freq_questions= dict()
    
    #dividing the frequency of pos atg with total frequency
    for k in questions_counter:
        normalized_freq_1=questions_counter[k]/total_questions_counter
        normalized_freq_1 = round(normalized_freq_1,4)
        normalized_freq_questions[k]=normalized_freq_1
        
    #converting the dictionary into list with list items as a tuple of pos tag and nomalized frequency
        
    questions_pos_freq = [(tag, count) for tag, count in sorted(normalized_freq_questions.items())]
    
     #creating an empty list to get count the the pos tags using counter function
    
    answers_pos = []
    for s in answers_tagged_sents:
        for w in s:
            answers_pos.append(w[1])
    answers_counter = collections.Counter(answers_pos)
    
    #add the values of each pos tag to get the total count
    total_answers_counter= sum(answers_counter.values())
    
    #creating an empty dictionary to add pos tags with their normalized frequencies
    normalized_freq_answers= dict()
    
    #dividing the frequency of pos atg with total frequency
    for k in answers_counter:
        normalized_freq_2=answers_counter[k]/total_answers_counter
        normalized_freq_2 = round(normalized_freq_2,4)
        normalized_freq_answers[k]=normalized_freq_2
        
    #converting the dictionary into list with list items as a tuple of pos tag and nomalized frequency
        
    answers_pos_freq = [(tag, count) for tag, count in sorted(normalized_freq_answers.items())]
        
    
 
        
    return questions_pos_freq ,answers_pos_freq

 
        
    

# Task 2 (3 marks)
def stats_top_stem_ngrams(csv_file_path, n, N):
    """Return the N most frequent n-gram of stems together with their normalized frequency 
    for questions and answers, respectively. Each is sorted in descending order of frequency
    Example:
    >>> stats_top_stem_ngrams('dev_test.csv', 2, 5)
    output would look like [('what', 'is', 0.43), ('how', 'many', 0.39), ....], [('I', 'feel', 0.64), ('pain', 'in', 0.32), ...]
    """
    
    # Read CSV file and extract questions and answers
    unique_questions = []
    unique_answers = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            if question not in unique_questions:
                unique_questions.append(question)
            answer = row['sentence text']
            if answer not in unique_answers:
                unique_answers.append(answer)
    # Concatenate questions and answers
    questions_text = ' '.join(unique_questions)
    answers_text = ' '.join(unique_answers)
    
    #Using NLTK's sentence tokeniser before NLTK's word tokeniser. Code from practical week 7     
    questions_sent_tokenize = [word_tokenize(s) for s in sent_tokenize(questions_text)]
    answers_sent_tokenize = [word_tokenize(s) for s in sent_tokenize(answers_text)]
    
    #Stemming words in each sentence
    stemmer = nltk.PorterStemmer()
              
    question_stemmed=[[stemmer.stem(w) for w in s] for s in questions_sent_tokenize ]
    answer_stemmed=[[stemmer.stem(w) for w in s] for s in answers_sent_tokenize ]
    
    #creating an empty list to get ngrams of each sentence
    l1=[]
    for s in question_stemmed:
        n_grams_question=list(nltk.ngrams(s,n))
        l1.append(n_grams_question)
    #flattening the list to get list of all ngrams without any nested list     
    l2=[bigrams for i in l1 for bigrams in i] 
    
    #count the ngrams using counter function
    questions_counter = collections.Counter(l2)
    
    #add the values of each grams to get the total count
    total_questions_counter= sum(questions_counter.values())
    
    #creating an empty dictionary to add grams with their normalized frequencies
    normalized_freq_questions= dict()
    
    #dividing the frequency of each grams with total frequency
    for k in questions_counter:
        normalized_freq_1=questions_counter[k]/total_questions_counter
        normalized_freq_1 = round(normalized_freq_1,4)
        normalized_freq_questions[k]=normalized_freq_1
        
    # Sorting n-grams based on frequency
    sorted_ngrams_questions = sorted(normalized_freq_questions.items(), key =lambda x:x[1],reverse=True)

    # Selecting the top N n-grams
    top_ngrams_questions = sorted_ngrams_questions[:N]

    #creating an empty list to get ngrams of each sentence
    l3=[]
    for s in answer_stemmed:
        n_grams_answers=list(nltk.ngrams(s,n))
        l3.append(n_grams_answers)
    #flattening the list to get list of all ngrams without any nested list     
    l4=[bigrams for i in l3 for bigrams in i] 
    
    #count the ngrams using counter function
    answers_counter = collections.Counter(l4)
    
    #add the values of each grams to get the total count
    total_answers_counter= sum(answers_counter.values())
    
    #creating an empty dictionary to add grams with their normalized frequencies
    normalized_freq_answers= dict()
    
    #dividing the frequency of each grams with total frequency
    for k in answers_counter:
        normalized_freq_2=answers_counter[k]/total_answers_counter
        normalized_freq_2 = round(normalized_freq_2,4)
        normalized_freq_answers[k]=normalized_freq_2
        
    # Sorting n-grams based on frequency
    sorted_ngrams_answers = sorted(normalized_freq_answers.items(), key =lambda x:x[1],reverse=True)

    # Selecting the top N n-grams
    top_ngrams_answers = sorted_ngrams_answers[:N]

     
        
    return top_ngrams_questions, top_ngrams_answers


# Task 3 (2 marks)
def stats_ne(csv_file_path):
    """Return the normalized frequency of all named entity types for questions and answers, respectively.
    Each is sorted in descending order of frequency.
    Example:
    >>> stats_ne('dev_test.csv')
    output would look like [(DATE, 0.34), ....]
    """
    
    
    # Read CSV file and extract questions and answers
    unique_questions = []
    unique_answers = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            if question not in unique_questions:
                unique_questions.append(question)
            answer = row['sentence text']
            if answer not in unique_answers:
                unique_answers.append(answer)
    # Concatenate questions and answers
    questions_text = ' '.join(unique_questions)
    answers_text = ' '.join(unique_answers)
    
    # Load spaCy's English model
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 5000000
    
    # Process questions spaCy
    questions_doc = nlp(questions_text)
    
    #Extracting named entities
    question_ne=[]
    for ent in questions_doc.ents:
        question_ne.append(ent.label_)
        
    #counting their frequencies
    questions_ne_counter = collections.Counter(question_ne)
    
    #add the values of each named entity to get the total count
    total_questions_ne_counter= sum(questions_ne_counter.values())
    
    #creating an empty dictionary to named entities with their normalized frequencies
    normalized_freq_questions_ne= dict()
    
    #dividing the frequency of named entity with total frequency
    for k in questions_ne_counter:
        normalized_freq_1=questions_ne_counter[k]/total_questions_ne_counter
        normalized_freq_1 = round(normalized_freq_1,4)
        normalized_freq_questions_ne[k]=normalized_freq_1
        
    #converting the dictionary into list with list items as a tuple of pos tag and nomalized frequency
        
    questions_ne_freq = [(tag, count) for tag, count in sorted(normalized_freq_questions_ne.items())]
    
    # Process questions spaCy
    answers_doc = nlp(answers_text)
    
    #Extracting named entities
    answer_ne=[]
    for ent in answers_doc.ents:
        answer_ne.append(ent.label_)
        
    #counting their frequencies
    answers_ne_counter = collections.Counter(answer_ne)
    
    #add the values of each named entity to get the total count
    total_answers_ne_counter= sum(answers_ne_counter.values())
    
    #creating an empty dictionary to named entities with their normalized frequencies
    normalized_freq_answers_ne= dict()
    
    #dividing the frequency of named entity with total frequency
    for k in answers_ne_counter:
        normalized_freq_2=answers_ne_counter[k]/total_answers_ne_counter
        normalized_freq_2 = round(normalized_freq_2,4)
        normalized_freq_answers_ne[k]=normalized_freq_2
        
    #converting the dictionary into list with list items as a tuple of pos tag and nomalized frequency
        
    answers_ne_freq = [(tag, count) for tag, count in sorted(normalized_freq_answers_ne.items())]
   
    
    
    return questions_ne_freq ,answers_ne_freq


# Task 4 (2 marks)
def stats_tfidf(csv_file_path):
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Extract all unique questions
    unique_questions = df.drop_duplicates(subset='qid')
    
    # Questions and their corresponding answers
    questions = unique_questions['question'].tolist()
    all_sentences = df['sentence text'].tolist()
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the data
    vectorizer.fit(questions + all_sentences)  # Fit on both questions and answers
    question_vectors = vectorizer.transform(questions)
    sentence_vectors = vectorizer.transform(all_sentences)
    
    # Compute cosine similarities
    cosine_similarities = cosine_similarity(question_vectors, sentence_vectors)
    
    # Determine if the highest similarity sentence is an answer to the question
    match_count = 0
    for idx, question in enumerate(questions):
        # Get the index of the max similarity score for this question
        max_sim_index = cosine_similarities[idx].argmax()
        # Check if this index points to an answer sentence for this question
        if df.iloc[max_sim_index]['qid'] == unique_questions.iloc[idx]['qid'] and df.iloc[max_sim_index]['label'] == 1:
            match_count += 1
    
    # Calculate the ratio
    ratio = round(match_count / len(questions), 4)
    
    return ratio



# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    # print("---------Task 1---------------")
    # print(stats_pos('data/dev_test.csv'))
  
    # print("---------Task 2---------------")
    # print(stats_top_stem_ngrams('data/dev_test.csv', 2, 5))

    # print("---------Task 3---------------")
    # print(stats_ne('data/dev_test.csv'))

    # print("---------Task 4---------------")
    # print(stats_tfidf('data/dev_test.csv'))
  
