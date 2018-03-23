# Baseline Python3 processing script for text version of CALL shared task v2
#
# The script reads a revised version of the XML prompt/response grammar supplied with the download
# and uses it to process the text task training data spreadsheet.
# 
# A prompt/recognition_result pair is 
#   accepted if all validation rules apply
#   rejected otherwise

# ---------------------------------------------------

# Example of running the script (all files in same directory as script, except for google news model):

# $ python3 rule_based_text_processor.py
# Read XML grammar: referenceGrammar_v3.0.0.xml
# Reading and processing spreadsheet: scst2_training_data_all_text.csv
# 
# INCORRECT UTTERANCES (250)
# CorrectReject    170
# GrossFalseAccept 16*3 = 48
# PlainFalseAccept 64
# RejectionRate    0.60

# CORRECT UTTERANCES (750)
# CorrectAccept    695
# FalseReject      55
# RejectionRate    0.07

# D                8.22

import os
import collections
import csv
import sys
import re
import xml.etree.ElementTree as ET
import gensim
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import multiprocessing
from joblib import Parallel, delayed
import math

input_csv = sys.argv[1]
cem_similarity_treshold = float(sys.argv[2]) #6.5
facebook_similarity_treshold = float(sys.argv[3]) #0.9

cem_model_file = "csuk_doc2vec.model"
facebook_model_file = "/home/shared/models/wiki.simple.vec"
reference_grammar = "referenceGrammar_v3.0.0.xml"
reference_grammar_pos = "valid_sentence_orders_all.pos"

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#
### --- Read grammar and files---
#
def read_grammar(reference_grammar):
    tree = ET.parse(reference_grammar)
    root = tree.getroot()
    dictionary = { get_prompt(unit): get_responses(unit) for unit in root.findall('prompt_unit') }
    return ( dictionary, dictionary.keys() )

def get_prompt(unit):
    prompt = unit.find('prompt').text
    return prompt

def get_responses(unit):
    return [ response.text for response in unit.findall('response') ]

def read_and_process_spreadsheet(input_csv, grammar_dic, known_prompts, tokenized_grammar):
    global scores
    with open(input_csv, 'r', encoding="utf-8") as csv_infile:
        reader = csv.reader(csv_infile, delimiter='\t', quotechar='"')
        next(reader)
        decisions = []
        decisions.append(Parallel(n_jobs=20)(delayed(process_spreadsheet_row)(row, grammar_dic, known_prompts, tokenized_grammar) for row in reader))
        for decision_sub_list in decisions:
            for row in decision_sub_list:
                decision, language_correct_gs, meaning_correct_gs, prompt, rec_result, transcription = row
                score_decision(decision, language_correct_gs, meaning_correct_gs, prompt, rec_result, transcription)

def is_header_row(row):
    return ( row[0] == 'Id' )

def read_tokenized_grammar(reference_grammar_pos):
    tokenized_grammar = set()
    with open(reference_grammar_pos) as f:
        content = f.readlines()
        for line in content:
            tokenized_grammar.add(line.strip())
    return tokenized_grammar

#
### --- Processing and classification ---
#
def process_spreadsheet_row(row, grammar_dic, known_prompts, tokenized_grammar):
    id = row[0]
    prompt = row[1]
    wav_file = row[2]
    rec_result = row[3]
    transcription = row[4]
    language_correct_gold_standard = row[5]
    meaning_correct_gold_standard = row[6]
    decision = classification(prompt, rec_result, grammar_dic, known_prompts, tokenized_grammar)
    return decision, language_correct_gold_standard, meaning_correct_gold_standard, prompt, rec_result, transcription

def classification(prompt, rec_result, grammar_dic, known_prompts, tokenized_grammar):
    if ( prompt in known_prompts ):
        rec_result_tuples = nltk.pos_tag(word_tokenize(clean_kaldi_tags(rec_result)))
        if rec_result_tuples:
            return 'accept' if (
                check_meaning(prompt, rec_result_tuples, grammar_dic)) and check_grammatic(prompt, rec_result, rec_result_tuples, tokenized_grammar, grammar_dic) else 'reject'
        else:
            return 'reject'
    else:
        print("*** Error: prompt not in XML dictionary: '" + prompt + "'") 
        return False

def check_grammatic(prompt, rec_result, rec_result_tuples, tokenized_grammar, grammar_dic):
    global rejection_state

    words, tags = zip(*rec_result_tuples)
    rec_result_tags = " ".join(tags)

    pos_structure_matches = False
    for pos_sentence in tokenized_grammar:
        if rec_result_tags.endswith(pos_sentence):
            pos_structure_matches = True
            break
    if not pos_structure_matches:
        rejection_state = "pos_structure"
        return False

    similarity_treshold_reached = True
    if use_gensim:
        similarity_treshold_reached = validate_similarity(rec_result, grammar_dic[prompt])
    if not similarity_treshold_reached:
        rejection_state = "similarity"
    return pos_structure_matches and similarity_treshold_reached
    
def check_meaning(prompt, rec_result_tuples, grammar_dic):
    global rejection_state

    filtered_rec_result_tuples = filter_stopwords(rec_result_tuples)
    if not filtered_rec_result_tuples and not rec_result_tuples:
        return False
    elif not filtered_rec_result_tuples:
        filtered_rec_result_tuples = rec_result_tuples
    prompt_responses = grammar_dic[prompt]
    rejection_state ="noun"
    tokenized_responses = []
    for response in prompt_responses:
        tokenized_responses.append(nltk.pos_tag(word_tokenize(clean_kaldi_tags(response))))

    meaning_correct = True
    if meaning_correct: 
        rejection_state="nouns"
        meaning_correct = validate_nouns(filtered_rec_result_tuples, tokenized_responses, prompt_responses)
    if meaning_correct: 
        rejection_state="determiner"
        meaning_correct = validate_determiner(rec_result_tuples, tokenized_responses)
    if meaning_correct: 
        rejection_state="prepositions"
        meaning_correct = validate_preposition(rec_result_tuples, tokenized_responses)
    if meaning_correct: 
        rejection_state="verb"
        meaning_correct = validate_verbs(filtered_rec_result_tuples, rec_result_tuples, tokenized_responses)
    if meaning_correct: 
        rejection_state="adj"
        meaning_correct = validate_adjectives(filtered_rec_result_tuples, tokenized_responses)
    if meaning_correct: 
        rejection_state="num"
        meaning_correct = validate_numbers(filtered_rec_result_tuples, tokenized_responses)
    if meaning_correct:
        rejection_state="grammar" 
    return meaning_correct

def validate_similarity(rec_result, responses):
    global facebook_similarity_treshold
    global cem_similarity_treshold
    global facebook_model
    global cem_model
    
    cem_distances = set()
    facebook_distances = set()
    for response in responses:
        if use_cem_model:
            cem_distances.add(cem_model.wmdistance(clean_kaldi_tags(rec_result), response))
        if use_facebook_model:
            facebook_distances.add(facebook_model.wmdistance(clean_kaldi_tags(rec_result), response))
    
    if not cem_distances:
        cem_distances.add(-99999)
    if not facebook_distances:
        facebook_distances.add(-99999)

    return min(cem_distances) < cem_similarity_treshold or min(facebook_distances) < facebook_similarity_treshold

def validate_determiner(rec_result_tuples, tokenized_responses):
    words, tags = zip(*rec_result_tuples)
    if not 'DT' in tags: return True

    determiners = set()
    the_preceded_nouns = set()

    for tokenized_response in tokenized_responses:
        for idx, response_tuple in enumerate(tokenized_response):
            if response_tuple[1] == 'DT' and (response_tuple[0] == 'a' or response_tuple[0] == 'the'):
                determiners.add(response_tuple[0])
                if response_tuple[0] == 'the' and (idx+1) < len(tokenized_response):
                    the_preceded_nouns.add(tokenized_response[idx+1][0])

    # bspw: can you show me THE way to [kein THE] oxford street
    # wenn in einer response ein "the" vorkommt merke das darauffolgende wort
    # prüfe für jedes the+noun in der utterance ob das noun in den responses auch ein the hat
    if the_preceded_nouns and 'the' in words:
        for idx, word in enumerate(words):
            if word == 'the' and (idx+1) < len(words):
                if not words[idx+1] in the_preceded_nouns:
                    return False

    if len(determiners) == 1 and ('a' in words or 'the' in words):
        det = determiners.pop()
        return det in words
    else:
        return True

def validate_preposition(rec_result_tuples, tokenized_responses):
    prepositions_correct = True
    words, tags = zip(*rec_result_tuples)
    if not 'IN' in tags: return True

    by_with_issue = set()
    at_on_issue = set()
    for tokenized_response in tokenized_responses:
        for response_tuple in tokenized_response:
            if response_tuple[1] == 'IN' and (response_tuple[0] == 'by' or response_tuple[0] == 'with'):
                by_with_issue.add(response_tuple[0])
            elif response_tuple[1] == 'IN' and (response_tuple[0] == 'at' or response_tuple[0] == 'on'):
                at_on_issue.add(response_tuple[0])

    if len(by_with_issue) == 1 and ('by' in words or 'with' in words):
        prep = by_with_issue.pop()
        prepositions_correct = prep in words
    elif len(at_on_issue) == 1 and ('at' in words or 'on' in words):
        prep = at_on_issue.pop()
        prepositions_correct = prep in words
    
    return prepositions_correct

def validate_nouns(filtered_rec_result_tuples, tokenized_responses, prompt_responses):
    nouns_correct = contains_compound_noun_if_necessary(filtered_rec_result_tuples, prompt_responses, tokenized_responses)
    #return nouns_correct
    #### only for Dfull
    if not nouns_correct: return False

    valid_noun_set = set()
    lemmatizer = WordNetLemmatizer()
    for tokenized_response in tokenized_responses:
        for response_tuple in tokenized_response:
            if response_tuple[1] == 'NN' or response_tuple[1] == 'NNS':
                lemmatized_response_word = lemmatizer.lemmatize(response_tuple[0], wordnet.NOUN)
                valid_noun_set.add(lemmatized_response_word)


    nouns_correct = True
    for word_tuple in filtered_rec_result_tuples:
        if (word_tuple[1] == 'NN' or word_tuple[1] == 'NNS'):
            if lemmatizer.lemmatize(word_tuple[0], wordnet.NOUN) not in valid_noun_set:
                nouns_correct = False
                break
            
    return nouns_correct

def validate_verbs(filtered_rec_result_tuples, rec_result_tuples, tokenized_responses):
    verbs_correct = True

    for idx, word_tuple in enumerate(filtered_rec_result_tuples):
        if (word_tuple[1] == 'VB' or word_tuple[1] == 'VBD' or word_tuple[1] == 'VBG' or word_tuple[1] == 'VBN' or word_tuple[1] == 'VBP' or word_tuple[1] == 'VBZ'):
            if verbs_correct:
                verbs_correct = is_word_in_responses(word_tuple, tokenized_responses, wordnet.VERB)
            if not verbs_correct:
                return False
    return verbs_correct

def validate_adjectives(filtered_rec_result_tuples, tokenized_responses):
    adjectives_correct = True

    for word_tuple in filtered_rec_result_tuples:
        if (word_tuple[1] == 'JJ' or word_tuple[1] == 'JJR' or word_tuple[1] == 'JJS'):
            adjectives_correct = is_word_in_responses(word_tuple, tokenized_responses, wordnet.ADJ)
            if not adjectives_correct:
                return False
            
    return adjectives_correct

def validate_numbers(filtered_rec_result_tuples, tokenized_responses):
    numbers_correct = True

    for word_tuple in filtered_rec_result_tuples:
        if (word_tuple[1] == 'CD'):
            numbers_correct = is_number_in_responses(word_tuple, tokenized_responses)
            if not numbers_correct:
                return False
            
    return numbers_correct

#
### --- Convenience methods ---
#

def contains_compound_noun_if_necessary(filtered_rec_result_tuples, prompt_responses, tokenized_responses):
    noun_index = 0
    compound_noun_dict = dict()
    for tokenized_response in tokenized_responses:
        response_has_compound_noun = False
        noun_dict = dict()
        for tuple_index, response_tuple in enumerate(tokenized_response):
            if (response_tuple[1] == 'NN' or response_tuple[1] == 'NNS' or response_tuple[1] == 'JJ') and response_tuple[0] != "i" and response_tuple[0] != "please" and response_tuple[0] != "yes":
                noun_dict[tuple_index] = response_tuple[0]

        for key in noun_dict.keys():
            # if there is a related noun but the array is empty then add both nouns (for NN NN)
            if (key-1) in noun_dict and noun_index not in compound_noun_dict:
                response_has_compound_noun = True
                compound_noun_dict[noun_index] = noun_dict[key-1] + " " + noun_dict[key]
            # if there is a related noun which is already in the array then add the new one (for NN NN NN...)
            elif (key-1) in noun_dict and noun_index in compound_noun_dict:
                compound_noun_dict[noun_index] += " " + noun_dict[key]
            # if there is no related noun and the current array index is not empty increase it
            elif (key-1) not in noun_dict and noun_index in compound_noun_dict:
                noun_index += 1

        if not response_has_compound_noun:
            # there are responses, without compound noun
            return True
        
        noun_index += 1

    compound_noun_set = set(compound_noun_dict.values())
    compount_nouns_1 = sorted(compound_noun_set) # sorts normally by alphabetical order
    compount_nouns_1.sort(key=len, reverse=True) # sorts by descending length
    if compount_nouns_1:
        # check for match in rec resultx
        words, tags = zip(*filtered_rec_result_tuples)
        rec_result = " ".join(words)
        for compound_noun in compount_nouns_1:
            if compound_noun in rec_result:
                return True
        return False

    # no compound noun needed in rec result
    return True

def is_number_in_responses(current_rec_tuple, tokenized_responses):
    for tokenized_response in tokenized_responses:
        words, tags = zip(*tokenized_response)
        if current_rec_tuple[0] in words:
            return True
    return False

def is_word_in_responses(current_rec_tuple, tokenized_responses, wordnet_type):
    lemmatizer = WordNetLemmatizer()
    lemmatized_word = lemmatizer.lemmatize(current_rec_tuple[0], pos=wordnet_type)
    for tokenized_response in tokenized_responses:
        for response_tuple in tokenized_response:
            if response_tuple[1] == current_rec_tuple[1]:
                lemmatized_response_word = lemmatizer.lemmatize(response_tuple[0], pos=wordnet_type)
                if lemmatized_response_word == lemmatized_word:
                    return True
    return False

def filter_stopwords(rec_result_tuples):
    stop_words = set(stopwords.words('english'))
    filtered_tuples = []
     
    for tuple in rec_result_tuples:
        if tuple[0] not in stop_words and tuple[0] != 'please' and tuple[1] != 'UH':
            filtered_tuples.append(tuple)
    return filtered_tuples

def clean_kaldi_tags(rec_result):
    words = rec_result.split(" ")
    sentence = []
    for idx, word in enumerate(words):
        word = re.sub(r"([A-Za-zöäüÖÄÜ]+[*][v])", "", word)
        word = re.sub(r"([A-Za-zöäüÖÄÜ]+[*][a])", "", word)
        word = re.sub(r"([A-Za-zöäüÖÄÜ]+[*][x])", "", word)
        word = re.sub(r"([*][z])", "", word)
        word = re.sub(r"-xxx-", "", word)
        word = re.sub(r"-xxx", "", word)
        word = re.sub(r"xxx", "", word)
        word = re.sub(r"ggg", "", word)
        
        # fix thinking words
        word = re.sub(r"ah", "" ,word)

        # fix stuttering word duplication
        if (idx+1) < len(words) and word == words[idx+1]:
            word = ""

        if word:
            sentence.append(word)
    return " ".join(sentence)

def match_tuple_in_responses(tuple, responses_pos):
    for response in responses_pos:
        for resp_tuple in response:
            if tuple[0] == resp_tuple[0]:
                return True
    return False

def tokenize_list(sentences):
    ret = []
    for sentence in sentences:
        ret.append(nltk.pos_tag(word_tokenize(sentence)))
    return ret


#
### --- Decision making and print score ---
#

def two_digits(x):
    if x == 'undefined':
        return 'undefined'
    else:
        return ( "%.2f" % x )


# Compare decision with gold standard judgements for language and meaning
def score_decision(decision, language_correct_gs, meaning_correct_gs, prompt, rec_result, transcription):
    global rejection_state
    global scores
    global k

    if ( decision == 'accept' and language_correct_gs == 'correct' ):
        result = 'CorrectAccept'
    elif ( decision == 'accept' and meaning_correct_gs == 'incorrect' ):
        result = 'GrossFalseAccept'
        rec_result_tuples = nltk.pos_tag(word_tokenize(clean_kaldi_tags(rec_result)))
        print("GFA: " + str(rec_result_tuples) + "                  || " + prompt + " || " + transcription)
    elif ( decision == 'accept' ):
        result = 'PlainFalseAccept'
        rec_result_tuples = nltk.pos_tag(word_tokenize(clean_kaldi_tags(rec_result)))
        print("PFA: " + str(rec_result_tuples) + "                  || " + prompt + " || " + transcription)
    elif ( decision == 'reject' and language_correct_gs == 'incorrect' ):
        result = 'CorrectReject'
    else:
        result = 'FalseReject'
        rec_result_tuples = nltk.pos_tag(word_tokenize(clean_kaldi_tags(rec_result)))        
        print("FR (" + rejection_state + "): " + str(rec_result_tuples) + "                  || " + prompt)
    scores[result] = scores[result] + 1
    return result

def print_scores(scores):
    k = 3.0
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA
    
    if ( CR + FA ) > 0 :
        IncorrectRejectionRate = CR / ( CR + FA )
    else:
        IncorrectRejectionRate = 'undefined'

    if ( FR + CA ) > 0 :
        CorrectRejectionRate = FR / ( FR + CA )
    else:
        CorrectRejectionRate = 'undefined'

    if ( CorrectRejectionRate != 'undefined' and IncorrectRejectionRate != 'undefined' ) :
        D = IncorrectRejectionRate / CorrectRejectionRate 
    else:
        D = 'undefined'

    print('\nINCORRECT UTTERANCES (' + str(Incorrect) + ')' )
    print('CorrectReject    ' + str(CR) )
    print('GrossFalseAccept ' + str(GFA) + '*' + str(k) + ' = ' + str(GFA * k) )
    print('PlainFalseAccept ' + str(PFA) )
    print('RejectionRate    ' + two_digits(IncorrectRejectionRate) )

    print('\nCORRECT UTTERANCES (' + str(Correct) + ')')
    print('CorrectAccept    ' + str(CA) )
    print('FalseReject      ' + str(FR) )
    print('RejectionRate    ' + two_digits(CorrectRejectionRate) )

    print('\nD                ' + two_digits(D) )

def init_scores():
    return {'CorrectAccept': 0, 'GrossFalseAccept': 0, 'PlainFalseAccept': 0, 'CorrectReject': 0, 'FalseReject': 0}

def do_all_processing():
    global facebook_model
    print("Read XML grammar: " + reference_grammar)
    ( grammar_dic, known_prompts ) = read_grammar(reference_grammar)
    print("Creating tokenized grammar ")
    tokenized_grammar = read_tokenized_grammar(reference_grammar_pos)
    print("Reading and processing spreadsheet: " + input_csv)
    read_and_process_spreadsheet(input_csv, grammar_dic, list(known_prompts), tokenized_grammar)
    print_scores(scores)

use_gensim = True
use_cem_model = cem_similarity_treshold > 0
use_facebook_model = facebook_similarity_treshold > 0
if use_gensim:
    print("Initializing gensim")
    if use_cem_model:
        cem_model = gensim.models.doc2vec.Doc2Vec.load(cem_model_file)
    if use_facebook_model:
        facebook_model = gensim.models.KeyedVectors.load_word2vec_format(facebook_model_file, binary=False)  

rejection_state = ""   
scores = init_scores()
do_all_processing()