# -*- coding: utf-8 -*-
"""
Quick and dirty implimentation of a simple
bigram character tokenizer to a vector space
with cosine similarity with Numpy
"""

import numpy as np
import pandas as pd

def tokenize_bigram(string2tokenize):
    #create dictionary to identify columns in token list
    dict_lookup={}
    characters="ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
    count=0
    for i in characters:
        for j in characters:
            dict_lookup.update({''+i+j:count})
            count+=1

    #initialize the output list
    string_token=[0]*len(characters)**2
    
    #actually tokenize the string
    for i in range(2,len(string2tokenize)):
        string_token[dict_lookup[string2tokenize[(i-2):i].upper()]]=string_token[dict_lookup[string2tokenize[(i-2):i].upper()]]+1
    return string_token

def cos_sim(token1,token2):
    #converst the token lists to vectors
    vec1=np.array(token1)
    vec2=np.array(token2)
    #compute and return the cosine of angle between them
    return np.dot(vec1,vec2)/(np.sqrt(np.dot(vec1,vec1))*np.sqrt(np.dot(vec2,vec2)))

def find_doc(query,doc_list):
    token_df=pd.DataFrame(columns=['ID','similarity_score'])
    for i in range(len(doc_list)):
        token_df=token_df.append({'ID':i,'similarity_score':cos_sim(tokenize_bigram(query),doc_list[i])},ignore_index=True)
    return token_df.sort_values(by='similarity_score',ascending=False)

#Some data
token_list=[]
token_list.append(tokenize_bigram('Oliver Bobbitt O O Bobbitt O Orange bobbitt'))
token_list.append(tokenize_bigram('Patrick Bobbitt Patrick Bobbitt patrick a bobbit p a bobit'))
token_list.append(tokenize_bigram('Heather Howland Bobbitt H H Bobbitt Heather H Bobbitt Heather Howland'))
token_list.append(tokenize_bigram('Jim Neel Jimmy Neel James a neel Jimmy neil'))
token_list.append(tokenize_bigram('11 River Drive 11 River Dr'))
token_list.append(tokenize_bigram('2475 Findley Avenue 2479 Findley Ave 2479 Findlie Ave'))
token_list.append(tokenize_bigram('1166 Ashland Avenue 1166 Ashland Ave'))
token_list.append(tokenize_bigram('14 River Drive 14 River Dr'))
token_list.append(tokenize_bigram('Tom paul Howland Thomas Howland'))
