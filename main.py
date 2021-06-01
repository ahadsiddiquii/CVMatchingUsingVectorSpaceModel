import copy
import json
from flask import Flask,jsonify,request
import flask
from contractions import contractions
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import re
import math

app = Flask(__name__)

term_document_dictionary = {}
filename = 'dataset/cvstxt'
total_documents = 0
list_of_documents = []
document_indexes = {}
list_of_documentID = []
euclidean_lengths_for_each_doc = []
cosineSim = []
output = []
output_dictionary = {}
final_list = []


@app.route('/cvs', methods = ['GET'])
def result():
    for item in output:
        final_list.append(document_indexes[output_dictionary[item]])
    return jsonify({'Matched CVs' : final_list})

@app.route('/collect', methods = ['POST'])
def collectingDocuments():
    request_data = request.data
    request_data = json.loads(request_data.decode('utf-8'))
    query = request_data['querydocument']
    files = os.listdir(filename + '/')
    global total_documents
    total_documents = 22
    open(os.path.join(filename, str(total_documents) + '.txt'), "w").close()
    f = open(os.path.join(filename, str(total_documents) + '.txt'), "w")
    f.write(query)
    f.close
    document_count = int(0)
    for file in files:
        #handling the encoding of the text files
        f = open(filename+'/' + file, "r", encoding="utf8")
        file_contents = f.read()
        new_contents = 0
        print(filename+'/' + file)
        f.close()
        # sending the document for tokenization
        final_tokenized_terms = []
        final_tokenized_terms = tokenization(file_contents)
        final_tokenized_terms.sort()
        # saving document name
        list_of_documents.append(file)
        document_indexes[document_count] = file
        list_of_documentID.append(document_count)
        # print(final_terms_with_stops)
        for word in final_tokenized_terms:
            if word in term_document_dictionary.keys():
                # if key exists then append the document list
                term_document_dictionary[word][document_count] = 0
            else:
                # if key does not exist then create a document list for the key
                term_document_dictionary[word] = [0] * (total_documents+3)
                term_document_dictionary[word][document_count] = 0
                #calculating term frequencies
        # print(final_terms_with_stops.count('adobe'))
        for term in final_terms_with_stops:
            if term in term_document_dictionary.keys():
                term_document_dictionary[term][document_count] = term_document_dictionary[term][document_count] + 1
        document_count += 1
    
    calculateDocumentFrequency()
    calculateInverseDocumentFrequency()
    normalizingTermFrequency()
    calculate_ntf_idf()
    cosine_similarity()
    # print(term_document_dictionary)
    documents = []
    print(len(cosineSim))
    for item in range(1,total_documents):
        if cosineSim[item] > 0.01:
            documents.append(item+1)
    
    print(cosineSim)
    print(documents)
    for item in documents:
        output_dictionary[cosineSim[item-1]] = item
        output.append(cosineSim[item-1])

    
    output.sort(reverse=True)
    print(output)
    print(output_dictionary)
      
    return ""

#tokenizing
def tokenization(document):
    #CASEFOLDING
    # casefolding or lower casing the whole document
    document_to_work = document.casefold()

    # OPENING CONTRACTIONS
    # making a list for contractions processing
    list_terms = document_to_work.split()
    list_terms_no_contractions = list()
    for word in list_terms:
        if word in contractions:
            # using an imported dictionary for contractions
            list_terms_no_contractions.append(contractions[word])
        else:
            list_terms_no_contractions.append(word)


    # REMOVING PUNCTUATIONS
    document_no_punctuation = ' '.join(list_terms_no_contractions)
    document_no_punctuation = re.sub(r'[^\w\s]', '', document_no_punctuation)

    # REMOVING FINAL LEFT OVER WHITESPACES
    finalised_terms_with_stopwords = document_no_punctuation.split()
    finalised_terms_with_stopwords.sort()

    # print(finalised_terms_with_stopwords)
    global final_terms_with_stops
    final_terms_with_stops = [0]
    final_terms_with_stops = finalised_terms_with_stopwords
    

    # REMOVING STOP WORDS AND DUPLICATE WORDS
    # opening stopwords file
    f = open('dataset/Stopword-List.txt', 'r', encoding='utf8')
    stop_words = f.read()
    stop_list = stop_words.split()
    f.close()

    # removing stop words
    finalised_terms_without_stopwords = list(set(finalised_terms_with_stopwords).difference(set(stop_list)))
    finalised_terms_without_stopwords = list(finalised_terms_without_stopwords)
    return finalised_terms_without_stopwords

#calculating document frequency
def calculateDocumentFrequency():
    for term in term_document_dictionary:
        count = total_documents + 1
        listOfDocs = term_document_dictionary[term]
        df = 0
        for item in range(0,count):
            if listOfDocs[item] != 0:
                df = df + 1
        term_document_dictionary[term][count] = df

#calculating inverse document frequency
def calculateInverseDocumentFrequency():
    for term in term_document_dictionary:
        df_location = total_documents + 1 #27
        idf_location = total_documents + 2 #28
        df = term_document_dictionary[term][df_location]
        idf = math.log10(total_documents/df)
        term_document_dictionary[term][idf_location] = idf


#length normalizing term frequencies
def normalizingTermFrequency():
    #0-49
    max_list = list()
    count = total_documents + 1
    document_id = 0
    tfmaxOfDoc = 0
    alpha = 0.0005
    tf = 0
    ntf = 0
    for iter in range(0,count):
        square_for_euclidean_doc = []
        for item in term_document_dictionary:
            square_for_euclidean_doc.append(term_document_dictionary[item][iter] ** 2)
        euc_len = math.sqrt(sum(square_for_euclidean_doc))
        euclidean_lengths_for_each_doc.append(euc_len)

    for term in term_document_dictionary:
        for iter in range(1,count):
            tf = term_document_dictionary[term][iter]
            ntf = float(tf/euclidean_lengths_for_each_doc[iter-1])
            term_document_dictionary[term][iter] = ntf
            # ntf = alpha+((1-alpha)*(float(tf/max_list[iter-1])))
            # term_document_dictionary[term][iter] = ntf

#ntf-idf for weighting
def calculate_ntf_idf():
    count = total_documents + 1
    idf_location = total_documents + 2
    ntf_idf = 0
    for term in term_document_dictionary:
        for iter in range(0,count):
            ntf = term_document_dictionary[term][iter]
            idf = term_document_dictionary[term][idf_location]
            ntf_idf = ntf * idf
            term_document_dictionary[term][iter] = ntf_idf


#cosine similarity
def cosine_similarity():
    count = total_documents+1
    query_location = 0
    query_ntf_idf = []
    for item in term_document_dictionary:
        query_ntf_idf.append(term_document_dictionary[item][query_location])
    # print(len(query_ntf_idf))
    square_for_euclidean = list()
    for i in query_ntf_idf:
        square_for_euclidean.append(i**2)
    euclidean_length_query = math.sqrt(sum(square_for_euclidean))

    for iter in range(1,count):
        ntf_idf_of_doc = []
        for term in term_document_dictionary:
            ntf_idf_of_doc.append(term_document_dictionary[term][iter])
        # ntf_idf_of_doc
        #dot product
        dotProduct = float(0)
        for item in range(0,len(ntf_idf_of_doc)):
             dotProduct = dotProduct + (query_ntf_idf[item]*ntf_idf_of_doc[item])

        square_for_euclidean_doc = []
        for i in ntf_idf_of_doc:
            square_for_euclidean_doc.append(i ** 2)

        euclidean_length_doc = math.sqrt(sum(square_for_euclidean_doc))
        if (euclidean_length_doc*euclidean_length_query) !=0:
            cosSim = dotProduct/(euclidean_length_doc*euclidean_length_query)
            cosineSim.append(cosSim)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port =5000)