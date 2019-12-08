import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import math
from numpy import dot
from numpy.linalg import norm
import pickle

class TextSearch:
    def __init__(self):
        self.length = [] #length of each document after tokenization
        self.totalDocument = 0
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.RecipeData = pd.read_csv("recipe.csv")
        self.postings = defaultdict(dict) #count of each term in every document
        self.docFreq = defaultdict(int) #Number of documents a word appeared in
        self.bagOfWords = set() #bag of words from all documents

    #remove stop words from instructions and apply stemming and lemmatizing
    def tokenize(self,instructions):
        if (pd.isnull(instructions)==False):
            terms = re.sub('[?|$|.|!]','',instructions.lower())
            terms = re.sub('[^a-zA-Z\' ]','',terms)
            terms = terms.split(' ')
            filtered = [word for word in terms if not word in stopwords.words('english')]
            filtered = list(filter(lambda x: len(x)>1, filtered))
            filtered = [self.lemmatizer.lemmatize(term) for term in filtered]
            filtered = [self.stemmer.stem(term) for term in filtered]
            # print(filtered)
            return filtered
        else:
            return []

    #function to build the inverted index
    def BuildInvertedIndex(self):
        print('building inverted index.Please wait...')
        self.totalDocument  = self.RecipeData.shape[0]
        self.totalDocument = 50 #need to comment only for debuging
        for index in range(self.totalDocument):
            terms = self.tokenize(self.RecipeData.loc[index, 'instructions'])
            self.length.append(len(terms))
            unique_terms = set(terms)
            self.bagOfWords = self.bagOfWords.union(unique_terms)
            for term in unique_terms:
                self.postings[term][index] = terms.count(term)
        for term in self.bagOfWords:
            self.docFreq[term] = len(self.postings[term])
        print('done.')

    def InitializePickle(self):
        print('TextSearch: saving pickle files. Please wait...')
        pickle_f = open("textsearch_files/RecipeData.pickle","wb")
        pickle.dump(self.RecipeData,pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/postings.pickle","wb")
        pickle.dump(self.postings,pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/docFreq.pickle","wb")
        pickle.dump(self.docFreqreq,pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/bow.pickle","wb")
        pickle.dump(self.bagOfWords,pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/length.pickle","wb")
        pickle.dump(self.length,pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/totalDocument.pickle","wb")
        pickle.dump(self.totalDocument,pickle_f)
        pickle_f.close()
        print('done.')

    def loadPickle(self):
        print('TextSearch: Loading pickle files. Please wait...')
        pickle_f = open("textsearch_files/RecipeData.pickle","rb")
        self.RecipeData = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/postings.pickle","rb")
        self.postings = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/docFreq.pickle","rb")
        self.docFreq = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/bow.pickle","rb")
        self.bagOfWords = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/length.pickle","rb")
        self.length = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("textsearch_files/totalDocument.pickle","rb")
        self.totalDocument = pickle.load(pickle_f)
        pickle_f.close()
        print('done')
        # print(self.postings['onion'])

    def Search(self,query):
        cosine_sim = []
        query_token = self.tokenize(query)
        flag = False
        if query_token == []:
            return []
        else:
            for term in query_token:
                if term in self.bagOfWords:
                    flag = True
        if flag == False:
            return []

        Query_vector = self.QueryVec(query_token)

        for index in range(self.totalDocument):
            document_Vector = self.DocVector(query_token,index)
            similarity = self.CosineSimilarity(Query_vector,document_Vector)
            cosine_sim.append(similarity)
        sorted_similarity = np.argsort(cosine_sim)
        recipe_name = []
        recipe_instructions = []
        calculations = []
        cos_sim = []
        retrive_data = {}
        for result_index in range(10):
            recipe_name.append(self.RecipeData.loc[sorted_similarity[result_index],'title'])
            instructions = self.RecipeData.loc[sorted_similarity[result_index],'instructions']
            temp = []
            for i in range(len(query_token)):
                d_length = self.length[sorted_similarity[result_index]]
                # print('document length: '+str(d_length))
                try:
                    t_count = self.postings[query_token[i]][sorted_similarity[result_index]]
                except KeyError:
                    t_count = 0
                # print('term count: '+str(t_count))
                tf = t_count/d_length
                idf = self.Idf(query_token[i])
                tf_idf = tf*idf
                temp.append([str(query_token[i]),tf,idf,tf_idf])
                # print(str(query_token[i])+" tf: "+str(tf))
                # print(str(query_token[i])+" idf: "+str(idf))
                # print(str(query_token[i])+" tf-idf: "+str(tf_idf))
            recipe_instructions.append(instructions)
            calculations.append(temp)
            
        for i in sorted_similarity[0:10]:
            cos_sim.append(cosine_sim[i])
        retrive_data.update({"Recipe":recipe_name})
        retrive_data.update({"Instructions": recipe_instructions})
        retrive_data.update({"CosineSim":cos_sim})
        retrive_data.update({"Calculations":calculations})
        # print(retrive_data['CosineSim'])
        return  retrive_data

    #calculate cosign similarity of two tf-idf vector
    def CosineSimilarity(self,query_vec,doc_vec):
        return dot(query_vec,doc_vec)/(norm(query_vec)*norm(doc_vec))

    # create tf-idf weight vector of query term
    def QueryVec(self,query_token):
        unique_terms = set(query_token)
        query_length = len(query_token)
        vector = []
        for term in unique_terms:
            term_count = query_token.count(term)
            term_F = term_count / query_length
            print("++"+term + " " + str(term_F) )
            query_idf = self.Idf(term)
            tf_idf = term_F * query_idf
            vector.append(tf_idf)
        return vector

    # create tf-idf weight vector of query term for document
    def DocVector(self,query_token,id):
        unique_terms = set(query_token)
        document_vector = []
        for q_term in unique_terms:
            if q_term in self.bagOfWords:
                if id in self.postings[q_term]:
                    tf = self.postings[q_term][id] / self.length[id]
                    tf_idf = tf * self.Idf(q_term)

                    document_vector.append(tf_idf)
                else:
                    document_vector.append(0.0)
            else:
                document_vector.append(0.0)
        return document_vector

    def Idf(self,term):
        if term in self.bagOfWords:
            return 1.0 + math.log(float(self.totalDocument)/self.docFreq[term]) 
        else:
            return 1.0

