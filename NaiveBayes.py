import json
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from collections import Counter
import pickle

class NaiveBayes():
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.RecipeData = []
        self.ClassFreq = defaultdict(int) # store classfrequency/ Probablity
        self.ClassTermCount = defaultdict(set)  # to store each class has how may total term
        self.TermClassFrequency = defaultdict(dict) # to store count of term in each class
        self.totalDocument = 0
        self.length = []
        self.unique_terms = set([]) # store number of unique term

    def tokenize(self,ingredients):
        if len(ingredients)==0:
            return []
        else:
            tokens = []
            for ingredient in ingredients:
                terms = ingredient.lower().split()
                for word in terms:
                    if word not in stopwords.words('english'):
                        tokens.append(word)
            tokens = [self.lemmatizer.lemmatize(term) for term in tokens]
            tokens = [self.stemmer.stem(term) for term in tokens]
            return tokens

    def InitializePickle(self):
        print('Classifier: saving pickle files. Please wait...')
        pickle_f = open("classifier_files/classF.pickle","wb")
        pickle.dump(self.ClassFreq,pickle_f)
        pickle_f.close()

        pickle_f = open("classifier_files/termClassFrequency.pickle","wb")
        pickle.dump(self.TermClassFrequency,pickle_f)
        pickle_f.close()

        pickle_f = open("classifier_files/classTermCount.pickle","wb")
        pickle.dump(self.ClassTermCount,pickle_f)
        pickle_f.close()

        pickle_f = open("classifier_files/unique_terms.pickle","wb")
        pickle.dump(self.unique_terms,pickle_f)
        pickle_f.close()
        print('done.')
    
    def LoadPickle(self):
        print('Classifier: Loading pickle files. Please wait...')
        pickle_f = open("classifier_files/classF.pickle","rb")
        self.ClassFreq = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("classifier_files/termClassFrequency.pickle","rb")
        self.TermClassFrequency = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("classifier_files/classTermCount.pickle","rb")
        self.ClassTermCount = pickle.load(pickle_f)
        pickle_f.close()

        pickle_f = open("classifier_files/unique_terms.pickle","rb")
        self.unique_terms = pickle.load(pickle_f)
        pickle_f.close()
        print('done.')

    def Initialize(self):
        print('initializing classifer. please wait...')
        jsonfile = r'./train.json'
        with open(jsonfile) as train_json:
            self.RecipeData = json.load(train_json)

        self.totalDocument = len(self.RecipeData)
        for index in range(self.totalDocument):
            current_class = self.RecipeData[index]['cuisine']
            if len(current_class)==0:
                continue
            terms = self.tokenize(self.RecipeData[index]['ingredients'])
            self.length.append(len(terms))
            self.ClassFreq[current_class] = self.ClassFreq[current_class] + 1
            u_term = Counter(terms).keys()
            u_count = list(Counter(terms).values())

            term_index = 0
            for term in u_term:
                self.ClassTermCount[current_class].add(term)
                self.TermClassFrequency[term][current_class] = self.TermClassFrequency[term].get(current_class,0) + u_count[term_index]
                term_index += 1

            self.unique_terms.update(set(terms))
        for key in self.ClassFreq:
            self.ClassFreq[key] = self.ClassFreq[key] / self.totalDocument
        print('done')

    def CalculateTermProbability(self,query):
        terms = self.tokenize(query)
        result = defaultdict()
        for key in self.ClassFreq:
            prob = self.ClassFreq[key]
            for term in terms:
                #P(y|x1,x2,…..xn ) = P(x1|y)P(x2|y)..P(xn|y) P(y) /(P(x1)P(x2)…..P(xn)
                prob = prob * (( self.TermClassFrequency[term].get(key,0)+1) /( len(self.ClassTermCount[key]) + len( self.unique_terms)))
            result[key] = prob
        return sorted(result.items(),key=lambda k:k[1],reverse=True)[0:5]

    def Test(self,start,stop):
        result = 0
        for i in range(start,stop):
            for j in self.CalculateTermProbablity(self.RecipeData[i]['cuisine']):
                if self.RecipeData[i]['cuisine'] in j:
                    result = result + 1
        return result/(stop-start)
