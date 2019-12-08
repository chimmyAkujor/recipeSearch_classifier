from flask import Flask, request,render_template
import TextSearch as ts
import NaiveBayes as nb
import re

app = Flask(__name__)

ClassifyObj = nb.NaiveBayes()
SearchObj = ts.TextSearch()

def highlight_terms(qterms, doc):
    qterms = SearchObj.tokenize(qterms)
    docTerms = SearchObj.tokenize(doc)
    for term in qterms:
        if term in docTerms:
            temp = re.compile(re.escape(term), re.IGNORECASE)
            doc = temp.sub("<mark>{term}</mark>".format(term=term), doc)
    return doc

@app.route('/', methods=['GET'])
def Index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/search', methods=['GET','POST'])
def SearchPage():
    if request.method == 'GET':
        return render_template('search.html')
    elif request.method == 'POST':
        query = request.form['query']
        result = SearchQuery(query)
        try:
            length = len(result['Recipe'])
            highlighted = []
            for i in range(length):
                highlighted.append(highlight_terms(query,result['Instructions'][i]))
            if length != 0:
                return render_template('search_results.html', recipes=result, length = length, highlighted=highlighted)
        except TypeError:
            return render_template('search_results.html', recipes=result, length = -1)

@app.route('/classifier',methods=['GET','POST'])
def ClassifierPage():
    if request.method == 'GET':
        return render_template('classifier.html')
    elif request.method == 'POST':
        query = request.form['query']
        if len(query) != 0:
            results = ClassifyQuery(query)
            print(results)
            return render_template('classifier_results.html',results = results, query = query)
        else:
            return render_template('classifier_results.html', results = None, query = query)




def InitialiseSearchObject():
    # SearchObj.BuildInvertedIndex()
    SearchObj.loadPickle()
def InitializeClassifierObject():
    ClassifyObj.LoadPickle()

def SearchQuery(query):
    return SearchObj.Search(query)

def ClassifyQuery(query):
    q = [query]
    return ClassifyObj.CalculateTermProbability(q)

InitialiseSearchObject()
InitializeClassifierObject()

if __name__ == '__main__':
   app.run()
