
from bs4 import BeautifulSoup
text = open("TRAIN_FILE.TXT", "r").read()

list=[]
list1=[]
with open("TRAIN_FILE.TXT", "r") as f:   
    count = 3
    for line in f:
        soup = BeautifulSoup(line, "lxml")
        count+=1
        if count % 4 == 0: 
          for e1_tag in soup.find_all('e1'):
               list.append((e1_tag.text+e1_tag.next_sibling))
          for e2_tag in soup.find_all('e2'):
               list1.append((e2_tag.text))     

listx=[' '.join(x) for x in zip(list,list1)]



listy=[]
with open("TRAIN_FILE.TXT", "r") as f:
    count = 2
    for line in f:
        count+=1
        if count % 4 == 0: 
            listy.append(line)
         
listm=[]
for line in listy:
    sep = '('
    rest= line.split(sep, 1)[0]
    rest=rest.strip('\n')
    listm.append(rest)
      



testx1=[]
testx2=[]
with open("TEST_FILE_CLEAN.TXT", "r") as f:
  
    #count = 3
    for line in f:
        e1='e1'
        e2='e2'
        soup = BeautifulSoup(line, "lxml")
        for e1_tag in soup.find_all(e1):
               testx1.append((e1_tag.text+e1_tag.next_sibling))
        for e2_tag in soup.find_all(e2):
               testx2.append((e2_tag.text)) 
  
         
testx=[''.join(x) for x in zip(testx1,testx2)]

listn=[]
with open("TEST_FILE_KEY.TXT", "r") as f:
    for line in f:
        line=line.split()[1]
        listn.append(line)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y=le.fit_transform(listm)

TY=le.transform(listn)
      

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
     


    

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    
text_clf = Pipeline([('vect', StemmedCountVectorizer(tokenizer=LemmaTokenizer(),
                                                     ngram_range=(1,10))),
                      ('clf', MultinomialNB(fit_prior=False)),
 ])
    
text_clf.fit(listx,Y )  


predicted = text_clf.predict(testx)
n=8000
thefile = open('output.txt', 'w')
for item in le.inverse_transform(predicted):
    n=n+1
    thefile.write(str(n)+"\t"+item+"\n")      
thefile.close()
print("Accuracy-")
print(np.mean(predicted == TY))
