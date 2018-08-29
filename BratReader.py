import sys
import csv

import sklearn
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from bratreader.repomodel import RepoModel
# from nltk.parse.stanford import StanfordDependencyParser
import nltk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Annotation:
    def __init__(self):
        self.representation = ""
        self.start = 0
        self.end = 0
        self.label = ""

class OutputCandidate:
    def __init__(self):
        self.LinkType = ""
        self.DrugRepr = ""
        self.OtherRepr = ""
        self.textBetween = ""
        self.text_before = ""
        self.text_after = ""
        self.drug_start = 0
        self.drug_end = 0
        self.other_start = 0
        self.other_end = 0
        self.isPositive = False


source = sys.argv[1]
r = RepoModel(source)
# print(r.documents )
myfile = open("DrugInteractionCSV15.arff", 'w')
# wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
# wr.writerow(["LinkType","DrugRepr","OtherRepr","BetweenLength","NumTokensBetween","IsPositive"])
myfile.write('''@RELATION wordcounts

@ATTRIBUTE LinkType string
@ATTRIBUTE DrugRepr string
@ATTRIBUTE OtherRepr string
@ATTRIBUTE BetweenLength numeric
@ATTRIBUTE NumTokensBetween numeric
@ATTRIBUTE betweenText string
@ATTRIBUTE isPositive {True,False}

@DATA
''')
negative_count = 0
samples = []
for doca in r.documents:
    AllAnnotation = []
    DrugAnnotations = []
    PositiveSamples = []
    NegativeSamples = []
    print(doca)
    file = open(source+"/"+doca+".txt")
    textA = file.read()
    doc = r.documents[doca]
     #rtext = doc.text.replace('\n','|n').replace('\r','|r').replace('\t','|t').replace('\f','|f').replace('\\','||').replace('\'','||').replace('\"','||')
    for ann in doc.annotations:
        for label in ann.labels:
            annotation = Annotation()
            annotation.representation = ann.repr
            annotation.start = ann.realspan[0]
            annotation.end = ann.realspan[1]
            annotation.label = label
            if label == "Drug":
                DrugAnnotations.append(annotation)
            else:
                AllAnnotation.append(annotation)

        for link in ann.links:
            for a in ann.links[link]:
                LinkType = link
                DrugRepr = str(ann.repr)
                OtherRepr = str(a.repr)
                drug_start = ann.realspan[0]
                drug_end = ann.realspan[1]
                for span in a.spans:
                    other_start = span[0]
                    other_end = span[1]
                #ann is a drug a is a other
                    if span[0]<=ann.realspan[0]:
                        textBetween = textA[span[1]:ann.realspan[0]]
                        if a.spans[0][0]-200>0:
                            text_before = textA[(span[0]-200):span[0]]
                        else:
                            text_before = textA[0:span[0]]
                        if ann.realspan[1]+200<len(textA):
                            text_after = textA[ann.realspan[1]:(ann.realspan[1]+200)]
                        else:
                            text_after = textA[ann.realspan[1]:len(textA)]
                    else:
                        textBetween = textA[ann.realspan[1]:span[0]]
                        if ann.realspan[0]-200>0:
                            text_before = textA[(ann.realspan[0]-200):ann.realspan[0]]
                        else:
                            text_before = textA[0:ann.realspan[0]]
                        if a.spans[0][1] + 200 < len(textA):
                            text_after = textA[span[1]:(span[1] + 200)]
                        else:
                            text_after = textA[span[1]:len(textA)]
                    oc = OutputCandidate()
                    oc.LinkType = LinkType
                    oc.DrugRepr = DrugRepr
                    oc.OtherRepr = OtherRepr
                    oc.textBetween = textBetween
                    oc.text_before = text_before
                    oc.text_after = text_after
                    oc.drug_start = drug_start
                    oc.drug_end = drug_end
                    oc.other_start = other_start
                    oc.other_end = other_end
                    oc.isPositive = True
                    # inputIt = True
                    # for posSample in PositiveSamples:
                    #     if posSample.drug_start ==oc.drug_start and posSample.drug_end==oc.drug_end and posSample.other_start == oc.other_start and posSample.other_end==oc.other_end:
                    #         inputIt = False
                    # if inputIt:
                    PositiveSamples.append(oc)
                    sample = oc
                    distance = len(sample.textBetween)
                    tokens = nltk.word_tokenize(sample.textBetween)
                    number_tokens = len(tokens)

                    myfile.write('"%s","%s","%s",%d,%d,"%s",%s\n' % (
                    sample.LinkType, sample.DrugRepr.replace('"',''), sample.OtherRepr.replace('"',''), distance,
                    number_tokens, textBetween.replace('\n',' ').replace('"',' ').replace('\'',' '),
                    str(sample.isPositive)))
                    samples.append(
                        (sample.LinkType, sample.DrugRepr.replace('"', ""), sample.OtherRepr.replace('"', ''), distance,
                         number_tokens, textBetween.replace('\n', ' ').replace('"', ' ').replace('\'', ' '),
                         str(sample.isPositive)))

    for drug in DrugAnnotations:
        for other in AllAnnotation:
            # if negative_count>1600000:
            #     break
            # negative_count = negative_count + 1
            if abs(drug.end - other.start) > 200:
                continue
            Discard = False
            for link in PositiveSamples:
                if link.drug_start == drug.start and link.drug_end == drug.end and link.other_start == other.start and link.other_end ==other.end:
                    Discard = True
                    break
            if Discard == False:
                oc = OutputCandidate()
                oc.LinkType = other.label+"-Drug"
                oc.DrugRepr = drug.representation
                oc.drug_start = drug.start
                oc.drug_end = drug.end
                oc.OtherRepr = other.representation
                oc.other_start = other.start
                oc.other_end = other_end
                oc.isPositive = False
                if drug.end < other.start:
                    textBetween = textA[drug.end:other.start]
                    if drug.start - 200 >0:
                        text_before = textA[(drug.start-200):drug.start]
                    else:
                        text_before = textA[0:drug.start]
                    if other.end+200<len(textA):
                        text_after = textA[other.end:(other.end+200)]
                    else:
                        text_after = textA[other.end:len(textA)]
                else:
                    textBetween = textA[other.end:drug.start]
                    if other.start - 200 > 0:
                        text_before = textA[(other.start-200):other.start]
                    else:
                        text_before = textA[0:other.start]
                    if drug.end+200<len(textA):
                        text_after = textA[drug.end:(drug.end+200)]
                    else:
                        text_after = textA[drug.end:len(textA)]
                oc.text_before = text_before
                oc.text_after = text_after
                oc.textBetween = textBetween
                NegativeSamples.append(oc)
                sample = oc
                distance = len(sample.textBetween)
                tokens = nltk.word_tokenize(sample.textBetween)
                number_tokens = len(tokens)
                myfile.write('"%s","%s","%s",%d,%d,"%s",%s\n' % (sample.LinkType, sample.DrugRepr.replace('"',""), sample.OtherRepr.replace('"',''), distance,
                                                         number_tokens, textBetween.replace('\n',' ').replace('"',' ').replace('\'',' '),
                                                         str(sample.isPositive)))
                samples.append((sample.LinkType, sample.DrugRepr.replace('"',""), sample.OtherRepr.replace('"',''), distance,
                                                         number_tokens, textBetween.replace('\n',' ').replace('"',' ').replace('\'',' '),
                                                         str(sample.isPositive)))

myfile.close()

df = pd.DataFrame(samples, columns=("LinkType","DrugRepr","OtherRepr","Distance","NumberOfTokens","TextBetween","IsPositive"))
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .8
train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
features = df.columns[1:6]
print(features)
y = pd.factorize(train['IsPositive'])[0]
print(y)
print(len(y))
feat_vector = train[features]
print("FeatureVector shape")
print(feat_vector.shape)
print(feat_vector.head())
count_vect1 = CountVectorizer(max_features = 800)
count_vect2 = CountVectorizer(max_features = 800)
count_vect3 = CountVectorizer(max_features = 800)
X_train_counts_1 = count_vect1.fit_transform(train["TextBetween"])
X_train_counts_drugs = count_vect2.fit_transform(train['DrugRepr'])
X_train_counts_other = count_vect3.fit_transform(train['OtherRepr'])
print(X_train_counts_1.shape)
print(X_train_counts_drugs.shape)
print(X_train_counts_other.shape)
X_train_counts = pd.merge(pd.DataFrame(X_train_counts_1.toarray()),pd.DataFrame(X_train_counts_drugs.toarray()),left_index=True,right_index=True)
X_train_counts= pd.merge(X_train_counts,pd.DataFrame(X_train_counts_other.toarray()),left_index=True,right_index=True)
#X_train_counts = pd.merge(X_train_counts,train['Distance'].to_frame(),left_index=True,right_index=True)
print("X_train_counts")
print(X_train_counts.shape)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print("X_train_ft")
print(X_train_tf.shape)
print("Training")
clf = RandomForestClassifier(n_jobs=3, random_state=0,n_estimators =150)
clf.fit(X_train_tf, y)
print("Trained")
feat_vector2 = test[features]
X_new_counts_1 = count_vect1.transform(test["TextBetween"])
X_new_counts_drugs = count_vect2.transform(test['DrugRepr'])
X_new_counts_other = count_vect3.transform(test['OtherRepr'])
X_new_counts = pd.merge(pd.DataFrame(X_new_counts_1.toarray()),pd.DataFrame(X_new_counts_drugs.toarray()),left_index=True,right_index=True)
X_new_counts= pd.merge(X_new_counts,pd.DataFrame(X_new_counts_other.toarray()),left_index=True,right_index=True)
#X_new_counts = pd.merge(X_new_counts,test['Distance'].to_frame(),left_index=True,right_index=True)
X_new_tfidf = tf_transformer.transform(X_new_counts)
y_pred = clf.predict(X_new_tfidf)
y_train = pd.factorize(test['IsPositive'])[0]
print(sklearn.metrics.classification_report(y_pred,y_train))
# View target




