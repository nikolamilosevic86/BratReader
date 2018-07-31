import sys
from bratreader.repomodel import RepoModel
from nltk.parse.stanford import StanfordDependencyParser
import nltk


source = sys.argv[1]
r = RepoModel(source)
# print(r.documents )

doc = r.documents["112329"]

# print(doc.sentences)    			# a list of sentences in document
# print(doc.annotations)  			# the annotation objects in a document
for ann in doc.annotations:
    print("Annotation:")
    print(ann.id)
    for link in ann.links:
        print("Link:"+link)
        print(ann.links[link][0].repr)
        print(ann.links[link][0].spans[0])
    for label in ann.labels:
        print("Label:"+str(label))
    print("Representation:"+str(ann.repr))
    for rs in ann.realspan:
        print(rs)
    print("==============")
#print(str(a.id)+" "+str(a.links.keys())+"  "+str(a.labels)+"  "+str(a.words))
path_to_jar = 'stanford-parser-full-2018-02-27/stanford-parser.jar'
path_to_models_jar = 'stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
sentences = nltk.sent_tokenize(doc.text)

possition_start = 0
possition_end = 0
l = 0
b = 0
position = 0
for sent in sentences:
    tokens = nltk.word_tokenize(sent)
    contains = True
    a = 0
    text_before = doc.text[0:doc.text.index(sent)]
    possition_start = doc.text.index(sent) + 2*text_before.count('\n')+2*text_before.count('\t')
    possition_end = possition_start + len(sent)+ 2*sent.count('\n')+2*sent.count('\t')
    position = position + len(sent)
    # possition_end = position
    drug_annotation = None
    try:
        current_span = doc.annotations[l].spans[0]
        while current_span[0]<=possition_start:
            l = l + 1
            current_span = doc.annotations[l].spans
        if current_span[1]<possition_end:
            m = l
            while current_span[0]>possition_start and current_span[1]<possition_end:
                for label in doc.annotations[m].labels:
                    if label=="Drug":
                        drug_annotation = doc.annotations[m]
                m = m+1
                current_span = doc.annotations[m].spans[0]
            current_span = doc.annotations[l].spans[0]
            m = l
            while current_span[0] > possition_start and current_span[1] < possition_end:
                for label in doc.annotations[l].labels:
                    if label=="Strength":
                        link_name = "Strength-Drug"
                        drug_name = drug_annotation.repr
                        current_name = doc.annotations[m].repr

                        print(link_name)
                        print(drug_name)
                        print(current_name)
                for label in doc.annotations[l].labels:
                    if label == "Route":
                        link_name = "Route-Drug"
                        drug_name = drug_annotation.repr
                        current_name = doc.annotations[l].repr

                        print(link_name)
                        print(drug_name)
                        print(current_name)
                for label in doc.annotations[l].labels:
                    if label == "Dosage":
                        link_name = "Dosage-Drug"
                        drug_name = drug_annotation.repr
                        current_name = doc.annotations[l].repr

                        print(link_name)
                        print(drug_name)
                        print(current_name)
                for label in doc.annotations[l].labels:
                    if label == "Frequency":
                        link_name = "Frequency-Drug"
                        drug_name = drug_annotation.repr
                        current_name = doc.annotations[l].repr

                        print(link_name)
                        print(drug_name)
                        print(current_name)
                for label in doc.annotations[l].labels:
                    if label == "Reason":
                        link_name = "Reason-Drug"
                        drug_name = drug_annotation.repr
                        current_name = doc.annotations[l].repr

                        print(link_name)
                        print(drug_name)
                        print(current_name)

                for label in doc.annotations[l].labels:
                    if label == "Form":
                        link_name = "Form-Drug"
                        drug_name = drug_annotation.repr
                        current_name = doc.annotations[l].repr

                        print(link_name)
                        print(drug_name)
                        print(current_name)
                l = l + 1
                current_span = doc.annotations[l].spans[0]

            # result = dependency_parser.raw_parse(sent)
            # dep = result.__next__()
             # print(list(dep.triples()))
        possition_start = possition_end+1
    except:
        pass

#print(doc.text)