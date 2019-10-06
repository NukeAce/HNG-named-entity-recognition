import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


#function to preprocess sentence
def preprocess(sample):
    sample = nltk.word_tokenize(sample)
    sample = nltk.pos_tag(sample)
    return sample


#functions to list results
def list_people():
    print("People: ")
    for i in list_of_people:
        print(i)

def list_organizations():
    print("\nOrganizations: ")
    for i in list_of_organizations:
        print(i)

def list_locations():
    print("\nLocations: ")
    for i in list_of_locations:
        print(i)


sentence = input("Please enter your sentence: ")
sentence1 = preprocess(sentence)
ne_tree = nltk.ne_chunk(sentence1)
list_of_people = [' '.join(leaf[0] for leaf in tree.leaves()) 
                      for tree in ne_tree.subtrees() 
                      if tree.label()=='PERSON']

list_of_organizations = [' '.join(leaf[0] for leaf in tree.leaves()) 
                      for tree in ne_tree.subtrees() 
                      if tree.label()=='ORGANIZATION']

list_of_locations = [' '.join(leaf[0] for leaf in tree.leaves()) 
                      for tree in ne_tree.subtrees() 
                      if tree.label()=='GPE']

list_people()
list_organizations()
list_locations()