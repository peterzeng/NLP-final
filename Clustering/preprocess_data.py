import os
import xml.etree.ElementTree as ET
import pandas as pd
from nltk import word_tokenize

# Preprocess and tokenize
# leave punctuation because it may contribute to author styles

sequences = []

dir = '../Data/base'
num_books = 0

for file in os.listdir(dir):
    if num_books < 99:
        filename = os.path.join(dir, file)
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # root[0][0] is header
        # we want to extract the author
        header = root[0][0].text
        author = [line for line in header.split('\n') if "Author" in line]
        # print(filename)
        if len(author) == 0:
            continue
        else:
            author = author[0].split(' ', 1)[1].replace(" ", "")
            if author == "various" or author == "Various" or author == "unknown" or author == "Unknown":
                continue
        
        # extract book
        if root.find("front").text:
            front = root.find('front').text 
        else:
            front = ""
        body = root.find('body').text

        # Remove index if it exists
        if body.rfind('INDEX'):
            body = body[0:body.rfind('INDEX')]
        elif body.rfind('Index'):
            body = body[0:body.rfind('INDEX')]

        if body.rfind('GLOSSARY'):
            body = body[0:body.rfind('GLOSSARY')]
        elif body.rfind('Glossary'):
            body = body[0:body.rfind('Glossary')]
        
        book = front + body
        # book.replace("\n", " ")
        words = book.split(" ")

        for i in range(int(len(words)/100)):
            sequence = []
            for j in range(100):
                sequence.append(words[i*100 + j])
        
            sentence = ' '.join(sequence).replace("\n", " ")
            sequences.append([sentence, author])

        num_books += 1
    
# print(len(sequences))
corpus = pd.DataFrame((sequences), columns=["text", "author"])

auth_sort = sorted(corpus['author'].unique())
dictOfAuthors = { i : auth_sort[i] for i in range(0, len(auth_sort) ) }
swap_dict = {value:key for key, value in dictOfAuthors.items()}
corpus['author_num'] = corpus['author'].map(swap_dict)
corpus.to_csv("full-corpus.csv", header = True, index=False)