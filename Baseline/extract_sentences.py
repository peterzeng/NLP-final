import os
import xml.etree.ElementTree as ET
import pandas as pd
import csv
sequences = []
authors = {}

dir = '../Data/base'
num_books = 10 # 5, 25, 50, 99
counter = 0

for file in os.listdir(dir):
    if counter < num_books:
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

        book = front + body
        words = book.split(" ")

        for i in range(int(len(words)/50)):
            sequence = []
            for j in range(50):
                sequence.append(words[i*50 + j])
            sequences.append((sequence, author))
        
        counter += 1
    
# print(sequences)
processed = pd.DataFrame((sequences), columns=["text", "Author"])
processed.to_csv("data.csv", header=True, index=False)