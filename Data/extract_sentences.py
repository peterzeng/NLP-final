import os
import xml.etree.ElementTree as ET
import pandas as pd

sequences = []
authors = {}

dir = './base'
num_authors = 0

for file in os.listdir(dir):
    if num_authors < 5:
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

        for i in range(int(len(words)/25)):
            sequence = []
            for j in range(25):
                sequence.append(words[i*25 + j])
            sequences.append([sequence, author])
        num_authors += 1
        
# print(sequences)
processed = pd.DataFrame((sequences), columns=["text", "Author"])
processed.to_csv("full data.csv", header=True, index=False)