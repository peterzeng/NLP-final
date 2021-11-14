import pandas as pd

# Returns a dataframe with the sentences and author
def load_data(filename: str, author: str):
    
    sequences = []
    author = [author]*1000
    data = open(filename, 'r', encoding="utf8")
    
    words = []
    for line in data:
        words.extend(line.split(" "))
    
    # print(len(words))

    # print(len(words))
    #     # print(data)
    # # data = list(data.split(" "))
    for i in range(1000):
        sequence = []
        for j in range(25):
            sequence.append(words[i*25 + j])
        sequences.append(sequence)
    
    processed = pd.DataFrame(list(zip(sequences,author)), columns=["sequences", "author"])
    processed.to_csv('data.csv', header=True, index=False)
    
    # print(len(data))

    # for line in data:
    #     words = line.split(",")

if __name__ == "__main__":
    author = "CharlesDickens"
    book = "./Books/Oliver Twist/book.txt"
    load_data(book, author)