import numpy as np
from nltk.corpus import stopwords

def k1(path_to_doc1, path_to_doc2):
    freq1, freq2 = {}, {} # frequency vectors for doc1 and doc2
    stop_words = set(stopwords.words('english'))
    # print(stop_words)
    with open(path_to_doc1, 'r') as f:
        words = f.read().split() # split by whitespace and newline
        for word in words:
            # ignore common words like 'the', 'a', 'an', etc.
            if word in stop_words:
                continue
            # if the word is not in the frequency vector, add it
            if word not in freq1:
                freq1[word] = 0
            if word not in freq2:
                freq2[word] = 0
            # increment the frequency of the word
            freq1[word] += 1
    with open(path_to_doc2, 'r') as f:
        words = f.read().split() # split by whitespace and newline
        for word in words:
            # ignore common words like 'the', 'a', 'an', etc.
            if word in stop_words:
                continue
            # if the word is not in the frequency vector, add it
            if word not in freq1:
                freq1[word] = 0
            if word not in freq2:
                freq2[word] = 0
            # increment the frequency of the word
            freq2[word] += 1
    # convert the frequency vectors to numpy arrays
    freq1 = np.array(list(freq1.values()))
    freq2 = np.array(list(freq2.values()))
    # return the cosine similarity
    return np.dot(freq1, freq2) / (np.linalg.norm(freq1) * np.linalg.norm(freq2))
    
if __name__ == '__main__':
    docs = ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt']
    for i in range(len(docs)):
        for j in range(i, len(docs)):
            # if docs[i] == docs[j]:
            #     continue
            print("Cosine similarity between", docs[i], "and", docs[j], "is :", k1(docs[i], docs[j]))