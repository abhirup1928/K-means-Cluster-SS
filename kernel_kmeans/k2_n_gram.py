import numpy as np
# n-gram i.e, n = 2 to n = 5

def k2(path_to_doc1, path_to_doc2):
    freq1, freq2 = {}, {} # frequency vectors for doc1 and doc2
    with open(path_to_doc1, 'r') as f:
        doc1 = f.read() # read the entire file including newline and whitespace    
    with open(path_to_doc2, 'r') as f:
        doc2 = f.read()
    # the n-gram logic
    for n in range(2, 6):
        # each word is of n characters
        for i in range(len(doc1) - n + 1):
            word = doc1[i:i+n]
            if word not in freq1:
                freq1[word] = 0
            if word not in freq2:
                freq2[word] = 0
            freq1[word] += 1
        for i in range(len(doc2) - n + 1):
            word = doc2[i:i+n]
            if word not in freq1:
                freq1[word] = 0
            if word not in freq2:
                freq2[word] = 0
            freq2[word] += 1
    
    # convert the frequency vectors to numpy arrays
    freq1 = np.array(list(freq1.values()))
    freq2 = np.array(list(freq2.values()))
    # return the cosine similarity
    return np.dot(freq1, freq2) / (np.linalg.norm(freq1) * np.linalg.norm(freq2))

if __name__ == '__main__':
    docs = ['./D1.txt', './D2.txt', './D3.txt', './D4.txt']
    for i in range(len(docs)):
        for j in range(i, len(docs)):
            if docs[i] == docs[j]:
                continue
            print("Cosine similarity between", docs[i], "and", docs[j], "is :", k2(docs[i], docs[j]))