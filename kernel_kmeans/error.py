import math
# entropy E = p(+)log(p(+)) + p(x)log(p(x)) + p(*)log(p(*)) ..., the less the entropyy, more is the cluster quality.
# input : get a dictionary -> key = cluster number, value = list of documents in that cluster (predicted), another list of list -> containing each clusters (actual labels)

target = [
    ['D1.txt', 'D2.txt', 'D3.txt'], # medical
    ['D4.txt', 'D5.txt'], # tech
    ['D6.txt', 'D7.txt', 'D8.txt'], # politics
    ['D9.txt', 'D10.txt'] # sports
]

def get_entropy(centroid_doc_map):
    entropy = 0
    for key in centroid_doc_map.keys():
        cluster_points = centroid_doc_map[key]
        cluster_entropy = 0
        total_points = len(cluster_points)
        for target_cluster in target:
            p = 0
            for point in cluster_points:
                if point in target_cluster:
                    p += 1
            prob = p / total_points
            if prob != 0:
                cluster_entropy += (prob * math.log(prob))*(-1)
        entropy += cluster_entropy
    entropy = entropy/len(centroid_doc_map.keys())
    return entropy