from collections import defaultdict
import random
import sys

class KMeans(object):
    def __init__(self, persons=None, similarity_calculator=None):
        self._similarity_calculator = similarity_calculator
        self._persons = persons
        
    def _computeInitialCentriods(self, data_points, k):
        centroids = []
        #choose random k points as seed
        if k > len(data_points):
            print('incorrect value of k specified')
            k = len(data_points)
        random.shuffle(data_points)
        for i in range(k):
            centroids.append(data_points[i])
        
        return centroids
    
    def _computeSimilarity(self, datapoint1, datapoint2):
        if self._similarity_calculator and self._persons:
            return self._similarity_calculator(self._persons, datapoint1, datapoint2)
        else:
            return 0
        
    def _areClustersSame(self, clusters1, clusters2):
        for cluster1 in clusters1:
            cluster1_present_inclusters2 = False
            for cluster2 in clusters2:
                if set(clusters1[cluster1]) == set(clusters2[cluster2]):
                    cluster1_present_inclusters2 = True
                    break
            if not cluster1_present_inclusters2:
                return False
        return True
    
    def _assignToClusters(self, data_points, centroids):
        clusters = defaultdict(list)
        
        #for all data points(friends in our case), findout which centroid they are most similar to
        #assign them to most similiar centroid
        for data_point in data_points:
            maximum_similarity = 0
            maximum_similarity_centroid = None
            for centroid in centroids:
                similarity = self._computeSimilarity(data_point, centroid)
                if similarity > maximum_similarity:
                    maximum_similarity = similarity
                    maximum_similarity_centroid = centroid
            if maximum_similarity_centroid:
                clusters[maximum_similarity_centroid].append(data_point)
        
        #return the computed clusters
        return clusters
        
    def _recomputeCentroids(self, clusters):
        centroids = []
        
        #for every cluster compute its mean centroid. mean centroid is the point which 
        #maximizes sum of similarities between datapoints in that cluster
        for centroid in clusters:
            maximum_similarity_score = 0
            maximum_similarity_point = None
            
            for i in range(len(clusters[centroid])):
                similarity_score = 0
                for j in range(len(clusters[centroid])):
                    similarity_score += self._computeSimilarity(clusters[centroid][i], clusters[centroid][j])
                if similarity_score > maximum_similarity_score:
                    maximum_similarity_score = similarity_score
                    maximum_similarity_point = self._clusters[centroid][i]
            
            centroids.append(maximum_similarity_point)
            
        return centroids
                
    def _trim_clusters(self, clusters, similarity_diff_threshold):
        for centroid in clusters:
            #compute maximum and minimum similarity from centroid for each circle/cluster
            minimum_similarity = sys.maxint
            maximum_similarity = -1
            for friend in clusters[centroid]:
                similarity = self._computeSimilarity(centroid, friend)
                if similarity < minimum_similarity:
                    minimum_similarity = similarity
                if similarity > maximum_similarity:
                    maximum_similarity = similarity
            #print('centroid: ' + str(centroid) + 'max sim: ' + str(maximum_similarity) + ' min sim: ' + str(minimum_similarity))
                    
            #if similarity difference is more than a threshold, drop friends with 
            #small similarity from circle/cluster
            similarity_difference_in_cluster = maximum_similarity - minimum_similarity
            if similarity_difference_in_cluster > similarity_diff_threshold:
                friends_to_remove = []
                for friend in clusters[centroid]:
                    if self._computeSimilarity(centroid, friend) < maximum_similarity - similarity_diff_threshold:
                        friends_to_remove.append(friend)
                for friend_to_remove in friends_to_remove:
                    clusters[centroid].remove(friend_to_remove)
     
        return clusters
    
    def setSimilarityCalculator(self, similarity_calculator):
        self._similarity_calculator = similarity_calculator
        
    def computeClusters(self, data_points, k, similarity_diff_threshold, max_iterations=20):
        data_points_list = list(data_points)
        centroids = self._computeInitialCentriods(data_points_list, k)
        prev_clusters = defaultdict(list)
        clusters = defaultdict(list)
        
        for _ in range(max_iterations):
            clusters = self._assignToClusters(data_points_list, centroids)
            if self._areClustersSame(prev_clusters, clusters):
                break
            prev_clusters = clusters
            centroids = self._recomputeCentroids(clusters)
        
        return self._trim_clusters(clusters, similarity_diff_threshold)
        
    
    