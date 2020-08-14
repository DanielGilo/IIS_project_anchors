from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop, heapify


Infinite = float('inf')

#node = {'anchor':{..}, 'remaining_features': ['feature1', 'feature2',..]}

def is_relevant(example, anchor):
    relevant = True
    for k in anchor.keys():
        if example[k] != anchor[k]:
            relevant = False
            break
    return relevant


class Anchor:

    def __init__(self, anchor, remaining_features):
        self.anchor = anchor
        self.remaining_features = remaining_features



class AStar:
    __metaclass__ = ABCMeta

    def __init__(self, example, pred_example, sample, pred_sample, classifier, threshold, features, distance_method):
        self.example = example
        self.sample = sample
        self.classifier = classifier
        self.threshold = threshold
        self.features = features
        self.pred_sample = pred_sample
        self.pred_example = pred_example
        self.distance_method = distance_method

    class SearchNode:
        __slots__ = ('data', 'gscore', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    @abstractmethod
    def heuristic_cost_estimate(self, current):
        """Computes the estimated (rough) distance between a node and the goal, this method must be implemented in a subclass. The second parameter is always the goal."""
        relevants = 0
        accurate_relevants = 0
        for i in range(len(self.sample)):
            if is_relevant(self.sample.iloc[i], current.anchor):
                relevants += 1
                if self.pred_sample.iloc[i] == self.pred_example:
                    accurate_relevants += 1
        accuracy = accurate_relevants/relevants
        if self.threshold-accuracy <= 0:
            x = 5
        return max(0, self.threshold - accuracy)



    @abstractmethod
    def distance_between(self, n1, n2):
        """Gives the real distance between two adjacent nodes n1 and n2 (i.e n2 belongs to the list of n1's neighbors).
           n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
           This method must be implemented in a subclass."""
        if self.distance_method == 'direct':
            n1_relevants = 0
            n2_relevants = 0
            for i in range(len(self.sample)):
                if is_relevant(self.sample.iloc[i], n1.anchor):
                    n1_relevants += 1
                if is_relevant(self.sample.iloc[i], n2.anchor):
                    n2_relevants += 1
            return (n1_relevants - n2_relevants)/len(self.sample)
        else:
            return 0.5


    @abstractmethod
    def neighbors(self, node):
        neighbors = []
        for f in node.remaining_features:
            current_anchor = node.anchor.copy()
            current_remaining_features = node.remaining_features.copy()
            current_anchor[f] = self.example[f]
            current_remaining_features.remove(f)
            neighbors.append(Anchor(current_anchor, current_remaining_features))
        return neighbors




    def is_goal_reached(self, current):
        return self.heuristic_cost_estimate(current) == 0


    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start, reversePath=False):
        if self.is_goal_reached(start):
            return [start]
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start))
        openSet = []
        heappush(openSet, startNode)
        i = 0
        while openSet:
            #print(i)
            i+=1
            current = heappop(openSet)
            if self.is_goal_reached(current.data):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    heapify(openSet)
        return None
