
import numpy as np

class KNN :
    def __init__ (self , k) :
        self.k = k

    #training
    def fit (self , x , y) :
        self.X_train = x
        self.Y_train = y

    #distance
    def euclidean_distances (self , x_1 , x_2) :
        return np.sqrt (np.sum ((x_1 - x_2)**2))

    #prediction
    def predict (self , X) :
        Y = []
        for x in X :
            distances = []
            for x_train in self.X_train :
                d = self.euclidean_distances (x , x_train)
                distances.append (d)
            
            nearest_neighbors = np.argsort (distances) [0 : self.k]
            result = np.bincount (self.Y_train [nearest_neighbors])
            y = np.argmax (result)

            Y.append (y)
        
        return Y
    
    #test
    def evaluate (self , X , Y) :
        Y_pred = self.predict (X)
        accuracy = np.sum (Y_pred == Y) / len (Y)
        return accuracy