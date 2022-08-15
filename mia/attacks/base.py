import pandas as pd

class BaseAttacker:
    def __init__(self, synthesizer, data, query_size=2000):
        self.synthesizer = synthesizer
        self.data = data
        self.query_size = query_size 
    
    def query(self, train_data, test_data):
        neg_query_index = pd.read_csv("result/{}/dist_info/dist_info.csv".format(self.data),index_col=0).iloc[:,0]
        neg_query = test_data[neg_query_index.argsort()][-self.query_size:] 
        pos_query_index = list(range(self.query_size))
        pos_query = train_data[pos_query_index]

        return pos_query, neg_query
        
    def optimize(self):
        pass

    def attack(self):
        pass
 