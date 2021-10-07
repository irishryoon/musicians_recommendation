"""
Perform parts of the hyperparameter tuning process
(1) Sample random walks from Graph for various parameter values of `p`, `q`, and `length`.
(2) train a word2vec model using the random walks generated.
The results of this script is then inspected to select the best model 
"""
import networkx as nx
import pickle
import random
import math 
from stellargraph import StellarGraph
from musicians import *
from random import sample
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

# create custom callbacks for Word2Vec training
class MonitorCallback(CallbackAny2Vec):
    # print training loss after each epoch
    def __init__(self):
        self.epoch = 0
        self.train_loss = []

    def on_epoch_end(self, model):
        # Compute training loss difference from last training
        # Necessary because gensim only returns a cumulative loss, and not the training loss after each epoch
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.train_loss.append(loss)
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
            self.train_loss.append(loss-self.loss_previous_step)
        self.epoch += 1
        self.loss_previous_step = loss
            
class EpochSaver(CallbackAny2Vec):
    # Save model at the end of epoch    
    def __init__(self, path):
        self.path = path
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path, self.epoch)
        model.save(output_path)
        self.epoch += 1

def load_data():
    ### load graph G
    # note the small graph only consists of artists with degree 10 or higher
    G = nx.read_gml("graph_80000/graph_min_degree_10.gml")
    # convert edge's 'album' attribute from list to set
    for n1,n2,edge in G.edges(data=True):
        edge['albums'] = set(edge['albums'])
    
    # normalize weights
    weights = []
    for n1, n2, e in G.edges(data = True):
        weights.append(e['weight'])
    max_weight = max(weights)
    for n1, n2, e in G.edges(data = True):
        e['weight'] = e['weight']/max_weight
        
    # convert networkx graph to stellargraph 
    G_s = StellarGraph.from_networkx(G)

    ### load artists' degree group info
    with open("graph_80000/artist_by_degree.pkl", "rb") as f:
        artists_by_degree_big = pickle.load(f)

    # only keep artists with degree 10 or greater
    artists_by_degree = {k:v for k,v in artists_by_degree_big.items() if k > 10}

    # specify the number of walks for artists in each degree group
    n_walks_by_degree = {}
    for deg in artists_by_degree.keys():
        n_walks_by_degree[deg] = int(deg**1.2)

    return G_s, artists_by_degree, n_walks_by_degree

def main():
    directory = "graph_80000/hyperparameter_tuning"
    # parameters of word2vec model
    vector_size = 300
    window = 5
    min_count = 100
    sg = 0
    epochs = 20

    # load data
    print("loading data... ")
    G_s, artists_by_degree, n_walks_by_degree = load_data()
    
    # run 2 different sessions
    for i in range(2):
        p = random.uniform(0.01, 10)
        q = random.uniform(0.01, 10)
        l = 15

        # generate random walk
        walks = sample_random_walks(G_s, artists_by_degree, n_walks_by_degree, p = p, q = q, length = l)

        ### train word2vec
        # callbacks
        monitor = MonitorCallback()
        save_model = EpochSaver(directory + '/word2vec_train/word2vec_iter_'+str(i))

        # train
        model = Word2Vec(walks, vector_size = vector_size, window= window, min_count= min_count, sg= sg, 
                        workers=2, epochs = epochs,
                        compute_loss = True, callbacks = [monitor, save_model])

        ### save
        # save parameters used for generating the random walk
        random_walk_param = {"p": p, "q": q, "length": l}
        with open(directory + "/iteration_" + str(i) + "_parameters.pkl", "wb") as f:
            pickle.dump(random_walk_param, f)
        
        # save training loss
        with open(directory + "/word2vec_train/train_loss_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(monitor.train_loss,f)
    
if __name__ == "__main__":
    main()