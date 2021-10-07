"""musicians module

This module contains the various functions used in building and analyzing the classical musicians recommender.
"""
import itertools
import networkx as nx
import numpy as np

from collections import deque
from stellargraph.data import BiasedRandomWalk

# ------------------------------------------------------------------------
# Functions for building artist graph
# ------------------------------------------------------------------------
def get_album_artists(album_id, sp):
    """Given an album, find the artists that have collaborative tracks in the album. 
    
    This function examines each track of an album and finds the artists for each track.
    The function also creates a dictionary of artist IDs and their names. 

    Parameters
    ----------
    album_id: (str) Spotify's album ID
    sp: spotipy.client.Spotify object
    
    Returns
    -------
    album_artists: (list) of set of artist with a collaborative track.
        Example: [{X,Y}, {X,Y,Z}] indicates that there is at least one track 
        with artists {X,Y}, and there is at least one track with artists {X, Y, Z}. 
    ID_artists: (dict) of artists' Spotify ID and names
        key: ID
        value: artist names
    """
    
    album_artists = []
    ID_artists = dict()
    
    tracks = sp.album_tracks(album_id)['items']
    for track in tracks:
        track_artists = track['artists']
        track_artists_ID = set(artist['id'] for artist in track_artists)
        
        # add track_artists_ID to album_artists if it doesn't already exist
        if track_artists_ID not in album_artists:
            album_artists.append(track_artists_ID)
        
        # update artist_ID dictionary
        for artist in track_artists:
            ID_artists[artist['id']] = artist['name']
   
    return album_artists, ID_artists

def get_collaborators(artist_uri, sp):
    """Given an artist, find the artist's albums and collaborators from each album. 
    
    The function also returns a dictionary of artists' Spotify ID and names. 
    
    Parameters
    ----------
    artist_uri: (str) An artist's Spotify URI
    sp: spotipy.client.Spotify object
    
    Returns
    -------
    collab_artists: (dict) of an artist's albums and collaborators
        key: (str) album ID
        value: (list) list of album artists' URIs. 
        Example: collab_artists['123'] = [{X,Y}, {X,Y,Z}] indicates that in album '123',
        there is at least one track with artists {X,Y}, 
        and there is at least one track with artists {X, Y, Z}. 
    ID_artists: (dict) of artists's Spotify ID and names
        key: (str) artist ID
        value: (str) artist name
    """
    
    ID_artists = {}
    collab_artists = {}
    
    # Find albums by current artist.
    # note: 50 is the maximum limit value allowed
    all_albums = sp.artist_albums(artist_uri, limit = 50, album_type='album')['items']
    
    for album in all_albums:
        album_id = album["id"]
        
        album_artists, IDs = get_album_artists(album_id, sp)
        collab_artists[album_id] = album_artists
        
        # update ID_artists
        ID_artists = {**ID_artists, **IDs}
        
    return collab_artists, ID_artists
        
def build_graph(start_uri, n, sp):
    """Builds a weighted graph G using BFS by visiting 'n' number of artists starting from 'start_uri'.
    
    The function creates initial data structures and calls the function 'continue_building_graph'.
    
    Parameters
    ----------
    start_uri: (str) Spotify URI of starting artist
    n: (int) number of artists to visit
    sp: spotipy.client.Spotify object
    
    Returns
    -------
    G: (networkx graph) updated graph
    queue: (collections.deque) updated queue
    visited: (set) updated set of visited artists
    ID_artists: (dict) updated dictionary of artist ID and names
    """
    # Initialize graph G
    G = nx.Graph()
    Q = deque()
    Q.append(start_uri)
    visited = set()
    ID_artists = dict()
    
    # build graph
    G, Q, visited, ID_artists = continue_building_graph(G, Q, visited, ID_artists, n, sp)

    return G, Q, visited, ID_artists

def continue_building_graph(G, queue, visited, ID_artists, max_count, sp):
    """Builds a weighted graph G using BFS.
    
    Given a (possibly empty) graph G and a (possibly empty) queue, 
    (1) create nodes for new artists,
    (2) create edges for new collaborations,
    (3) update the 'albums' attribute of an edge to keep track of all albums accounted for.
    In the very last step, find the weight of each edge as the number of albums 
    in the 'albums' attribute.
    Furthermore, update the dictionary of artist ID and names. 
    
    Parameters
    ----------
    G: (networkx graph) Can be empty or non-empty
    queue: (collections.deque) of artists (IDs) to be explored
    visited: (set) of artists (IDs) who have been visited
    ID_artists: (dict) of artists' Spotify ID and names
        key: ID
        value: artist name
    max_count: (int) number of artists to visit
    sp: spotipy.client.Spotify object
    
    Returns
    -------
    G: (networkx graph) updated graph
    queue: (collections.deque) updated queue
    visited: (set) updated set of visited artists
    ID_artists: (dict) updated dictionary of artist ID and names
    """

    count = 0
    while queue and (count < max_count):

        current_uri = queue.popleft()
        
        if current_uri in visited:
            pass
        else:
            visited.add(current_uri)

            # get albums, collaborative artists
            collab_artists, IDs = get_collaborators(current_uri, sp)
            # update 'ID_artists'
            ID_artists = {**ID_artists, **IDs}

            # go through each album
            for (album, album_artists) in collab_artists.items():
                
                # go through each collection of collaborators in the album
                for collection in album_artists:
                    # check if the collection consists of more than one artist.
                    if len(collection) > 1:
                        # create artist nodes as needed
                        for artist in collection:
                            if artist not in G:
                                G.add_node(artist)
                                
                        # examine each pair of artists
                        # create new edge or update the 'albums' attribute of existing edge
                        for pair in itertools.combinations(collection,2):
                            # update edge
                            if G.has_edge(pair[0], pair[1]):
                                G[pair[0]][pair[1]]['albums'].add(album)
                            # create edge
                            else:
                                G.add_edge(pair[0], pair[1], weight = 1, albums = {album})

                        # add collaborative artists to the Queue
                        for artist in collection:
                            if artist not in visited:
                                queue.append(artist)
            count += 1
            
    # compute edge weights
    for n1,n2,edge in G.edges(data=True):
        edge['weight'] = len(edge['albums'])
        
    return G, queue, visited, ID_artists
    
# ------------------------------------------------------------------------
# Other functions
# ------------------------------------------------------------------------

def community_members(Gix, partition_membership, n, ID_name):
    """Return 20 members of a community with highest degrees
    
    Parameters
    ----------
    Gix: igraph graph object
    partition_membership: (list) of partition membership. 
    n: (int) community number to inspect
    ID_name: (dict) of artists' Spotify ID and names
            key: Spotify
            value: artist names
            
    Returns
    -------
    print 20 members of a cluster with highest degrees
    """
    
    # find members belonging to cluster
    idx = [i for i, value in enumerate(partition_membership) if value == n]
    
    # find degrees
    idx_degree = [Gix.degree(i) for i in idx]
    max_degree = np.argsort(idx_degree)[::-1][:20]
    
    print("Artists with maximum degree in community %i" %n)
    for i in max_degree:
        print(ID_name[Gix.vs[idx[i]]['label']])

def sample_random_walks(G_s, artists_by_degree, n_walks_by_degree, p = 3.8, q = 1.7, length = 15):
    """sample weighted random walks from G_s
    
    Implements StellarGraph's biased random walk generation for a custom number of random walks for each node. 
    For each artist A in G_s, if A is in artists_by_degree[i] for some i, then sample n_walks_by_degree[i] number of random walks starting from artist A.

    Parameters
    ----------
    G_s: (StellarGraph). Each edge weight must be in [0,1]
    artists_by_degree: (dict) Grouping of artist according to their degree in G_s
            'artists_by_degree[i]' is a list of artists that belong to group 'i'
    n_walks_by_degree: (dict) Specifies the number of random walks to sample for artists in each group.
            If n_walks_by_degree[i] = j, then we sample 'j' number of random walks for each artist in group 'i'
    p: (float) defines un-normalized probability 1/p of returning to source node
    q: (float) defines un-normalized probability 1/q of moving away from source node
    length: (int) length of each random walk

    Returns
    -------
    walks: (list) of random walks.
    """
    walks = []

    rw = BiasedRandomWalk(G_s, 
                          p = p, 
                          q = q, 
                          length = length, 
                          weighted = True, # weighted random walk
                          seed = 0)
    
    for (degree, artists) in artists_by_degree.items():
        sub_walks = rw.run(nodes = artists,
                           n = n_walks_by_degree[degree] # number of random walks per root node 
                           )
        walks += sub_walks

    return walks

def get_related_artists(artist_list, sp):
    """get related artists from Spotify API.
    
    For each artist in `artist_list`, get related artists from Spotify API and save in a dictionary. 
    
    Parameters
    ----------
    artists_list: (set or list) of artist IDs
    sp: spotipy.client.Spotify object
    
    Returns
    -------
    related_artists: (dict) of artist ID and their related artists' IDs. 

    Example use
    -----------
    # get visited artists
    with open('graph_80000/visited_artists.pkl', 'rb') as f:
        visited = pickle.load(f)
    visited = set(visited)

    cid ="" 
    secret = "" 

    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # create dictionary of related artists
    related_artists = get_related_artists(visited, sp)
    """
    related_artists = dict()
    for artist_uri in artist_list:

        results = sp.artist_related_artists(artist_uri)['artists']
        related = set()
        for item in results:
            related.add(item['id'])
        
        related_artists[artist_uri] = related
    return related_artists

def compare(recommended_artists, comparison_artists):
    """Compute precision, recall, F1 between recommended_artists (most similar artists from node2vec embedding) and comparison_artists (either 'related artists' or 'collaborators').
    Returns Nones if comparison_artists is empty.

    Parameters
    ----------
    recommended_artists: (list) of similar artists returned by node2vec
    comparison_artists: (set) of artists to compare to. Either 'related artists' or 'collaborators'

    Returns
    -------
    precision: (float) Of the similar_artists, what proportion are in comparison_artists?
    recall: (float) Of the comparison_artists, what proportion are in similar_artists?
    F1: (float) 2 * precision * recall / (precision + recall)
    """
    
    if len(comparison_artists) == 0:
          return None, None, None
    else:
        recommended_artists = [item[0] for item in recommended_artists]
        TP = list(set(recommended_artists) & comparison_artists)

        precision = len(TP)/len(recommended_artists)
        recall = len(TP)/len(comparison_artists)
    
    if (precision != 0) and (recall != 0):
        F1 = 2 * precision * recall / (precision + recall)
    else:
        F1 = 0
    return precision, recall, F1

def compute_metrics(model, comparison_dict, topn):
    """Compute the precision, recall, and F1 for all artists. Note the score is computed only for artists whose values of comparison_dict is non-empty
    
    Parameters
    ----------
    model: a trained word2vec model
    comparison_dict: (dict) To compare the performance of the model to.
                    Either 'related_artists' or 'neighbors'
    
    Returns
    -------
    precisions: (list) of precision scores for each artist
    recalls: (list) of recall scores for each artist
    F1: (list) of F1 scores for each artist
    """
    
    precisions = []
    recalls = []
    F1 = []
    
    for artist, artist_neighbor in comparison_dict.items():
        try:
            similar_artists = model.wv.most_similar(artist, topn = topn)
            if len(artist_neighbor) > 0:
                p, r, f1 = compare(similar_artists, artist_neighbor)
                precisions.append(p)
                recalls.append(r)
                F1.append(f1)
        except:
            pass
        
    return precisions, recalls, F1

def get_similar_artists(model, ID, ID_name, topn = 20):
    """Return a string of artists that are similar to a given artist (ID)
    
    Parameters
    ----------
    model: (word2vec) model
    ID: (string) of artist's Spotify ID
    ID_name: (dict) of Spotify ID and names
    topn: (int) number of similar artists to find

    Returns
    -------
    sim: (string) of artists that are similar to given artist
    """
    sim = model.wv.most_similar(ID, topn = topn)
    sim = [ID_name[item[0]] for item in sim]
    sim = ', '.join(sim)
    
    return sim

def recommend_artists(favorites, model, ID_name, n):
    """Return recommended artists based on proximity to the average vector of the favorite artists.
    
    Parameters
    ----------
    favorites: (list) of favorite artists' IDs
    model: trained word2vec model
    ID_name: (dict) of Spotify ID and artist names
    n: (int) number of artists to recommend
    
    Returns
    -------
    recommend: (list) of artists to recommend
    """
    
    v = np.zeros((model.vector_size, ))
    for artist in favorites:
        v += model[artist]
    v = v/len(favorites)
    
    rec = model.similar_by_vector(v, topn = n + len(favorites)) 
    rec = [ID_name[item[0]] for item in rec if item[0] not in favorites]
    return rec