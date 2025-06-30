import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder
import json
import itertools
import os
import random
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
import implicit.evaluation
from collections import Counter
class MusicReccomender:
    def __init__(self):
        self. user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()
        self.original_song_encoder = LabelEncoder() # Use a distinct encoder for initial mapping
        self.song_key_map = {}
        self.popular_songs = None
        self.user_item_matrix = None
        self.model = None

    def create_song_key(self, row):
        if pd.notna(row['name_mbid']) and row['name_mbid'].strip():
            key = (str(row['artist']), str(row['name']), str(row['album']), str(row['name_mbid']))
        else:
            key = (str(row['artist']), str(row['name']), str(row['album']))
        return json.dumps(key)
    
    
    def preprocess_data(self, df:pd.DataFrame):
        df['song_key'] = df.apply(self.create_song_key, axis=1) # create song key for identifying the song
        # simplify the dataset by encoding user and song data into two columns
        df['user_id'] = self.user_encoder.fit_transform(df['user'])
        self.original_song_encoder.fit(df['song_key']) 
        df['original_song_id'] = self.original_song_encoder.transform(df['song_key'])

        play_counts = df.groupby(['user_id', 'original_song_id']).size().reset_index(name='count')
        filtered_play_counts = play_counts[play_counts['count'] > 7].copy() 

        remaining_song_keys = df[df['original_song_id'].isin(filtered_play_counts['original_song_id'])]['song_key'].unique()
        self.song_encoder = LabelEncoder() 
        self.song_encoder.fit(remaining_song_keys)


        filtered_play_counts['song_id'] = self.song_encoder.transform(
            self.original_song_encoder.inverse_transform(filtered_play_counts['original_song_id'])
        )

        users = filtered_play_counts['user_id']
        items = filtered_play_counts['song_id']
        counts = filtered_play_counts['count'].astype(np.int16)
        
        self.user_item_matrix = csr_matrix(
            (counts, (users,items)),
            shape=(len(self.user_encoder.classes_), len(self.song_encoder.classes_))
        )

        self.song_key_map = dict(zip([str(x) for x in self.song_encoder.transform(remaining_song_keys)], remaining_song_keys))

        song_counts = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        self.popular_songs = np.argsort(-song_counts)

        print(f"Final User-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"Number of non-zero elements (interactions): {self.user_item_matrix.nnz}")
        print(f"Number of unique songs after filtering: {len(self.song_encoder.classes_)}")

    def train_model(self, factors, iterations, regularization, num_threads, bm25_k1, bm25_b):
        item_user_matrix = self.user_item_matrix.T.tocsr()
        weighted_matrix = bm25_weight(item_user_matrix,K1=bm25_k1,B=bm25_b).tocsr()
        training_matrix = weighted_matrix.T.tocsr()
        print(f"Starting model fit with training_matrix shape: {training_matrix.shape} and dtype: {training_matrix.dtype}")

        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            num_threads=num_threads,
            random_state=3
            )
        try:
            self.model.fit(training_matrix)
        except Exception as e:
            print(f"An exception occurred during model.fit(): {e}")


    def reccomend(self,user_data:pd.DataFrame, num_reccomendations=20):
        print(f"{self.user_item_matrix.nnz}")
        user_data['song_key'] = user_data.apply(lambda row: self.create_song_key(row), axis=1)
        if not hasattr(self, "_song_encoder_classes_set"):
            self._song_encoder_classes_set = set(self.song_encoder.classes_)
        bool_mask = user_data['song_key'].isin(self._song_encoder_classes_set)
        filtered_keys = user_data.loc[bool_mask, 'song_key']
        
        songs_count = Counter()
        if not filtered_keys.empty:
            song_ids = self.song_encoder.transform(filtered_keys)
            songs_count = Counter(song_ids)
        
        print("Loaded songs count")
        if not songs_count:
            print("Something is wrong with reccomendation system. Reccomending popular songs instead:")
            return self.get_popular_reccomendations(num_reccomendations=num_reccomendations)
        
        song_ids = list(songs_count.keys())
        data = [songs_count[i] for i in song_ids]
        print(f"Loaded model.item_factors shape: {self.model.item_factors.shape}")
        user_vector = csr_matrix(
            (data, song_ids, [0, len(song_ids)]),
            shape=(1, len(self.song_encoder.classes_))
        )
        item_ids, score = self.model.recommend(
            userid=0,
            user_items=user_vector,
            N=num_reccomendations,
            filter_already_liked_items=True
        )
        reccomended = [i for i in item_ids]
        return self.get_song_details(reccomended)

    
    def get_popular_reccomendations(self, num_reccomendations):
        return self.get_song_details(self.popular_songs[:num_reccomendations])
    
    def get_song_details(self, song_ids):
        song_details = []
        for song_id in song_ids:
            key = json.loads(self.song_key_map.get(song_id))
            if not key:
                continue
            try:
                artist, name, album, *mbid = key
                details = {'artist': artist, 'name':name,'album':album}
                if mbid:
                    details['mbid'] = mbid[0]
                song_details.append(details)
            except:
                continue
        return song_details
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/user_classes.npy",self.user_encoder.classes_)
        np.save(f"{path}/song_classes.npy", self.song_encoder.classes_)
        np.save(f"{path}/popular_songs.npy", self.popular_songs)
        with open(f"{path}/song_key_map.json","w") as f:
            json.dump(self.song_key_map, f)
        
        model_params = {
        'user_factors': self.model.user_factors,
        'item_factors': self.model.item_factors,
        'factors': np.array([self.model.factors]),
        'regularization': np.array([self.model.regularization]),
        'alpha': np.array([self.model.alpha]),
        'iterations': np.array([self.model.iterations]),
        'random_state': np.array([self.model.random_state], dtype=object),
        'num_threads': np.array([self.model.num_threads])
        }
        save_npz(f"{path}/user_item_matrix.npz", self.user_item_matrix)
        np.savez(f"{path}/model.npz", **model_params)


    def load_model(self,path):

        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()

        self.user_encoder.classes_ = np.load(f"{path}/user_classes.npy", allow_pickle=True)
        self.song_encoder.classes_ = np.load(f"{path}/song_classes.npy", allow_pickle=True)
        self.popular_songs = np.load(f"{path}/popular_songs.npy", allow_pickle=True)
        with open(f"{path}/song_key_map.json", 'r') as f:
            self.song_key_map = {int(k): v for k, v in json.load(f).items()}
        data = np.load(f"{path}/model.npz", allow_pickle=True)
        self.user_item_matrix = load_npz(f"{path}/user_item_matrix.npz").tocsr()
        print(data['factors'])
        self.model = AlternatingLeastSquares(
            factors=int(data['factors']),
            regularization=float(data['regularization']),
            alpha=float(data['alpha']),
            iterations=int(data['iterations']),
            random_state=data['random_state'],
            num_threads=int(data['num_threads'])
            )
        self.model.user_factors = data['user_factors']
        self.model.item_factors = data['item_factors']

    def make_evaluations(self, hyperparams:dict, train_size=0.8):
        user_item_matrix_coo = self.user_item_matrix.tocoo()
        train_coo, test_coo = implicit.evaluation.train_test_split(
            ratings=user_item_matrix_coo,
            train_percentage=train_size
        )
        
        train_csr, test_csr = train_coo.tocsr(), test_coo.tocsr()


        temp_model = AlternatingLeastSquares(
            factors=hyperparams["factors"],
            iterations=hyperparams["iters"],
            regularization=hyperparams["regularization"],
            num_threads=4,
            random_state=3
        )
        train_weighted = bm25_weight(train_csr.T.astype(np.float32),K1=hyperparams["k1"],B=hyperparams['b']).tocsr()
        train_training = train_weighted.T.tocsr()
        temp_model.fit(train_training)

        r_at_k = implicit.evaluation.ranking_metrics_at_k(
            model=temp_model,
            train_user_items=train_csr,
            test_user_items=test_csr,
            K=20,
            show_progress=True,
            num_threads=0
        )

        return r_at_k

    def tune_hyperparams(self):
        hyperparams_list = {
        'factors': (64, 128, 256, 518),
        'iters': (10, 20, 32, 40, 50, 60),
        'regularization':(0.5, 1.0, 2.0, 5.0),
        'k1': (1.2, 1.6, 2.0, 5.0, 20),
        'b': (0.7, 0.8, 0.9)
        }
        results = []
        combs = []
        n_combinations = 50

        combinations = list(itertools.product(*hyperparams_list.values()))

        if len(combinations) <= n_combinations:
            sampled_combinations = combinations
        else:
            sampled_combinations = random.sample(combinations, n_combinations)

        for i, combination in enumerate(sampled_combinations):
            print(f"model {i}")
            c_factors, c_iters, c_reg, c_k1, c_b = combination
            c_hyperparams = {
                'factors': c_factors,
                'iters': c_iters,
                'regularization': c_reg,
                'k1': c_k1, 
                'b': c_b
            }
            evals = self.make_evaluations(hyperparams=c_hyperparams)
            print(f"{i}: {evals} for f:{c_factors}, i:{c_iters}, r:{c_reg}, k:{c_k1}")
            results.append(evals)
            combs.append(c_hyperparams)

        best_idx = np.argmax([res['auc'] for res in results])
        print(f"best auc: {results[best_idx]['auc']} with params {combs[best_idx]}")
        print(results)






reccomender = MusicReccomender()
# df = pd.read_csv("lastfm_recent_tracks.csv")
# reccomender.preprocess_data(df=df)

# reccomender.train_model(factors=128,iterations=20,regularization=2.0,num_threads=4,bm25_k1=1.2,bm25_b=0.9)
# reccomender.save_model('reccomendation_model')
reccomender.load_model("reccomendation_model_")
user_history = pd.read_csv("lastfm_recent_tracks_user.csv")


reccomendations = reccomender.reccomend(user_history,num_reccomendations=20)

for song in reccomendations:
    print(song)


best_hyperparams = {
    'factors': 128,
    'iters': 20,
    'regularization': 2.0,
    'k1': 0.9,
    'b': 0.75
}

print(reccomender.make_evaluations(hyperparams=best_hyperparams, train_size=0.8))
reccomender.tune_hyperparams()

# CURRENT SCORE: 0.55 for <8 plays and 150000 uniques
#IMPROVEMENT:
# - Remove duplicate interactions and create distinct listening events connecting timestamps to the model
# - maybe filter out data further
# - experiment with k1 and b 

# if less than 0.6 - create new dataset
