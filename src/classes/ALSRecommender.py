
import numpy as np
from scipy.sparse import csr_matrix
import implicit

from tqdm import tqdm
from pathlib import Path

class ALSRecommender():
    def __init__(self):

        self.als = None

        self.user_id_dict = None
        self.inverted_user_id_dict = None

        self.item_id_dict = None
        self.inverted_item_id_dict = None

        self.most_popular_item_ids = None
        self.user_items_dict = None

        pass

    def build_csr_u2i_matrix(self, frequency_dict, use_sum_counts = False, n_most_popular=12):

        user_ids = []
        item_ids = []
        item_sum_counts = []
        for user_id in tqdm(frequency_dict.keys(), desc="Building sparse matrix User2Item"):
            user_items_dict = frequency_dict[user_id]
            for item_id in user_items_dict.keys():
                item_ids.append(item_id)
                user_ids.append(user_id)

                if use_sum_counts:
                    buy_count = user_items_dict[item_id]
                    item_sum_counts.append( buy_count )
                else:
                    item_sum_counts.append(1)

        # set most popular items
        purchased_items_count = {}
        for i in range(len(item_sum_counts)):
            item_id = item_ids[i]
            if item_id not in purchased_items_count.keys():
                purchased_items_count[ item_id ] = 0
            purchased_items_count[ item_id ] += item_sum_counts[i]
        item_id_keys = list(purchased_items_count.keys())
        item_purchased_counts = list( purchased_items_count.values() )
        most_popular_list_ids = np.argsort(item_purchased_counts)[-n_most_popular:][::-1]
        most_popular_item_ids = []
        for popular_id in most_popular_list_ids:
            most_popular_item_ids.append( item_id_keys[popular_id] )
        self.most_popular_item_ids = most_popular_item_ids

        uniq_user_ids = list(sorted(list(set(user_ids))))
        user_id_dict = {}
        for i in tqdm(range(len(uniq_user_ids)), desc="Building user --> id mapping"):
            user_id_dict[uniq_user_ids[i]] = i
        inverted_user_id_dict = {v: k for k, v in user_id_dict.items()}
        self.user_id_dict = user_id_dict
        self.inverted_user_id_dict = inverted_user_id_dict

        uniq_items = list(sorted(list(set(item_ids))))
        item_id_dict = {}
        for i in tqdm(range(len(uniq_items)), desc="Building item --> id mapping"):
            item_id_dict[uniq_items[i]] = i
        inverted_item_id_dict = {v: k for k, v in item_id_dict.items()}
        self.item_id_dict = item_id_dict
        self.inverted_item_id_dict = inverted_item_id_dict

        # set purchased items
        """user_items_dict = {}
        for user_id in frequency_dict.keys():
            matrix_user_id = self.user_id_dict[user_id]
            user_purchases = frequency_dict[user_id]
            user_items_dict[matrix_user_id] = []
            for user_purchased_item in user_purchases.keys():
                matrix_item_id = self.item_id_dict[ user_purchased_item ]
                user_items_dict[matrix_user_id].append( matrix_item_id )
            user_items_dict[matrix_user_id] = np.array( user_items_dict[matrix_user_id] )
        self.user_items_dict = user_items_dict"""

        for i in tqdm(range(len(item_sum_counts)), desc="Mapping content for building sparse matrix"):
            user_ids[i] = user_id_dict[user_ids[i]]
            item_ids[i] = item_id_dict[item_ids[i]]
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        item_sum_counts = np.array(item_sum_counts)

        csr_u2i_matrix = csr_matrix((item_sum_counts, (user_ids, item_ids)),
                                                shape=(np.max(user_ids) + 1, np.max(item_ids) + 1))

        return csr_u2i_matrix

    def fit_als(self, csr_u2i_matrix,
                factors=50, iterations=100,
                nlist=400, nprobe=20, use_gpu=False,
                calculate_training_loss=False,
                regularization=0.1, random_state=45):

        # set items_dict for users with transactions
        user_items_dict = {}
        item_ids = []
        for i in range(csr_u2i_matrix.shape[1]):
            item_ids.append( i )
        item_ids = np.array( item_ids )
        for i in tqdm(range(csr_u2i_matrix.shape[0]), desc="Extracting purchased "):
            user_possible_items = csr_u2i_matrix[i, :].toarray()[0]
            user_purchased_items = item_ids[ user_possible_items != 0 ]
            user_items_dict[ i ] = user_purchased_items
        self.user_items_dict = user_items_dict

        # set n_most_popular for users without transactions
        """sum_counts = []
        for i in tqdm(range( csr_u2i_matrix.shape[1] ), desc="Finding most popular items"):
            item_counts = csr_u2i_matrix[:, i]
            item_sum_count = np.sum( item_counts )
            sum_counts.append( item_sum_count )
        sum_counts = np.array( sum_counts )
        most_popular_item_ids = np.argsort(sum_counts)[-n_most_popular:][::-1]
        self.most_popular_item_ids = []
        for popular_item_id in most_popular_item_ids:
            direct_item_id = self.inverted_item_id_dict[popular_item_id]
            self.most_popular_item_ids.append( direct_item_id )"""

        als = implicit.approximate_als.FaissAlternatingLeastSquares(factors = factors,
                                                                    iterations = iterations,
                                                                    nlist = nlist,
                                                                    nprobe = nprobe,
                                                                    use_gpu = use_gpu,
                                                                    calculate_training_loss = calculate_training_loss,
                                                                    regularization = regularization,
                                                                    random_state = random_state)

        als.fit( csr_u2i_matrix )
        self.als = als

        return self

    def recommend(self, user_ids, N=12, filter_already_liked_items=False):

        recommendations = {}
        already_seen_users_count = 0
        new_users_count = 0
        for user_id in tqdm(user_ids, desc = "Making recommendations for each user"):
            if user_id in self.user_id_dict.keys():
                matrix_user_id = self.user_id_dict[user_id]
                user_items = self.user_items_dict[matrix_user_id]
                user_recommendations = self.als.recommend( matrix_user_id, user_items, N=N,
                                                           filter_already_liked_items=filter_already_liked_items )
                user_recommendations = user_recommendations[0]
                user_recommendations = list(user_recommendations)
                for i, matrix_item_id in enumerate(user_recommendations):
                    user_recommendations[i] = self.inverted_item_id_dict[matrix_item_id]
                already_seen_users_count += 1
            else:
                user_recommendations = self.most_popular_item_ids
                new_users_count += 1
            recommendations[ user_id ] = user_recommendations

        print("Already seen users: {}".format( already_seen_users_count ))
        print("New users: {}".format( new_users_count ))

        return recommendations

    def get_user_factors(self):
        if self.als is not None:
            return self.als.user_factors

        raise NotImplementedError("self.als is None")

    def get_item_factors(self):
        if self.als is not None:
            return self.als.item_factors

        raise NotImplementedError("self.als is None")