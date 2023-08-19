
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from implicit.evaluation import mean_average_precision_at_k

from tqdm import tqdm
from pathlib import Path

class ALSRecommender():
    def __init__(self, all_users_df, all_items_df):

        self.als = None
        self.csr_u2i_matrix = None

        self.user_ids = None
        self.item_ids = None
        self.user_map = None
        self.item_map = None
        self.build_ui_maps( all_users_df, all_items_df )

        pass

    def build_ui_maps(self, all_users_df, all_items_df):

        uniq_user_ids_list = all_users_df['customer_id'].unique().tolist()
        uniq_item_ids_list = all_items_df['article_id'].unique().tolist()

        self.user_ids = dict(list(enumerate(uniq_user_ids_list)))
        self.item_ids = dict(list(enumerate(uniq_item_ids_list)))

        self.user_map = {u: uidx for uidx, u in self.user_ids.items()}
        self.item_map = {i: iidx for iidx, i in self.item_ids.items()}

        pass

    def add_mapped_user_item_ids(self, transactions_df):

        transactions_df['user_id'] = transactions_df['customer_id'].map(self.user_map)
        transactions_df['item_id'] = transactions_df['article_id'].map(self.item_map)

        return transactions_df

    def build_csr_u2i_matrix(self, transactions_df):

        row = transactions_df['user_id'].values
        col = transactions_df['item_id'].values

        data = np.ones(transactions_df.shape[0])
        csr_u2i_matrix = csr_matrix((data, (row, col)), shape=(len(self.user_map), len(self.item_map)))

        return csr_u2i_matrix

    def get_cv_datetime_splits(self, transactions_df, cv=5, train_days=28, validation_days=7):

        cv_datetime_splits = []
        total_validation_days = cv * validation_days
        val_start = transactions_df["t_dat"].max() - pd.Timedelta( days=total_validation_days )
        train_start = val_start - pd.Timedelta( days=train_days )
        for i in range( cv ):
            current_split = []
            current_split.append( train_start )
            current_split.append( val_start )
            val_end = val_start + pd.Timedelta( days=validation_days )
            current_split.append( val_end )
            current_split = tuple( current_split )
            cv_datetime_splits.append( current_split )
            val_start = val_end
            train_start = val_start - pd.Timedelta(days=train_days)

        return cv_datetime_splits

    def split_into_train_val(self, transactions_df, datetime_split):
        train_start = datetime_split[0]
        val_start = datetime_split[1]
        val_end = datetime_split[2]

        train_df = transactions_df[ (transactions_df["t_dat"] >= train_start) \
                                   & (transactions_df["t_dat"] < val_start) ]
        val_df = transactions_df[ (transactions_df["t_dat"] >= val_start) \
                                   & (transactions_df["t_dat"] < val_end) ]

        return train_df, val_df

    def evaluate_als_cv(self,
                        transactions_df, cv=5, train_days=28, validation_days=7,
                        factors=100, iterations=12,
                        nlist=400, nprobe=20, use_gpu=False,
                        calculate_training_loss=False,
                        regularization=0.01, random_state=45
                        ):

        cv_datetime_splits = self.get_cv_datetime_splits( transactions_df, cv, train_days, validation_days)

        cv_fold_scores = []
        for datetime_split in tqdm(cv_datetime_splits, desc="Fitting ALS on cv folds"):
            train_df, val_df = self.split_into_train_val( transactions_df, datetime_split )

            self.fit_als(train_df,
                         factors = factors,
                         iterations = iterations,
                         nlist = nlist,
                         nprobe = nprobe,
                         use_gpu = use_gpu,
                         calculate_training_loss = calculate_training_loss,
                         regularization = regularization,
                         random_state = random_state)

            train_csr = self.build_csr_u2i_matrix(train_df)
            val_csr = self.build_csr_u2i_matrix(val_df)
            score = mean_average_precision_at_k(self.als, train_csr, val_csr, K=12, show_progress=True, num_threads=8)
            cv_fold_scores.append( score )
            print("Val period: {} | MAP12 score: {}".format( datetime_split, round(score, 6) ))

        mean_cv_score = np.mean( cv_fold_scores )
        print("Mean CV score: {}".format( mean_cv_score ))

        return mean_cv_score

    def fit_als(self, transactions_df,
                factors=100, iterations=12,
                nlist=400, nprobe=20, use_gpu=False,
                calculate_training_loss=False,
                regularization=0.01, random_state=45):

        csr_u2i_matrix = self.build_csr_u2i_matrix(transactions_df)
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
        self.csr_u2i_matrix = csr_u2i_matrix

        return self

    def recommend(self, user_ids_list, N=12, filter_already_liked_items=False, batch_size=2000):

        recommendations = {}

        mapped_user_ids = []
        not_found_ids = []
        for i in tqdm(range( len(user_ids_list) ), desc="Mapping user ids"):
            current_id = user_ids_list[i]
            if current_id not in self.user_map.keys():
                not_found_ids.append( current_id )
                continue
            mapped_id = self.user_map[ current_id ]
            mapped_user_ids.append( mapped_id )

        for startidx in tqdm(range(0, len(mapped_user_ids), batch_size), desc="Making recommendations"):
            batch = mapped_user_ids[startidx: startidx + batch_size]
            ids, scores = self.als.recommend(batch,
                                             self.csr_u2i_matrix[batch],
                                             N=N,
                                             filter_already_liked_items=filter_already_liked_items)
            for i, userid in enumerate(batch):
                customer_id = self.user_ids[userid]
                user_items = ids[i]
                article_ids = [self.item_ids[item_id] for item_id in user_items]
                recommendations[customer_id] = article_ids

        return recommendations

    def get_user_factors(self):
        if self.als is not None:
            return self.als.user_factors

        raise NotImplementedError("self.als is None")

    def get_item_factors(self):
        if self.als is not None:
            return self.als.item_factors

        raise NotImplementedError("self.als is None")