import os
from tokenize import String  # paths to file
import numpy as np  # linear algebra
import pandas as pd  # data processing
import warnings  # warning filter
import scipy as sp  # pivot egineering
import json
from sklearn.metrics.pairwise import cosine_similarity


pd.options.display.max_columns


warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

rating_path = "./rating.csv"
anime_path = "./anime.csv"


rating_df = pd.read_csv(rating_path)
rating_df.head()

anime_df = pd.read_csv(anime_path)
anime_df.head()

# deleting anime with 0 rating
anime_df = anime_df[~np.isnan(anime_df["rating"])]

# filling mode value for genre and type
anime_df['genre'] = anime_df['genre'].fillna(
    anime_df['genre'].dropna().mode().values[0])

anime_df['type'] = anime_df['type'].fillna(
    anime_df['type'].dropna().mode().values[0])

# checking if all null values are filled
anime_df.isnull().sum()

rating_df['rating'] = rating_df['rating'].apply(
    lambda x: np.nan if x == -1 else x)
rating_df.head(20)


# step 1
anime_df = anime_df[anime_df['type'] == 'TV']

# step 2
rated_anime = rating_df.merge(
    anime_df, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])

# step 3
rated_anime = rated_anime[['user_id', 'name', 'rating']]

# step 4
rated_anime_7500 = rated_anime[rated_anime.user_id <= 7500]
rated_anime_7500.head()


pivot = rated_anime_7500.pivot_table(
    index=['user_id'], columns=['name'], values='rating')
pivot.head()

# step 1
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

# step 2
pivot_n.fillna(0, inplace=True)

# step 3
pivot_n = pivot_n.T

# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)


# model based on anime similarity
anime_similarity = cosine_similarity(piv_sparse)

# Df of anime similarities
ani_sim_df = pd.DataFrame(
    anime_similarity, index=pivot_n.index, columns=pivot_n.index)


def anime_recommendation(ani_name):
    result = {"animes": []}
    number = 1
    print('Recommended because you watched {}:\n'.format(ani_name))
    for anime in ani_sim_df.sort_values(by=ani_name, ascending=False).index[1:6]:
        result["animes"].append(
            {"name": anime, "match": f'{round(ani_sim_df[anime][ani_name]*100, 2)} % match'})
        # result[number] = f'{anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match'
        #     f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match')
        # resultf'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match'
        number += 1
    with open("./anime_recommendation_frontend/data_file.json", "w") as write_file:
        json.dump(result, write_file)


anime_recommendation(input())
