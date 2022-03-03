import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

df_movie = pd.read_csv('/home/thorsten/kaggle/datasets/tmdb_movie_metadata/tmdb_5000_movies.csv')

def pre_work():
    #print df_movie.head(5)

    #print df_movie.info()

    # find duplicate row in data
    #print(df_movie.duplicated().sum())

    # drop any dupliacte rows
    df_movie.drop_duplicates(inplace=True)

    df_movie.drop(['homepage','tagline','production_companies','keywords'], axis=1, inplace=True)

    # find any null values & drop them
    #print(df_movie.isnull().sum())
    df_movie.dropna(inplace=True)

# Q1: Which movie has the most popularity
def part_1():
    df_sorted_movies = df_movie.sort_values(by='popularity', ascending=False)
    titles = df_sorted_movies.original_title[:10]

    # plot graph
    plt.subplots(figsize=(10,5))
    plt.gca().invert_yaxis()
    plt.title('Popular Movies')
    plt.ylabel('Title')
    plt.xlabel('Popularity')
    plt.barh(titles, df_sorted_movies['popularity'].head(10))
    #plt.show()

    # @@ least popular Movies

# Q2: Which movies generate the largest profit
def part_2():

    # set average budget cost for movies without a stipulated budget
    df_movie['budget'].replace(to_replace=0, value=df_movie['budget'].mean(), inplace=True)

    # calculate profit
    df_movie['profit'] = df_movie['revenue'] - df_movie['budget']

    #print(df_movie['profit'].head(15))

    df_q2 = df_movie.sort_values(by='profit', ascending=False).head(10)

    # horizontal bar
    plt.subplots(figsize=(10,5))
    plt.gca().invert_yaxis()
    plt.title('Most Profitable Movies')
    plt.ylabel('Title')
    plt.xlabel('Profit ($)')
    plt.barh(df_q2.original_title, df_q2.profit)
    plt.show()


# Q3: Which genres trend the most, over the years?
def part_3():

    l = []
    ptrn = r"(\"name\":\s\")([^\"]+)(\")"
    for c in df_movie['genres'].str.strip("[]"):
        for k in re.findall(ptrn, c):
            l.append(k[1])

    cntr = Counter(l)
    df_q3 = pd.DataFrame(data=cntr.values(), index=cntr.keys(), columns=['Count'])
    df_q3.sort_values(by=['Count'], ascending=True, inplace=True)

    plt.subplots(figsize=(10,5))
    plt.ylabel('Genre')
    plt.xlabel('Genre Count')
    plt.title('Most Trending Genres')
    plt.barh(df_q3.index, df_q3['Count'])
    plt.show()


if __name__ == "__main__":

    pre_work()
    # part_1()
    # part_2()
    part_3()

    # (\"name\":\s)(\"[^\"]+\")
    # (\"name\":\s)((?=\").+)((?!\"))
    # (name:)(\s*((?:\w(?!\s+\")+|\s(?!\s*\"))+\w)\s*)
    # print len(re.findall(ptrn, df_movie['genres'].iloc[0].strip("[]")))
    # l = []
    # ptrn = r"(\"name\":\s\")([^\"]+)(\")"
    # for c in df_movie['genres'].str.strip("[]"):
    #     for k in re.findall(ptrn, c):
    #         l.append(k[1])
    #
    # cntr = Counter(l)
    # arr_cntr = np.array([list(cntr.keys()), list(cntr.values())]).T
    # print arr_cntr
    # print arr_cntr[arr_cntr[:, 1].argsort()]
    # df_q3 = pd.DataFrame(arr_cntr, columns=['Genre','Count'])

    # df_q3 = pd.DataFrame(data=cntr.values(), index=cntr.keys(), columns=['Count'])
    # df_q3.sort_values(by=['Count'], ascending=False, inplace=True)
    # print df_q3

    # df_sorted_movies = df_movie.sort_values(by='popularity', ascending=False)
    # print df_sorted_movies[['title', 'popularity']]
