import pandas as pd
import datetime
import numpy as np
from collections import defaultdict


def budget_process(row, adj):
    bs = row['budget'].split(' ')
    year = row['rel_date'].year
    if len(bs) == 1:
        return float(bs[0])*adj[year]
    elif bs[1] == 'million':
        return float(bs[0])*1e6*adj[year]


def genre_process(s):
    words = s.split(' ')
    if words[0] == 'Romantic':
        words[0] = 'Romance'
    if len(words) == 1:
        if words[0] in key_genres:
            return words[0],
        return ()
    else:
        if words[0] in key_genres:
            if words[-1] in key_genres:
                return words[0], words[-1]
            return words[0],
        elif words[-1] in key_genres:
            return words[-1],
        return ()


def get_genre_dummy(genre_df):
    genre_dict = defaultdict(list)
    for _, row in genre_df.iterrows():
        if len(row['Genre']):
            genre_dict['title'].append(row['title'])
            genre_dict['Genre'].append(row['Genre'][0])
        if len(row['Genre']) == 2:
            genre_dict['title'].append(row['title'])
            genre_dict['Genre'].append(row['Genre'][1])
    genre_df = pd.DataFrame(genre_dict)
    genre_dummy = pd.get_dummies(genre_df['Genre'])
    genre_df = pd.concat([genre_df, genre_dummy], axis=1)
    genre_final = genre_df.groupby('title').sum().reset_index()
    return genre_final

# clean data
inflation = pd.read_csv('inflation.csv')

inf_adj = np.cumprod(np.array([float(x.split('%')[0].strip())/100+1 for x in inflation['inflation'].tolist()]))
years = inflation.year.tolist()

year_adj = defaultdict(float)
for y, adj in zip(years, inf_adj):
    year_adj[y] = adj

key_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Horror', 'Thriller',
              'Crime', 'Fantasy', 'Romance', 'Sci-Fi']

mojo_df = pd.read_csv('mojo_data.csv')
mojo_df['rel_date'] = mojo_df['rel_date'].apply(lambda x:
                                                datetime.datetime.strptime(x, '%Y-%m-%d'))

mojo_df['Gross'] = mojo_df['Gross'].apply(int)
mojo_df = mojo_df[mojo_df['rel_date'] < datetime.datetime(2016, 1, 1)]
mojo_df['budget'] = mojo_df.apply(lambda x: budget_process(x, year_adj), axis=1)

mojo_df['ROI'] = mojo_df['Gross']/mojo_df['budget'] - 1
mojo_df = mojo_df[mojo_df['budget'] >= 2e6]

mojo_df = mojo_df[mojo_df['directors'].notnull()]
mojo_df = mojo_df[mojo_df['actors'].notnull()]


# genre processing
mojo_df['Genre'] = mojo_df['Genre'].apply(genre_process)
mojo_df = mojo_df[mojo_df['Genre'] != ()]
genre_df = mojo_df[['title', 'Genre']].copy()
genre_final = get_genre_dummy(genre_df)

mojo_final = pd.merge(mojo_df, genre_final, on='title')

# create master movie dataframe
movies_df = pd.read_csv('1990_mojo.csv')
movies_df.drop('Unnamed: 0', axis=1, inplace=True)

movies_df['Gross'] = movies_df['Gross'].apply(int)
movies_df['rel_date'] = movies_df['rel_date'].apply(lambda x:
                                                    datetime.datetime.strptime(x, '%Y-%m-%d'))
movies_df = movies_df[movies_df['rel_date'] <= datetime.datetime(2015, 1, 1)]

genre_df = movies_df[['title', 'Genre']].copy()
genre_df['Genre'] = genre_df['Genre'].apply(genre_process)
genre_final = get_genre_dummy(genre_df)

movies_final = pd.merge(movies_df, genre_final, on='title')

# build graphs for 1990-1998


def find_pair(row):
    da_pair = []
    dp_pair = []
    pw_pair = []

    d_s = [d.split('*')[0] for d in row['directors'].split('/')]
    a_s = row['actors']
    if type(a_s) is str:
        da_pair = [(x, y) for x in d_s for y in a_s.split('/') if x != y]

    pr_s = row['producers']
    w_s = row['writers']
    if type(pr_s) is str:
        pr_s = pr_s.split('/')
        dp_pair = [(x, y) for x in d_s for y in pr_s if x != y]
        if type(w_s) is str:
            pw_pair = [(x, y) for x in pr_s for y in w_s.split('/') if x != y]

    return da_pair, dp_pair, pw_pair


def build_graph(movies):
    pairs = ([], [], [])
    for _, row in movies.iterrows():
        pair = find_pair(row)
        [ps.extend(p) for ps, p in zip(pairs, pair)];
    return pairs

hw_graph_10yr = defaultdict(list)

last_y = 1988
for year in range(1998, 2015):
    add_movies = movies_final[(movies_final['rel_date'] > datetime.datetime(last_y, 12, 31))
                              & (movies_final['rel_date'] <= datetime.datetime(year, 12, 31))]

    da, dp, pw = build_graph(add_movies)

    da_dict = Counter(da)
    dp_dict = Counter(dp)
    pw_dict = Counter(pw)

    hw_graph_10yr[year] = [da_dict, dp_dict, pw_dict]
    last_y += 1


# actor and director star power


def get_star_score(movies_master, p_dict):
    score_by_year = defaultdict(list)
    for year in range(1998, 2015):
        print(year)
        by_year = defaultdict(list)
        for p, ms in p_dict.items():
            titles_df = pd.DataFrame({'title': ms})
            df = pd.merge(titles_df, movies_master, on='title')
            cutoff_date = datetime.datetime(year, 12, 31)
            df = df[df['rel_date'] > cutoff_date - datetime.timedelta(weeks=522)]
            df = df[df['rel_date'] <= datetime.datetime(year, 12, 31)]
            by_year['person'].append(p)
            by_year['total'].append(df['Gross'].sum())
        year_df = pd.DataFrame(by_year)
        year_df.sort_values('total', ascending=False, inplace=True)
        year_df.reset_index(inplace=True)
        year_df['score'] = year_df['total']/1e6
        year_df['rank'] = year_df.index
        score_by_year[year] = year_df
    return score_by_year

actor_scores = get_star_score(movies_final, actors)
director_scores = get_star_score(movies_final, directors)




# calculate score, exposure

def find_n_cooperation(ps, p_dict):
    n_coop = 0
    if len(ps):
        for p in ps:
            n_coop += p_dict[p]
        return n_coop/len(ps)
    return n_coop


def get_score_star(p_s, scores, year, max_p):
    p_df = pd.DataFrame({'person': p_s[:max_p]})
    rank_df = pd.merge(p_df, scores[year], on='person')
    score = rank_df['score'].mean()
    star_count = rank_df[rank_df['rank'].isin(range(100))].shape[0]
    return score, star_count


def get_gross_exp(movies_master, p_s, p_list, max_p, g_s, c_date):
    scores = []
    exposures = []
    for p in p_s[:max_p]:
        titles_df = pd.DataFrame({'title': p_list[p]})
        df = pd.merge(titles_df, movies_master, on='title')
        df = df[df['rel_date'] <= c_date]

        #look back 10 yrs
        i_date = c_date - datetime.timedelta(weeks=522)
        df = df[df['rel_date'] >= i_date]
        exposures.append(sum([df[g].sum() for g in g_s]))

        if df.shape[0]:
            scores.append(df['Gross'].mean())
        else:
            scores.append(0)

    return np.mean(scores), np.mean(exposures)


# feature generation

MAXD = 2
MAXA = 4
count = 0
movies_dict = defaultdict(list)
for _, row in mojo_df.iterrows():
    print(count)
    count += 1
    d_s = [d.split('*')[0] for d in row['directors'].split('/')]
    a_s = row['actors'].split('/')

    cutoff_date = row['rel_date'] - datetime.timedelta(weeks=104)
    prod_year = cutoff_date.year
    genres = row['Genre']

    _, d_exp = get_gross_exp(movies_final, d_s, directors, MAXD, genres, cutoff_date)
    _, a_exp = get_gross_exp(movies_final, a_s, actors, MAXA, genres, cutoff_date)

    d_score, _ = get_score_star(d_s, director_scores, prod_year, MAXD)
    a_score, a_star = get_score_star(a_s, actor_scores, prod_year, MAXA)

    da_pair, dp_pair, pw_pair = find_pair(row)
    [da_dict, dp_dict, pw_dict] = hw_graph_10yr[prod_year]

    movies_dict['title'].append(row['title'])
    movies_dict['d_score'].append(d_score)
    movies_dict['d_exp'].append(d_exp)

    movies_dict['a_score'].append(a_score)
    movies_dict['a_exp'].append(a_exp)
    movies_dict['a_star'].append(a_star)

    movies_dict['da_coop'].append(find_n_cooperation(da_pair, da_dict))
    movies_dict['dp_coop'].append(find_n_cooperation(dp_pair, dp_dict))
    movies_dict['pw_coop'].append(find_n_cooperation(pw_pair, pw_dict))

scores_df = pd.DataFrame(movies_dict)
scores_df.to_csv('scores.csv', ignore_index=True)
mojo_final.to_csv('mojo_final.csv', ignore_index=True)
pickle.dump(hw_graph_10yr, open('graph_10yr.pickle', 'wb'))
pickle.dump(actor_scores, open('actor_scores.pickle', 'wb'))
pickle.dump(director_scores, open('director_scores.pickle', 'wb'))



