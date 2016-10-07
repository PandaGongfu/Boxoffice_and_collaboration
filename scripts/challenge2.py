import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import seaborn as sns
import matplotlib.cm as cm

movie_df = pd.read_csv('1990_mojo.csv')
movie_df['rel_date'] = movie_df['rel_date'].apply(lambda x:
                                                  datetime.datetime.strptime(x, '%Y-%m-%d'))
movie_df = movie_df[movie_df['rel_date'] < datetime.datetime(2016, 1, 1)]

time_df = movie_df.copy()
time_df['rel_date'] = time_df['rel_date'].apply(mdt.date2num)

sns.set(color_codes=True)
fig, ax = plt.subplots(figsize=(10, 3))
sns.regplot(x='rel_date', y='Gross', data=time_df, truncate=True)
ax.xaxis.set_major_locator(mdt.AutoDateLocator())
ax.xaxis.set_major_formatter(mdt.DateFormatter('%b %y'))
ax.set_ylim(0, 1e9)
ax.set_xlim(726286, 736157)
ax.set_title('Domestic Gross v.s. Release Date')

# 2.2
movie_2013 = pd.read_csv('2013_movies.csv')
sns.regplot(x='Runtime', y='DomesticTotalGross', data=movie_2013)

# 2.3
by_rating = movie_2013.groupby(['Rating']).mean()
by_rating.drop('Budget', axis=1, inplace=True)
by_rating

#2.4
movie_rating = movie_df.copy()
movie_rating.reset_index(inplace=True)

ratings = list(set(movie_rating['MPAA'].tolist()))
ratings.sort()
ratings = np.array(ratings)
ratings = ratings.reshape((2, 3))

colors = cm.rainbow(np.linspace(0, 1, 6))

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
for i, rating_row in enumerate(ratings):
    for j, rating in enumerate(rating_row):
        rating_df = movie_rating[movie_rating['MPAA'] == rating]
        axes[i][j].scatter(rating_df['rel_date'], rating_df['Gross'], c=colors[3*i+j])
        axes[i][j].set_title(rating)
        axes[i][j].xaxis.set_major_locator(mdt.AutoDateLocator())
        axes[i][j].xaxis.set_major_formatter(mdt.DateFormatter('%b %y'))
        axes[i][j].set_ylim(0, 7e8)
        fig.autofmt_xdate()
fig.suptitle('Domestic Gross vs Relase Date')

# 2.5
director_dict = defaultdict(list)
for _, row in movie_df.iterrows():
    if type(row['directors']) is str:
        d_s = row['directors'].split('/')
        if len(d_s):
            for d in d_s:
                director_dict['Gross'].append(row['Gross'])
                director_dict['director'].append(d)
director_df = pd.DataFrame(director_dict)
df = director_df.groupby('director').mean().reset_index()
df.sort_values('Gross', ascending=False, inplace=True)

print('The director has the highest average gross is: \033[1m %s \033[0m' % df.iloc[0, 0])

# 2.6
movie_df.set_index(['rel_date'], inplace=True)

gross_by_month = movie_df[['Gross']].groupby(pd.TimeGrouper(freq='M')).sum()
gross_by_month.reset_index(inplace=True)
gross_by_month['year'] = gross_by_month['rel_date'].apply(lambda x: x.year)
gross_by_month['month'] = gross_by_month['rel_date'].apply(lambda x: datetime.datetime(2015, x.month, 1))
gross_by_month['month'] = gross_by_month['month'].apply(mdt.date2num)


fig, ax = plt.subplots(figsize=(10, 3))
sns.tsplot(gross_by_month, time='month', unit='year', value='Gross')

ax.xaxis.set_major_locator(mdt.AutoDateLocator())
ax.xaxis.set_major_formatter(mdt.DateFormatter('%b'))
# fig.autofmt_xdate()

ax.set_xlabel('')
ax.set_ylabel('Monthly Gross Box Office')
ax.set_title('Average Monthly Gross Box Office from 2000-2015 with CI (in 2016 dollars)');


