from bs4 import BeautifulSoup
import requests
import re
import datetime
from collections import defaultdict
import pandas as pd
from pandas.io.pytables import HDFStore
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

mojo_prefix = 'http://www.boxofficemojo.com'
url = mojo_prefix + '/movies/alphabetical.htm?letter=NUM&p=.html'
resp = requests.get(url)
assert resp.ok
soup = BeautifulSoup(resp.content, 'lxml')
links = soup.findAll('a', href=re.compile('letter='))

page_links = []
for link in links:
    p_url = mojo_prefix + link['href']
    page_links.append(p_url)
    resp = requests.get(p_url)
    assert resp.ok
    soup = BeautifulSoup(resp.content, 'lxml')
    plinks = soup.findAll('a', href=re.compile('page='))
    [page_links.append(mojo_prefix + p['href']) for p in plinks]

movie_links = []
count = 0
for p in page_links:
    count+=1
    print(count)
    resp = requests.get(p)
    assert resp.ok
    soup = BeautifulSoup(resp.text, 'lxml')

    mlinks = soup.findAll('a', href=re.compile('id='))
    [movie_links.append(mojo_prefix + mlink['href']) for mlink in mlinks]

movie_links = list(set(movie_links))

def get_movie_value(soup, field_name):
    obj = soup.find(text=re.compile(field_name))
    if not obj:
        return None
    next_sibling = obj.findNextSibling()
    if next_sibling:
        return next_sibling.text.strip()
    else:
        return None

mojo_data = defaultdict(list)
count = 0
error_count = 0
for m in movie_links:
    print(count)
    time.sleep(0.005+np.random.random()/500)
    count += 1
    if m == 'http://www.boxofficemojo.com/movies/?id=behavingbadly.htm':
        continue
    m += '&adjust_yr=2016&p=.htm'

    try:
        resp = requests.get(m)
        assert resp.ok
    except:
        print('ERROR!')
        error_count += 1
        continue

    soup = BeautifulSoup(resp.text, 'lxml')

    rel_str = get_movie_value(soup, 'Release Date')
    if rel_str is None:
        continue
    try:
        rel_date = datetime.datetime.strptime(rel_str, '%B %d, %Y')
    except ValueError:
        continue

    budget = get_movie_value(soup, 'Production Budget')
    gross = get_movie_value(soup, 'Domestic Total Adj')
    genre = get_movie_value(soup, 'Genre:')
    directors = soup.findAll('a', href=re.compile('Director&id'))
    actors = soup.findAll('a', href=re.compile('Actor&id'))

    if rel_date < datetime.datetime(1990, 1, 1):
        print(rel_date)
        continue
    elif ('N/A' == gross) or ('n/a' == gross) or (gross is None):
        print('$')
        continue
    elif ('N/A' == genre) or (genre is None):
        print('g')
        continue
    elif not len(directors):
        continue
    elif not len(actors):
        continue
    else:
        mojo_data['rel_date'].append(rel_date)
        mojo_data['Genre'].append(genre)
        mojo_data['Gross'].append(gross.replace('$', '').replace(',', ''))

        mojo_data['title'].append(soup.find('title').text.split('(')[0].strip())
        print(soup.find('title').text.split('(')[0].strip())

        mojo_data['directors'].append('/'.join([director.decode_contents().strip() for director in directors]))
        writers = soup.findAll('a', href=re.compile('Writer&id'))
        mojo_data['writers'].append('/'.join([writer.decode_contents().strip() for writer in writers]))
        producers = soup.findAll('a', href=re.compile('Producer&id'))
        producers_list = [producer.decode_contents() for producer in producers]
        mojo_data['producers'].append('/'.join([p.split('(')[0].strip() for p in producers_list]))

        actors_list = [actor.decode_contents() for actor in actors]
        mojo_data['actors'].append('/'.join([actor.strip() for actor in actors_list if actor[-1] != '*']))



mojo_df = pd.DataFrame(mojo_data)
mojo_df.drop_duplicates(inplace=True)
mojo_df = mojo_df[mojo_df['Gross'] != 'n/a']
mojo_df.to_csv('1990_data.csv')

temp = mojo_df.copy()
