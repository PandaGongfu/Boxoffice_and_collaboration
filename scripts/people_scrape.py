from bs4 import BeautifulSoup
import requests
import re
from collections import defaultdict
import pickle
import time
import numpy as np


def get_movies(people_links):
    people = defaultdict(list)
    mojo_people_prefix = 'http://www.boxofficemojo.com/people'
    count = 0
    for p_link in people_links:
        print(count)
        count += 1
        time.sleep(0.25 + np.random.random() / 50)
        p_url = mojo_people_prefix + p_link['href'][1:]
        resp = requests.get(p_url)
        assert resp.ok
        soup = BeautifulSoup(resp.content, 'lxml')
        m_links = soup.findAll('a', href=re.compile('movies'))
        people[p_link.text] = list(set([m_link.text.split('(')[0].strip() for m_link in m_links[1:]]))
    return people


def get_people_links(role_str):
    mojo_prefix = 'http://www.boxofficemojo.com'
    mojo_people_prefix = 'http://www.boxofficemojo.com/people'
    url = mojo_people_prefix+'/?view='+role_str+'&p=.htm'
    resp = requests.get(url)
    assert resp.ok
    soup = BeautifulSoup(resp.content, 'lxml')
    people_links = soup.findAll('a', href=re.compile(role_str+'&id='))

    if role_str in ['Actor', 'Director']:
        page_links = [mojo_prefix + p_link['href'] for p_link in soup.findAll('a', href=re.compile('pagenum='))]
        page_links = list(set(page_links))
        for p_link in page_links:
            resp = requests.get(p_link)
            assert resp.ok
            soup = BeautifulSoup(resp.content, 'lxml')
            people_links += soup.findAll('a', href=re.compile(role_str + '&id='))
    return people_links

actor_links = get_people_links('Actor')
actors = get_movies(actor_links)

director_links = get_people_links('Director')
directors = get_movies(director_links)

producer_links = get_people_links('Producer')
producers = get_movies(producer_links)

writer_links = get_people_links('Writer')
writers = get_movies(writer_links)


pickle.dump(actors, open('actor.p', 'wb'))
pickle.dump(directors, open('director.p', 'wb'))
pickle.dump(producers, open('producer.p', 'wb'))
pickle.dump(writers, open('writer.p', 'wb'))


# data_actor = pickle.load(open('actor.p', 'rb'))

ts = [len(x) for x in list(writers.values())]




