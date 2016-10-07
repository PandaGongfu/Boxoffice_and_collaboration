import networkx as nx
import matplotlib.pyplot as plt
kn=nx.karate_club_graph()
kn = nx.read_gml('karate.gml')

G = nx.watts_strogatz_graph(24,4,.3)
nx.draw_circular(G)
plt.show()

from graphviz import Digraph
dot = Digraph(comment='The Round Table')
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

dot.render('test-output/round-table.gv', view=True)


from collections import Counter
c=Counter(movies_df.title.tolist())
print([k for k, x in c.items() if x>1])


movies_df.loc[190,'title']='The Other Woman (2014)'
movies_df.loc[1906,'title']='Twilight (1998)'
movies_df.loc[874,'title']='Total Recall (2012)'
movies_df.loc[378,'title']='Trespass (2011)'
movies_df.loc[1691,'title']='The Rookie (1990)'
movies_df.loc[1776,'title']='Fantastic Four (2005)'
movies_df.loc[574,'title']='Fantastic Four (2014)'
movies_df.loc[1000,'title']='Great Expectations (2013)'


# stats.percentileofscore(mojo_df['ROI'], -0.35)

tt = mojo_df[['title', 'Gross', 'budget', 'ROI']]
tt.sort_values('ROI', ascending=False, inplace=True)
tt.iloc[-20:, :]


max(list(test.values()))
[k for k,v in test.items() if v==8]
8
[('Tim Burton', 'Johnny Depp'),
 ('Dennis Dugan', 'Adam Sandler'),
 ('Robert Rodriguez', 'Antonio Banderas')]

>10
('Ron Howard', 'Brian Grazer')
('Joel Coen', 'Ethan Coen')
('Clint Eastwood', 'Robert Lorenz')
('The Farrelly Bros.', 'Bradley Thomas')


12
('Ethan Coen', 'Joel Coen')
8
[('Emma Thomas', 'Christopher Nolan'),
 ('Peter Jackson', 'Fran Walsh'),
 ('Fran Walsh', 'Philippa Boyens'),
 ('Sam Mercer', 'M. Night Shyamalan')]
 ('Peter Jackson', 'Philippa Boyens')

from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_


def budget_bin(x):
    if x < 3e7:
        return 'low'
    if x < 1e8:
        return 'mid'
    else:
        return 'high'

ROI_df = mojo_df[['budget', 'ROI']]
ROI_df['bin'] = ROI_df['budget'].apply(budget_bin)



# ROI_list = ROI_df.values.tolist()

top_ROI = ROI_df[ROI_df['ROI']<-0.35]
top_ROI = ROI_df
# _, ax = plt.subplots()
g = sns.swarmplot(x='budget', y='ROI', data=top_ROI, hue='bin')
# ax.set_xticklables([])
# g.axes[0][0].xaxix.set_visible(False)
# plt.setp(g.setxticklables([]) )
sns.regplot(x='budget', y='ROI_scaled', data=data_df)

cutoff = 3e7

big_ROI = ROI_df[ROI_df['budget']>1e8]
_, ax = plt.subplots()
sns.regplot(x='budget', y='ROI', data=big_ROI)

mid_ROI = ROI_df[(ROI_df['budget']>cutoff) & (ROI_df['budget']<1e8)]
_, ax = plt.subplots()
sns.regplot(x='budget', y='ROI', data=mid_ROI)

small_ROI = ROI_df[ROI_df['budget']<cutoff]
_, ax = plt.subplots()
sns.regplot(x='budget', y='ROI', data=small_ROI)

###### box office bomb
bottom_ROI = ROI_df[ROI_df['ROI']<-0.35]
_, ax = plt.subplots()
sns.regplot(x='budget', y='ROI', data=bottom_ROI)
ax.set_xlim(0,3e8)
ax.set_ylim(-1, -0.3)

# max(list(dp_dict.values()))
# [k for k,v in dp_dict.items() if v==15]