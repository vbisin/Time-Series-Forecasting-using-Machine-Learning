import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold, covariance, cluster
import numpy as np
import pickle

## Choose stock Market 
AMEXdict=pickle.load(open("AMEXdictRanked.p","rb"))
AMEXquotes=pickle.load(open("AMEXquotesRanked.p","rb"))
#NASDAQdict=pickle.load(open("NASDAQdictRanked.p","rb"))
#NASDAQquotes=pickle.load(open("NASDAQquotesRanked.p","rb"))
#NYSEdict=pickle.load(open("NYSEdictRanked.p","rb"))
#NYSEquotes=pickle.load(open("NYSEquotesRanked.p","rb"))

featureMatrix=pickle.load(open("AMEXweight.p","rb"))
#featureMatrix=pickle.load(open("NYSEweight.p","rb"))
#featureMatrix=pickle.load(open("NASDAQweight.p","rb"))

Stockdict=AMEXdict
Stockquotes=AMEXquotes


##Get array with names of stocks
val = Stockdict.values()
names = list()
for i in range(len(val)):
   names.append(val[i][0])  
names=np.asarray(names)





edge_model = covariance.GraphLassoCV()

X = featureMatrix.copy()
X /= X.std(axis=0)
edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

#for i in range(n_labels + 1):
 #   print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

    


#########################################


node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='auto', n_neighbors=320)

embedding = node_position_model.fit_transform(featureMatrix.T).T

plt.figure(1, facecolor='w', figsize=(35, 45))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.002)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.savefig('AMEXcluster.pdf')
plt.show()
