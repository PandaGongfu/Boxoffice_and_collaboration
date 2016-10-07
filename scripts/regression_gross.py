from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import grid_search
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz

##############  REGRESSION  ####################


def scaler(df, col):
    df[col] /= df[col].max()

# create OLS ANOVA table


def lm_regress( data_df, y ):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                            data_df, y, test_size=0.2)

    X_train = sm.add_constant(X_train,has_constant='add')
    ols_model = sm.OLS(y_train, X_train).fit()
    return ols_model



############ sklearn grid search

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        data_df, y, test_size=0.2)
models = {}
models['lin_reg'] = linear_model.LinearRegression(fit_intercept=False)
models['lasso'] = linear_model.Lasso(alpha=.02)

for name, model in models.items():
    model.fit(X_train, y_train)
    print('Model: '+name)
    print("Score: " + str(model.score(X_train, y_train)))
    sorted_features = sorted(zip(data_df.columns, model.coef_), key=lambda tup: abs(tup[1]), reverse=True)
    for feature in sorted_features:
        print(feature)


####### Grid Search on Lasso ##########
lasso = linear_model.Lasso()

parameters = {'normalize': (True, False),
              'alpha': np.logspace(-4, -.1, 30)}
grid_searcher = grid_search.GridSearchCV(lasso, parameters)
grid_searcher.fit(boston.data, boston.target)


######## Decision Tree ###############
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        data_df, y, test_size=0)
dtrmodel = tree.DecisionTreeRegressor(max_depth=3)
dtrmodel.fit(X_train, y_train)
export_graphviz(dtrmodel, feature_names=X_train.columns, out_file="mytree.dot", max_depth=3)
with open("mytree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


def ROI_scaler(b):
    if b < 5e7:
        return 3.5-(b/25e6)
    elif b < 1e8:
        return 1.5-(b-5e7)/1e8
    else:
        return 1



####### Create Dummy varaiables #############
cols = ['title', 'budget', 'Gross', 'rel_date']
cols += key_genres
data_df = pd.merge(mojo_final[cols], scores_df, on='title')



# data_df['winter'] = data_df['rel_date'].apply(lambda x: int(x.month in [11, 12]))
# data_df['summer'] = data_df['rel_date'].apply(lambda x: int(x.month in [5,6,7]))
# data_df['dump_release'] = data_df['rel_date'].apply(lambda x: int(x.month in [1,2,3,4,8,9,10]))

# data_df = data_df[(data_df['budget'] >5e7)&(data_df['budget'] <1e8)]
# data_df = data_df[(data_df['ROI'] >1)]

# data_df = data_df[(data_df['budget'] <2e7)]
# data_df = data_df[(data_df['budget'] >5e7)]


# COOPERATION DUMMIES
# data_df['da_0'] = (data_df['da_coop']==0)
# data_df['da_1'] = (data_df['da_coop']>0) & (data_df['da_coop']<=1)
# data_df['da_1+'] = (data_df['da_coop']>1)
#
# data_df['da_0'] = data_df['da_0'].apply(int)
# data_df['da_1'] = data_df['da_1'].apply(int)
# data_df['da_1+'] = data_df['da_1+'].apply(int)
#
# data_df['dp_0'] = (data_df['dp_coop']==0)
# data_df['dp_1'] = (data_df['dp_coop']>0) & (data_df['dp_coop']<=1)
# data_df['dp_1+'] = (data_df['dp_coop']>1)
#
# data_df['dp_0'] = data_df['dp_0'].apply(int)
# data_df['dp_1'] = data_df['dp_1'].apply(int)
# data_df['dp_1+'] = data_df['dp_1+'].apply(int)
#
# data_df['pw_0'] = (data_df['pw_coop']==0)
# data_df['pw_1'] = (data_df['pw_coop']>0) & (data_df['pw_coop']<=1)
# data_df['pw_1+'] = (data_df['pw_coop']>1)
#
# data_df['pw_0'] = data_df['pw_0'].apply(int)
# data_df['pw_1'] = data_df['pw_1'].apply(int)
# data_df['pw_1+'] = data_df['pw_1+'].apply(int)

for col in key_genres:
    data_df['a_'+col] = data_df[col] * data_df['a_exp']
    data_df['d_'+col] = data_df[col] * data_df['d_exp']


y=data_df['Gross']/1e6
# drop_cols = ['title', 'Gross', 'rel_date']
drop_cols = ['title', 'Gross', 'rel_date', 'budget', 'a_score']
data_df.drop(drop_cols, axis=1, inplace=True)
# data_df.drop(['da_coop', 'dp_coop', 'pw_coop'], axis=1, inplace=True)
data_df.drop(['a_exp', 'd_exp'], axis=1, inplace=True)
data_df.drop(key_genres, axis=1, inplace=True)

lm_regress(data_df, y).summary()





from sklearn import grid_search
lasso = linear_model.Lasso()

parameters = {'alpha': np.logspace(-4, -.1, 30)}

grid_searcher = grid_search.GridSearchCV(lasso, parameters)
grid_searcher.fit(data_df, y)
grid_searcher.best_params_


X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        data_df, y, test_size=0.2)
models = {}
models['lin_reg'] = linear_model.LinearRegression(fit_intercept=False)
models['lasso'] = linear_model.Lasso(alpha=.794)

for name, model in models.items():
    model.fit(X_train, y_train)
    print('Model: '+name)
    print("Score: " + str(model.score(X_test, y_test)))
    sorted_features = sorted(zip(data_df.columns, model.coef_), key=lambda tup: abs(tup[1]), reverse=True)
    for feature in sorted_features:
        print(feature)



# data_df['decay']=data_df['budget'].apply(lambda x: math.exp(-x/1e8))
#
# X = mojo_df[['budget']]
#
# Y=mojo_df['Gross']/mojo_df['Gross'].max()
#
# cols = ['title', 'budget', 'Gross', 'rel_date']
# cols = ['title', 'Gross', 'rel_date']
# # cols += key_genres
# X = mojo_final[cols]
#
# X['winter'] = X['rel_date'].apply(lambda x: int(x.month in [11, 12]))
# X['summer'] = X['rel_date'].apply(lambda x: int(x.month in [5,6,7]))
# X['dump_release'] = X['rel_date'].apply(lambda x: int(x.month in [1,2,3,4,8,9,10]))
#
# Y= X['Gross']/X['Gross'].max()
# drop_cols = ['title', 'Gross', 'rel_date']
# X.drop(drop_cols, axis=1, inplace=True)
# def lm_regress( data_df, y ):
#     # scaling_cols = ['budget']
#     # scaling_cols = ['a_score', 'd_score']
#     # [scaler(data_df, col) for col in scaling_cols];
#
#
#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#                                             data_df, y, test_size=0)
#
#     X_train = sm.add_constant(X_train,has_constant='add')
#     ols_model = sm.OLS(y_train, X_train).fit()
#     return ols_model
#
# lm_regress(X, Y).summary()



