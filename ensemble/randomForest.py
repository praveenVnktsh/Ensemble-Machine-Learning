from re import S
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini_index', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        dic = {
            'gini_index' : 'gini',
            'information_gain' : 'entropy'
        }
        self.estimators = [DecisionTreeClassifier(criterion=dic[criterion], max_depth=max_depth) for _ in range(n_estimators)]

        self.nEstimators = n_estimators
        self.data = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.trainX = X
        self.trainY = y

        m = int(X.shape[1] ** 0.5)
        for i, estimator in enumerate(self.estimators):
            newX = X.sample(n = m, replace = True, axis = 1, random_state = i)
            newY = y
            estimator.fit(newX,newY)

            self.data.append((newX.to_numpy(), newY.to_numpy(), newX.columns))

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)
        for i, estimator in enumerate(self.estimators):
            yNew = estimator.predict(X[self.data[i][2]])
            y.append(yNew)


        y = pd.DataFrame(y).T
        y = y.mode(axis = 1)[0]
        return y
    def plotOneEstimator(self, ax, predict, xx, yy, data):
        
        X, y, columns = data
        if len(columns) != 2:
            if columns[0] == 0:
                Z = predict(xx.ravel().reshape(-1, 1))
            else:
                Z = predict(yy.ravel().reshape(-1, 1))

        else:
            inp = np.c_[xx.ravel(), yy.ravel()]
            Z = predict(inp)

        if type(Z) != np.ndarray:
            Z = Z.to_numpy()
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap= plt.cm.RdBu, alpha=0.8)

        
        ax.scatter(
            X[:, 0], X[:, 1], c=y,cmap=ListedColormap(["#FF0000", "#0000FF"]), edgecolors="k"
        )
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())



       
    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        h = 0.02
        X_train = self.trainX.to_numpy()
        y_train = self.trainY.to_numpy()
        
        
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        fig1, ax = plt.subplots(figsize = (50, 50))
        for i in range(self.nEstimators):
            ax = plt.subplot(1, self.nEstimators, i+1)
            plot_tree(self.estimators[i], ax = ax)

        if self.trainX.columns.shape[0] == 2:
            fig2, ax = plt.subplots(figsize=(20, 5))
            for i in range(self.nEstimators):
                ax = plt.subplot(1, self.nEstimators, i+1)
                plt.title(f'Estimator {i}')
                self.plotOneEstimator(ax, self.estimators[i].predict, xx, yy, (X_train, y_train, self.data[i][2]))

            fig3, ax = plt.subplots()
            plt.title(f'Final Estimator')
            self.plotOneEstimator(ax, self.predict, xx, yy, (X_train, y_train, pd.DataFrame(X_train).columns))

            return [fig1, fig2, fig3]
        else:
            return [fig1]



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.estimators = [DecisionTreeRegressor(max_depth=max_depth) for _ in range(n_estimators)]

        self.nEstimators = n_estimators
        self.data = []

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.trainX = X
        self.trainY = y

        m = int(X.shape[1] /2)
        for estimator in self.estimators:
            newX = X.sample(n = m, replace = True, axis = 1)
            newY = y[newX.index]
            estimator.fit(newX,newY)

            self.data.append((newX.to_numpy(), newY.to_numpy(), newX.columns))

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        for i, estimator in enumerate(self.estimators):
            yNew = estimator.predict(X[self.data[i][2]])
            y.append(yNew)


        y = pd.DataFrame(y).T
        y = y.mean(axis = 1)
        return y

    def plotOneEstimator(self, ax, predict, xx, yy, X, y,):

        Z = predict(np.c_[xx.ravel(), yy.ravel()])
        if type(Z) != np.ndarray:
            Z = Z.to_numpy()
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap= plt.cm.RdBu, alpha=0.8)

        # Plot the training points
        
        ax.scatter(
            X[:, 0], X[:, 1], c=y,cmap=ListedColormap(["#FF0000", "#0000FF"]), edgecolors="k"
        )
        # # Plot the testing points
        # ax.scatter(
        #     X_test[:, 0],
        #     X_test[:, 1],
        #     c=y_test,
        #     cmap=cm_bright,
        #     edgecolors="k",
        #     alpha=0.6,
        # )
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        
    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        h = 0.02
        X_train = self.trainX.to_numpy()
        y_train = self.trainY.to_numpy()
        
        
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        fig1, ax = plt.subplots(figsize=(20, 5))
        for i in range(self.nEstimators):
            ax = plt.subplot(1, self.nEstimators, i+1)
            plot_tree(self.estimators[i], ax = ax)

        if self.trainX.columns.shape[0] == 2:
            fig2, ax = plt.subplots(figsize=(20, 5))
            for i in range(self.nEstimators):
                ax = plt.subplot(1, self.nEstimators, i+1)
                plt.title(f'Estimator {i}')
                self.plotOneEstimator(ax, self.estimators[i].predict, xx, yy, self.data)

            fig3, ax = plt.subplots()
            plt.title(f'Final Estimator')
            self.plotOneEstimator(ax, self.predict, xx, yy, X_train, y_train)

    
            return [fig1, fig2, fig3]
        else:
            return [fig1]
