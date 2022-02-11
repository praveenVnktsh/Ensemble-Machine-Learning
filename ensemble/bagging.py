from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''

        self.estimators = [base_estimator(criterion = 'entropy') for _ in range(n_estimators)]

        self.data = []
        self.nEstimators = n_estimators
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.trainX = X
        self.trainY = y
        for estimator in self.estimators:
            newX = X.sample(frac=1, replace = True)
            newY = y[newX.index]
            self.data.append((newX.to_numpy(), newY.to_numpy()))
            estimator.fit(newX, newY)

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        for estimator in self.estimators:
            yNew = estimator.predict(X)
            y.append(yNew)


        y = pd.DataFrame(y).T
        y = y.mode(axis = 1)[0]
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
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
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
            plt.title(f'Estimator {i}')
            self.plotOneEstimator(ax, self.estimators[i].predict, xx, yy, self.data[i][0], self.data[i][1])

        fig2, ax = plt.subplots()
        plt.title(f'Final Estimator')
        self.plotOneEstimator(ax, self.predict, xx, yy, X_train, y_train)

    
        return fig1, fig2
