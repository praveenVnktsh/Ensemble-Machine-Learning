import pandas as pd 
def accuracy(y_hat : pd.Series, y : pd.Series):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    assert(y_hat.size == y.size)
    return sum(y_hat == y)/y.size


def precision(y_hat : pd.Series, y : pd.Series, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)

    truePos = len(y_hat[(y_hat == cls) & (y == cls)])
    falsePos = len(y_hat[(y_hat == cls) & (y != cls)])

    return truePos/(truePos + falsePos)


def recall(y_hat : pd.Series, y : pd.Series, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)

    truePos = len(y_hat[(y_hat == cls) & (y == cls)])
    falseNeg = len(y_hat[(y_hat != cls) & (y == cls)])

    return truePos/(truePos + falseNeg)

def rmse(y_hat : pd.Series, y : pd.Series):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    assert(y_hat.size == y.size)

    return pow((y_hat - y).pow(2).sum()/y.size, 0.5)

def mae(y_hat : pd.Series, y : pd.Series):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)

    return sum(abs(y_hat - y))/y.size
