from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from mlflow.models import infer_signature

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
 
    # transform the features (PULocationID and DOLocationID) with a dictvectorizer
    # duration is the target variable

    features = data[['PULocationID', 'DOLocationID']]

    features = features.to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(features)

    y_train = data['duration']

    # train a baseline Linear regression model
    signature = infer_signature(X_train, y_train)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("model intercept:", lr.intercept_)

    return dv, lr, signature


@test
def test_output(dv, model, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert dv is not None, 'The dict vectorizer is undefined'
    assert model is not None, 'The model is undefined'
    assert model.intercept_ is not None, "The model has not be trained"
