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
    
    # create duration column expressed in minutes

    data['duration'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data.duration = data.duration.dt.total_seconds() / 60

    # retain only trips that lasts between 1 minute and an hour

    data = data[(data.duration >= 1) & (data.duration <= 60)]

    # make 'PULocationID' and 'DOLocationID' strings.

    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)

    print("number of records after filtering:", data.shape[0])

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert output['PULocationID'].dtype == 'object', 'The PULocationID column is not a string'
    assert output['DOLocationID'].dtype == 'object', 'The DoLocationID column is not a string'
    assert output.duration.min() >= 1 and output.duration.max() <= 60,\
     'The duration column is not in the correct range'
