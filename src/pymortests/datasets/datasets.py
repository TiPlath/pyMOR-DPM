# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from pymor.parameters.base import Parameters
from pymor.datasets.base import Dataset
from pymor.vectorarrays.numpy import NumpyVectorSpace


@pytest.fixture
def mock_parameters():
    return Parameters({'a': 1, 'b': 2}).parse([0.1, 0.2, 0.3])


@pytest.fixture
def mock_solution_array():
    space = NumpyVectorSpace(2)
    return space.from_numpy(np.array([[1., 2.], [3., 4.]]))


@pytest.fixture
def mock_output():
    return np.array([42.0])


@pytest.fixture
def dataset(mock_parameters, mock_solution_array, mock_output):
    return Dataset([
        {'parameters': mock_parameters, 'solution': mock_solution_array, 'output': mock_output},
        {'parameters': mock_parameters, 'solution': None, 'output': None}
    ])


def test_len(dataset):
    assert len(dataset) == 2


def test_getitem(dataset):
    entry = dataset[0]
    assert isinstance(entry, Dataset)
    assert len(entry) == 1


def test_slicing(dataset):
    sliced = dataset[:]
    assert isinstance(sliced, Dataset)
    assert len(sliced) == 2


def test_iter(dataset):
    entries = list(dataset)
    assert len(entries) == 2
    for entry in entries:
        assert 'parameters' in entry
        assert 'solution' in entry
        assert 'output' in entry


def test_to_numpy(dataset):
    data = dataset.to_numpy()
    assert set(data.keys()) == {'parameters', 'solutions', 'outputs'}
    assert len(data['parameters']) == 2
    assert data['parameters'][0].shape == (3,)
    assert data['solutions'][0] is not None
    assert data['solutions'][1] is None
    assert data['outputs'][0] is not None
    assert data['outputs'][1] is None


def test_with_entries(dataset, mock_parameters, mock_solution_array, mock_output):
    new_entry = {'parameters': mock_parameters, 'solution': mock_solution_array, 'output': mock_output}
    new_ds = dataset.with_entries(new_entry)
    assert isinstance(new_ds, Dataset)
    assert len(new_ds) == 3
    assert len(dataset) == 2


def test_filter(dataset):
    result = dataset.filter(lambda e: e['output'] is not None and e['output'][0] > 40)
    assert len(result) == 1

    result = dataset.filter(lambda e: e['output'] is not None and e['output'][0] > 100)
    assert len(result) == 0


def test_from_components_full(mock_parameters, mock_solution_array, mock_output):
    parameters = [mock_parameters, mock_parameters]
    solutions = [mock_solution_array, mock_solution_array]
    outputs = [mock_output, mock_output]

    dataset = Dataset.from_components(parameters, solutions, outputs)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    for entry in dataset:
        assert entry['parameters'] is not None
        assert entry['solution'] is not None
        assert entry['output'] is not None


def test_from_components_missing_solution(mock_parameters, mock_output):
    parameters = [mock_parameters, mock_parameters]
    outputs = [mock_output, mock_output]

    dataset = Dataset.from_components(parameters, solutions=None, outputs=outputs)
    assert isinstance(dataset, Dataset)
    for entry in dataset:
        assert entry['parameters'] is not None
        assert entry['solution'] is None
        assert entry['output'] is not None


def test_from_components_missing_output(mock_parameters, mock_solution_array):
    parameters = [mock_parameters, mock_parameters]
    solutions = [mock_solution_array, mock_solution_array]

    dataset = Dataset.from_components(parameters, solutions=solutions, outputs=None)
    assert isinstance(dataset, Dataset)
    for entry in dataset:
        assert entry['parameters'] is not None
        assert entry['solution'] is not None
        assert entry['output'] is None


def test_from_components_only_parameters(mock_parameters):
    parameters = [mock_parameters, mock_parameters]

    dataset = Dataset.from_components(parameters)
    assert isinstance(dataset, Dataset)
    for entry in dataset:
        assert entry['parameters'] is not None
        assert entry['solution'] is None
        assert entry['output'] is None


def test_from_components_length_mismatch(mock_parameters, mock_solution_array, mock_output):
    parameters = [mock_parameters, mock_parameters]
    solutions = [mock_solution_array]
    outputs = [mock_output, mock_output]

    with pytest.raises(ValueError):
        Dataset.from_components(parameters, solutions, outputs)

    with pytest.raises(ValueError):
        Dataset.from_components(parameters, solutions=None, outputs=[mock_output])