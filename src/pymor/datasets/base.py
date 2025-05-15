# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import ImmutableObject

class Dataset(ImmutableObject):
    """
    A container class for externally generated simulation or experimental data.

    This class is immutable and designed to store a collection of data entries, each consisting of:
      - a `parameters` object (must be a pyMOR `Parameters` instance),
      - a `solution` object (can be a VectorArray or any other structure),
      - an `output` object (usually an array or scalar).

    Attributes
    ----------
    entries : tuple
        An immutable sequence of data entries. Each entry is a dictionary
        with keys 'parameters', 'solution', and 'output'.
        Any value may be None to indicate missing data.
    """

    def __init__(self, entries):
        """
        Initialize the dataset with a list of data entries.

        Parameters
        ----------
        entries : list of dict
            Each dict must contain the keys:
              - 'parameters' (pyMOR Parameters instance),
              - 'solution' (VectorArray or compatible object),
              - 'output' (scalar or array-like).
        """
        required_keys = {'parameters', 'solution', 'output'}
        for i, entry in enumerate(entries):
            if not required_keys.issubset(entry):
                raise ValueError(f"Dataset entry {i} is missing required keys: {required_keys - entry.keys()}")
        self.entries = tuple(entries)
        super().__init__()

    def __len__(self):
        """Return the number of entries in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Return a sliced or indexed subset of the dataset.

        Parameters
        ----------
        idx : int, slice, or list of int
            Index or slice to retrieve a subset of the dataset.

        Returns
        -------
        Dataset
            A new Dataset instance containing the selected entries.
        """
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(len(self.entries)))[idx]
        return Dataset([self.entries[i] for i in idx])

    def __iter__(self):
        """Return an iterator over the dataset entries."""
        return iter(self.entries)

    @classmethod
    def from_components(cls, parameters, solutions=None, outputs=None):
        """
        Construct a Dataset from lists of parameters, solutions, and outputs.

        Parameters
        ----------
        parameters : list
            List of pyMOR Parameters instances.
        solutions : list, optional
            List of solution objects (e.g., VectorArrays). Can be None.
        outputs : list, optional
            List of outputs (scalars or array-like). Can be None.

        Returns
        -------
        Dataset
            A new Dataset instance constructed from the provided components.

        Raises
        ------
        ValueError
            If the lengths of the input lists do not match.
        """
        n = len(parameters)
        if solutions is not None and len(solutions) != n:
            raise ValueError("Length of solutions does not match length of parameters.")
        if outputs is not None and len(outputs) != n:
            raise ValueError("Length of outputs does not match length of parameters.")

        entries = []
        for i in range(n):
            entry = {
                'parameters': parameters[i],
                'solution': solutions[i] if solutions is not None else None,
                'output': outputs[i] if outputs is not None else None
            }
            entries.append(entry)
        return cls(entries)

    def to_numpy(self):
        """
        Convert the dataset to a dictionary of NumPy arrays.
        Values may be `None` if not available.

        Assumes that:
          - each 'parameters' entry has a `.to_numpy()` method,
          - each 'solution' either has `.to_numpy()`, is array-like, or None,
          - each 'output' is array-like or None.

        Returns
        -------
        dict
            A dictionary with keys:
              - 'parameters': list of NumPy arrays,
              - 'solutions': list of NumPy arrays or None,
              - 'outputs': list of NumPy arrays or None.
        """
        parameters, solutions, outputs = [], [], []

        for entry in self.entries:
            p = entry['parameters'].to_numpy()
            s = entry['solution'].to_numpy() if entry['solution'] is not None else None
            o = np.asarray(entry['output']) if entry['output'] is not None else None

            parameters.append(p)
            solutions.append(s)
            outputs.append(o)

        return {
            'parameters': parameters,
            'solutions': solutions,
            'outputs': outputs
        }

    def with_entries(self, entries):
        """
        Return a new Dataset with one or more entries appended.

        Parameters
        ----------
        entries : dict or list of dict
            One or more new data entries, each with keys 'parameters', 'solution', and 'output'.

        Returns
        -------
        Dataset
            A new Dataset instance with the added entry or entries.
        """
        if isinstance(entries, dict):
            entries = [entries]
        return Dataset(list(self.entries) + entries)

    def filter(self, assertion):
        """
        Return a new Dataset containing only entries for which assertion(entry) is True.

        Parameters
        ----------
        assertion : callable
            A function that takes a dataset entry and returns True or False.

        Returns
        -------
        Dataset
            A new Dataset with only the filtered entries.
        """
        return Dataset([e for e in self.entries if assertion(e)])
