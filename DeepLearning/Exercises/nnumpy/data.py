import gzip
import io
import os
import tempfile
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import numpy as np

DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".nnumpy")


class CachedDownload:
    """
    State of a possibly already cached download
    from a file at some URL to the local filesystem.
    """

    def __init__(self, base_url, file_name, base_path=None,
                 overwrite=False, block_size=4096):
        """
        Set up the cached download.

        Parameters
        ----------
        base_url : str
            URL that points to the directory where the file is located.
        file_name : str
            Name of the file that is to be downloaded.
        base_path : str, optional
            Path to the location where the downloaded file should be stored.
            If not specified, the file is stored in a temporary directory.
        overwrite : bool, optional
            Whether or not an existing local file should be overwritten.
        block_size : int, optional
            Number of bytes to read at once.
        """
        if base_path is None:
            base_path = tempfile.gettempdir()
        elif not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        self.url = '/'.join([base_url.rstrip('/'), file_name])
        self.file = os.path.join(base_path, file_name)
        self.block_size = block_size
        self.overwrite = overwrite

        try:
            self._response = urlopen(self.url)
            self._response.close()
        except HTTPError:
            raise ValueError("wrong URL? {}".format(self.url))
        except URLError:
            raise RuntimeError("could not connect to URL: {}".format(self.url))

    @property
    def file_name(self):
        """ Name of the file that is downloaded. """
        return os.path.basename(self.file)

    def _download_file(self):
        """ Download to file. """
        with open(self.file, 'wb') as fp:
            chunk = self._response.read(self.block_size)
            while chunk:
                fp.write(chunk)
                yield chunk
                chunk = self._response.read(self.block_size)

    def _read_file(self):
        """ Read from existing file. """
        with open(self.file, 'rb') as fp:
            chunk = fp.read(self.block_size)
            while chunk:
                yield chunk
                chunk = fp.read(self.block_size)

    def __enter__(self):
        self._response = urlopen(self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._response.close()

    def __iter__(self):
        if self.overwrite or not os.path.exists(self.file):
            return self._download_file()
        else:
            return self._read_file()

    def __len__(self):
        content_length = int(self._response.getheader('Content-Length', 0))
        return 1 + (content_length - 1) // self.block_size


MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_RAW = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def get_mnist_data(path=None, test=False):
    """
    Load the MNIST dataset.

    Parameters
    ----------
    path : str, optional
        Path to directory where the dataset will be stored.
    test : bool, optional
        Flag to return test set instead of training data.

    Returns
    -------
    x : ndarray
        The input features in the data.
    y : ndarray
        The output labels in the data.
    """
    if path is None:
        path = os.path.join(DEFAULT_PATH, "mnist")

    arrays = []
    for file in MNIST_RAW:
        with CachedDownload(MNIST_URL, file, path) as chunks:
            data = gzip.decompress(b''.join(chunks))
            arr = _parse_idx(data)
            arrays.append(arr)

    if test is False:
        return tuple(arrays[:2])
    elif test is True:
        return tuple(arrays[2:])
    else:
        xs = np.concatenate(arrays[0::2], axis=0)
        ys = np.concatenate(arrays[1::2], axis=0)
        return xs, ys


def _parse_idx(data):
    """ Parse IDX file for vectors and multidimensional arrays. """
    import struct
    stream = io.BytesIO(data)
    zero, type_code, ndim = struct.unpack('HBB', stream.read(4))

    if zero != 0:
        raise ValueError("invalid data format")

    dtype_map = {
        0x08: 'uint8', 0x09: 'int8',
        0x0B: 'int16', 0x0C: 'int32',
        0x0D: 'float32', 0x0E: 'float64'
    }

    if type_code not in dtype_map:
        stream.close()
        raise ValueError("invalid type code: 0x{:02X}".format(type_code))

    dtype = np.dtype(dtype_map[type_code]).newbyteorder('>')
    shape = struct.unpack('>' + ndim * 'I', stream.read(4 * ndim))
    content = np.frombuffer(stream.read(), dtype)
    return content.reshape(shape)


def gen_linear_data(num_samples=100, num_dimensions=2,
                    centre=(0, 0), scale=(1, 1), log_snr=.5, seed=None):
    """
    Generate samples for a linear regression dataset (on a plane).

    Parameters
    ----------
    num_samples : int, optional
        Number of samples to generate.
    num_dimensions : int, optional
        Number of dimensions in the input data.
    centre : (float, float) or (ndarray, float), optional
        Point in space where the centre of the plane should intersect.
        The two points shift the input and output, respectively.
        When a numpy array is provided, it must have `d` entries.
    scale : (float, float) or (ndarray, float), optional
        Factors by which the plane should be scaled.
        The two points scale the input and output, respectively.
        When a numpy array is provided, it must have `d` entries.
    log_snr : float, optional
        Base 10 logarithm of the Signal-to-Noise Ratio for the output.
        This value specifies how noisy the data will be.
    seed : int, optional
        Set this number to get reproducible results.

    Returns
    -------
    x : (num_samples, num_dimensions) ndarray
        Generated input data.
    y : (num_samples, 1) ndarray
        Generated output values.

    """
    # start from uniform samples with unit variance
    rng = np.random.default_rng(seed)
    raw_x = rng.uniform(-3 ** .5, 3 ** .5, size=(num_samples, num_dimensions))
    raw_target = raw_x.mean(axis=-1, keepdims=True)

    # transform data
    shift_x, shift_y = centre
    scale_x, scale_y = scale
    x = np.asarray(scale_x) * raw_x + np.asarray(shift_x)
    target = scale_y * raw_target + shift_y

    # add noise
    noise_var = (scale_y ** 2 + shift_y ** 2) / 10 ** log_snr
    y = target + rng.normal(0, noise_var ** .5, size=(num_samples, 1))

    return x, y


def gen_blob_data(num_samples=(50, 50), num_dimensions=2, std=.5, seed=None):
    """
    Generate samples for a classification dataset (Gaussian blobs).

    Parameters
    ----------
    num_samples : tuple of ints, optional
        Number of samples to generate per blob.
    num_dimensions : int, optional
        Number of dimensions in the input data.
    std : float, optional
        Standard deviation for the additive Gaussian noise.
    seed : int, optional
        Set this number to get reproducable results.

    Returns
    -------
    x : (num_samples, num_dimensions) ndarray
        Generated input data.
    y : (num_samples, 1) ndarray
        Generated output values.

    """
    # start from uniformly sampled blob centres
    num_centres = len(num_samples)
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-3 ** .5, 3 ** .5, size=(num_centres, num_dimensions))
    radii = np.full_like(centres, fill_value=std)

    # generate inputs as Gaussian blobs
    x = np.concatenate(
        [std * rng.normal(0, 1, size=(n, num_dimensions)) + mu
         for n, mu, std in zip(num_samples, centres, radii)]
    )

    # generate corresponding targets
    y = np.concatenate([[i] * n for i, n in enumerate(num_samples)])
    return x, np.expand_dims(y, -1)
