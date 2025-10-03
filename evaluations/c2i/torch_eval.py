import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple
import io
import numpy as np
from scipy import linalg
import warnings

import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import torch.nn as nn
import torchvision
import requests

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501
INCEPTION_V3_PATH = "pt_inception-2015-12-05-6726825d.pth"

base_path = os.path.abspath(
    os.path.join(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        ),
        "../../",
    )
)


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """

    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def numpy_partition(arr, kth, **kwargs):
    """Parallel numpy partition for faster k-nearest neighbor computation."""
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


def torch_partition(arr_tensor, kth, dim=-1):
    """Torch-based partition using topk for k-nearest neighbor computation."""
    # Use topk to get the k-th smallest elements efficiently on GPU
    # topk returns largest by default, so we need to handle this carefully
    if dim == -1:
        dim = arr_tensor.dim() - 1

    k = max(kth) + 1 if isinstance(kth, (list, tuple, np.ndarray)) else kth + 1

    # Get the k smallest values using topk with largest=False
    values, _ = torch.topk(arr_tensor, k, dim=dim, largest=False, sorted=True)

    return values


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return INCEPTION_V3_PATH
    print("downloading InceptionV3 model...")
    with requests.get(FID_WEIGHTS_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)
    return INCEPTION_V3_PATH


class DistanceBlock:
    """
    Calculate pairwise distances between vectors using PyTorch.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self):
        self.device = torch.device("cuda")

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.
        """
        U_tensor = torch.from_numpy(U.astype(np.float32)).to(self.device)
        V_tensor = torch.from_numpy(V.astype(np.float32)).to(self.device)

        with torch.no_grad():
            distances = self._batch_pairwise_distances(U_tensor, V_tensor)
            return distances.cpu().numpy()

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        """Compute which points are within radii of each other."""
        batch_1_tensor = torch.from_numpy(batch_1.astype(np.float32)).to(self.device)
        batch_2_tensor = torch.from_numpy(batch_2.astype(np.float32)).to(self.device)
        radii_1_tensor = torch.from_numpy(radii_1.astype(np.float32)).to(self.device)
        radii_2_tensor = torch.from_numpy(radii_2.astype(np.float32)).to(self.device)

        with torch.no_grad():
            distances = self._batch_pairwise_distances(batch_1_tensor, batch_2_tensor)

            # Add dimension for broadcasting
            dist_expanded = distances.unsqueeze(-1)

            # Check which distances are within radii
            batch_1_in = torch.any(dist_expanded <= radii_2_tensor, dim=1)
            batch_2_in = torch.any(dist_expanded <= radii_1_tensor.unsqueeze(1), dim=0)

            return batch_1_in.cpu().numpy(), batch_2_in.cpu().numpy()

    def _batch_pairwise_distances(self, U, V):
        """
        Compute pairwise distances between two batches of feature vectors.
        """
        # Squared norms of each row in U and V.
        norm_u = torch.sum(U**2, dim=1)
        norm_v = torch.sum(V**2, dim=1)

        # norm_u as a column and norm_v as a row vectors.
        norm_u = norm_u.reshape(-1, 1)
        norm_v = norm_v.reshape(1, -1)

        # Pairwise squared Euclidean distances.
        D = torch.clamp(norm_u - 2 * torch.matmul(U, V.t()) + norm_v, min=0.0)

        return D

    def pairwise_distances_torch(self, U_tensor, V_tensor):
        """
        Torch-native version that takes and returns torch tensors on CUDA.
        """
        with torch.no_grad():
            return self._batch_pairwise_distances(U_tensor, V_tensor)

    def less_thans_torch(
        self, batch_1_tensor, radii_1_tensor, batch_2_tensor, radii_2_tensor
    ):
        """
        Torch-native version that takes and returns torch tensors on CUDA.
        """
        with torch.no_grad():
            distances = self._batch_pairwise_distances(batch_1_tensor, batch_2_tensor)

            # Add dimension for broadcasting
            dist_expanded = distances.unsqueeze(-1)

            # Check which distances are within radii
            batch_1_in = torch.any(dist_expanded <= radii_2_tensor, dim=1)
            batch_2_in = torch.any(dist_expanded <= radii_1_tensor.unsqueeze(1), dim=0)

            return batch_1_in, batch_2_in


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param device: torch device for computations
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        self.device = torch.device("cuda")
        self.distance_block = DistanceBlock()
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        """Compute manifold radii using k-nearest neighbors."""
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0 : end1 - begin1, begin2:end2] = (
                    self.distance_block.pairwise_distances(row_batch, col_batch)
                )

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in numpy_partition(
                        distance_batch[0 : end1 - begin1, :], seq, axis=1
                    )
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def manifold_radii_torch(self, features: np.ndarray) -> np.ndarray:
        """Optimized version using torch operations to minimize CPU/GPU transfers."""
        num_images = len(features)

        # Convert features to torch tensor once
        features_tensor = torch.from_numpy(features.astype(np.float32)).to(self.device)

        # Pre-allocate result tensors on GPU
        radii_tensor = torch.zeros(
            [num_images, self.num_nhoods], dtype=torch.float32, device=self.device
        )
        seq_tensor = torch.arange(
            max(self.nhood_sizes) + 1, dtype=torch.int32, device=self.device
        )

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch_tensor = features_tensor[begin1:end1]

            # Pre-allocate distance batch on GPU
            distance_batch_tensor = torch.zeros(
                [end1 - begin1, num_images], dtype=torch.float32, device=self.device
            )

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch_tensor = features_tensor[begin2:end2]

                # Compute distances between batches using torch-native method
                distances = self.distance_block.pairwise_distances_torch(
                    row_batch_tensor, col_batch_tensor
                )
                distance_batch_tensor[:, begin2:end2] = distances

            # Use torch topk instead of numpy_partition for k-nearest neighbors
            # Get the k+1 smallest distances (including self-distance)
            k_vals = torch.topk(
                distance_batch_tensor,
                max(self.nhood_sizes) + 1,
                dim=1,
                largest=False,
                sorted=True,
            )[0]

            # Extract the required neighborhood sizes
            for i, nhood_size in enumerate(self.nhood_sizes):
                radii_tensor[begin1:end1, i] = k_vals[:, nhood_size]

        # Apply percentile clamping if specified
        if self.clamp_to_percentile is not None:
            max_distances = torch.quantile(
                radii_tensor, self.clamp_to_percentile / 100.0, dim=0
            )
            radii_tensor = torch.where(
                radii_tensor <= max_distances,
                radii_tensor,
                torch.zeros_like(radii_tensor),
            )

        # Convert back to numpy only at the end
        return radii_tensor.cpu().numpy()

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )

    def evaluate_pr_torch(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized version using torch operations to minimize CPU/GPU transfers.
        """
        # Convert all inputs to torch tensors once
        features_1_tensor = torch.from_numpy(features_1.astype(np.float32)).to(
            self.device
        )
        features_2_tensor = torch.from_numpy(features_2.astype(np.float32)).to(
            self.device
        )
        radii_1_tensor = torch.from_numpy(radii_1.astype(np.float32)).to(self.device)
        radii_2_tensor = torch.from_numpy(radii_2.astype(np.float32)).to(self.device)

        # Pre-allocate result tensors on GPU
        features_1_status = torch.zeros(
            [len(features_1), radii_2.shape[1]], dtype=torch.bool, device=self.device
        )
        features_2_status = torch.zeros(
            [len(features_2), radii_1.shape[1]], dtype=torch.bool, device=self.device
        )

        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1_tensor = features_1_tensor[begin_1:end_1]

            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2_tensor = features_2_tensor[begin_2:end_2]

                # Use torch-native less_thans method
                batch_1_in, batch_2_in = self.distance_block.less_thans_torch(
                    batch_1_tensor,
                    radii_1_tensor[begin_1:end_1],
                    batch_2_tensor,
                    radii_2_tensor[begin_2:end_2],
                )

                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in

        # Compute means on GPU and convert to numpy only at the end
        precision = torch.mean(features_2_status.float(), dim=0).cpu().numpy()
        recall = torch.mean(features_1_status.float(), dim=0).cpu().numpy()

        return (precision, recall)


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    # Skips default weight inititialization if supported by torchvision
    # version. See https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs["init_weights"] = False

    # Backwards compatibility: `weights` argument was handled by `pretrained`
    # argument prior to version 0.13.
    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                "weights=={} not supported in torchvision {}".format(
                    kwargs["weights"], torchvision.__version__
                )
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    inception.load_state_dict(torch.load(_download_inception_model()))
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights="DEFAULT")

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward_nopreprocess(self, inp):
        outp = []
        x = inp

        outp.append(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

    def forward(self, inp):
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        outp.append(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


class Evaluator:
    def __init__(
        self,
        batch_size=64,
        softmax_batch_size=512,
    ):
        self.device = torch.device("cuda")
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator()

        # Initialize InceptionV3 model for pool features (block 3)
        self.inception_model = InceptionV3(
            output_blocks=[3],  # Block 3 for pool features
            resize_input=True,
            normalize_input=True,
            requires_grad=False,
            use_fid_inception=True,
        ).to(self.device)
        self.inception_model.eval()

        self.full_model = fid_inception_v3()
        # Create a separate model for extracting spatial features from Mixed_6e
        self.spatial_model = self._create_spatial_model(self.full_model).to(self.device)

        # Create a separate model for softmax (we need the weights from the final layer)
        self.softmax_model = self._create_softmax_model(self.full_model).to(self.device)

    def _create_spatial_model(self, inception_model):
        """Create a model for extracting spatial features from Mixed_6e.branch1x1 layer."""

        # Create a model that extracts features up to Mixed_6e.branch1x1
        # This corresponds to TF's mixed_6/conv:0 (the 1x1 conv branch)
        class SpatialFeatureExtractor(nn.Module):
            def __init__(self, inception_model):
                super().__init__()
                self.Conv2d_1a_3x3 = inception_model.Conv2d_1a_3x3
                self.Conv2d_2a_3x3 = inception_model.Conv2d_2a_3x3
                self.Conv2d_2b_3x3 = inception_model.Conv2d_2b_3x3
                self.Conv2d_3b_1x1 = inception_model.Conv2d_3b_1x1
                self.Conv2d_4a_3x3 = inception_model.Conv2d_4a_3x3
                self.Mixed_5b = inception_model.Mixed_5b
                self.Mixed_5c = inception_model.Mixed_5c
                self.Mixed_5d = inception_model.Mixed_5d
                self.Mixed_6a = inception_model.Mixed_6a
                self.Mixed_6b = inception_model.Mixed_6b
                self.Mixed_6c = inception_model.Mixed_6c
                self.Mixed_6d = inception_model.Mixed_6d

            def forward(self, x):
                x = F.interpolate(
                    x, size=(299, 299), mode="bilinear", align_corners=False
                )
                x = 2 * x - 1
                # Forward through all layers up to Mixed_6e
                x = self.Conv2d_1a_3x3(x)
                x = self.Conv2d_2a_3x3(x)
                x = self.Conv2d_2b_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.Conv2d_3b_1x1(x)
                x = self.Conv2d_4a_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.Mixed_5b(x)
                x = self.Mixed_5c(x)
                x = self.Mixed_5d(x)
                x = self.Mixed_6a(x)
                x = self.Mixed_6b(x)
                x = self.Mixed_6c(x)

                # Extract only the branch1x1 output from Mixed_6e (corresponds to TF mixed_6/conv:0)
                branch1x1 = self.Mixed_6d.branch1x1(x)
                return branch1x1

            def forward_nopreprocess(self, x):
                # Forward through all layers up to Mixed_6e
                x = self.Conv2d_1a_3x3(x)
                x = self.Conv2d_2a_3x3(x)
                x = self.Conv2d_2b_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.Conv2d_3b_1x1(x)
                x = self.Conv2d_4a_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.Mixed_5b(x)
                x = self.Mixed_5c(x)
                x = self.Mixed_5d(x)
                x = self.Mixed_6a(x)
                x = self.Mixed_6b(x)
                x = self.Mixed_6c(x)

                # Extract only the branch1x1 output from Mixed_6d (corresponds to TF mixed_6/conv:0)
                branch1x1 = self.Mixed_6d.branch1x1(x)
                return branch1x1  # .permute(0, 2, 3, 1)  # NHWC

        spatial_model = SpatialFeatureExtractor(inception_model)
        spatial_model = spatial_model.to(self.device)
        spatial_model.eval()
        return spatial_model

    def _create_softmax_model(self, inception_model):
        class SoftmaxModel(nn.Module):
            def __init__(self, inception_model):
                super(SoftmaxModel, self).__init__()
                self.w = nn.Parameter(inception_model.fc.weight.data)

            def forward(self, x):
                logits = torch.matmul(x, self.w.T)
                probs = F.softmax(logits, dim=1)
                return probs

        softmax_model = SoftmaxModel(inception_model)
        softmax_model = softmax_model.to(self.device)
        softmax_model.eval()
        return softmax_model

    def warmup(self):
        """Warmup the model with dummy data."""
        for _ in range(5):
            dummy_data = torch.zeros(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                self.inception_model(dummy_data)

    def debug_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read activations from NPZ file."""
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.debug_features(reader.read_batches(self.batch_size))

    def debug_features(
        self, batches: Iterable[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        pool_preds = []

        self.inception_model.eval()

        with torch.no_grad():
            for batch in tqdm(batches):
                # Convert from NHWC to NCHW and normalize to [0, 1]
                batch_tensor = torch.from_numpy(batch.astype(np.float32)).permute(
                    0, 3, 1, 2
                )
                batch_tensor = batch_tensor.to(self.device)
                # pool_preds.append(batch_tensor.cpu().numpy())
                # Get pool features from Block 3 (final avgpool output)
                pool_features = self.inception_model(batch_tensor)
                for i, feat in enumerate(pool_features):
                    pool_preds.append(feat.cpu().numpy())

        return pool_preds

    def forward_single_batch_no_preprocess(self, batch):
        pool_preds = []

        self.inception_model.eval()

        with torch.no_grad():
            # Convert from NHWC to NCHW
            batch_tensor = torch.from_numpy(batch.astype(np.float32)).permute(
                0, 3, 1, 2
            )
            batch_tensor = batch_tensor.to(self.device)
            # Get pool features from Block 3 (final avgpool output)
            pool_features = self.inception_model.forward_nopreprocess(batch_tensor)
            for i, feat in enumerate(pool_features):
                pool_preds.append(feat.cpu().numpy())

        return pool_preds

    def compute_activations_nopreprocess(
        self, batches: Iterable[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.

        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                dimension. The tuple is (pool_3, spatial).
        """
        pool_preds = []
        spatial_preds = []

        self.inception_model.eval()
        self.spatial_model.eval()

        with torch.no_grad():
            for batch in tqdm(batches):
                # Convert from NHWC to NCHW and normalize to [0, 1]
                batch_tensor = torch.from_numpy(batch).to(
                    self.device, dtype=torch.float32
                )
                batch_tensor = batch_tensor.permute(0, 3, 1, 2)

                # Get pool features from Block 3 (final avgpool output)
                pool_features = self.inception_model.forward_nopreprocess(batch_tensor)[
                    -1
                ]  # Block 3 output
                pool_pred = pool_features.reshape(pool_features.shape[0], -1)

                # Get spatial features from Mixed_6e.branch1x1 output (corresponds to TF mixed_6/conv:0)
                spatial_features = self.spatial_model.forward_nopreprocess(batch_tensor)
                # Take only first 7 channels like TF version: spatial[..., :7]
                spatial_features = spatial_features[:, :7, :, :]
                # Flatten spatial features
                spatial_pred = spatial_features.permute(0, 2, 3, 1).reshape(
                    spatial_features.shape[0], -1
                )

                pool_preds.append(pool_pred)
                spatial_preds.append(spatial_pred)

        return (
            (
                torch.cat(pool_preds, dim=0).cpu().numpy(),
                torch.cat(spatial_preds, dim=0).cpu().numpy(),
            ),
            (pool_features, spatial_features),
        )

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read activations from NPZ file."""
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(
        self, batches: Iterable[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.

        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                 dimension. The tuple is (pool_3, spatial).
        """
        pool_preds = []
        spatial_preds = []

        self.inception_model.eval()
        self.spatial_model.eval()

        with torch.no_grad():
            for batch in tqdm(batches):
                # Convert from NHWC to NCHW and normalize to [0, 1]
                batch_tensor = torch.from_numpy(batch).to(
                    self.device, dtype=torch.float32
                )
                batch_tensor = batch_tensor.permute(0, 3, 1, 2) / 255

                # Get pool features from Block 3 (final avgpool output)
                pool_features = self.inception_model(batch_tensor)[-1]  # Block 3 output
                pool_pred = pool_features.reshape(pool_features.shape[0], -1)

                # Get spatial features
                spatial_features = self.spatial_model(batch_tensor)
                # Take only first 7 channels like TF version: spatial[..., :7]
                spatial_features = spatial_features[:, :7, :, :]
                # Flatten spatial features
                spatial_pred = spatial_features.permute(0, 2, 3, 1).reshape(
                    spatial_features.shape[0], -1
                )

                pool_preds.append(pool_pred)
                spatial_preds.append(spatial_pred)

        return (
            torch.cat(pool_preds, dim=0).cpu().numpy(),
            torch.cat(spatial_preds, dim=0).cpu().numpy(),
        )

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        """Read or compute statistics from activations."""
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
                obj["mu_s"], obj["sigma_s"]
            )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        # """Compute mean and covariance statistics."""
        # mu = np.mean(activations, axis=0)
        # sigma = np.cov(activations, rowvar=False)
        # return FIDStatistics(mu, sigma)
        device = torch.device("cuda")

        activations_torch = torch.from_numpy(activations).to(
            device, dtype=torch.float32
        )

        mu_torch = torch.mean(activations_torch, dim=0)
        sigma_torch = torch.cov(
            activations_torch.T
        )  # torch.cov expects [D, N] so we transpose

        mu = mu_torch.cpu().numpy()
        sigma = sigma_torch.cpu().numpy()

        return FIDStatistics(mu, sigma)

    def compute_inception_score(
        self, activations: np.ndarray, split_size: int = 5000
    ) -> float:
        """Compute Inception Score from pool features."""
        softmax_out = []

        self.softmax_model.eval()
        with torch.no_grad():
            for i in range(0, len(activations), self.softmax_batch_size):
                acts = activations[i : i + self.softmax_batch_size]
                acts_tensor = torch.from_numpy(acts).to(self.device)
                softmax_out.append(self.softmax_model(acts_tensor).cpu().numpy())

        preds = np.concatenate(softmax_out, axis=0)

        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_prec_recall(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        """Compute precision and recall metrics."""
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))

    def compute_prec_recall_torch(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        """Optimized version using torch operations to minimize CPU/GPU transfers."""
        radii_1 = self.manifold_estimator.manifold_radii_torch(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii_torch(activations_sample)
        pr = self.manifold_estimator.evaluate_pr_torch(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))


def evaluate(
    sample_batch,
    ref_batch=os.path.join(
        base_path, "dataset/ImageNet-1k/reference/VIRTUAL_imagenet256_labeled.npz"
    ),
):
    evaluator = Evaluator()

    print("warming up PyTorch...")
    evaluator.warmup()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(ref_batch)
    print("computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch, ref_acts)

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(
        sample_batch, sample_acts
    )

    print("Computing IS...")
    IS = evaluator.compute_inception_score(sample_acts[0])
    print("Computing FID...")
    FID = sample_stats.frechet_distance(ref_stats)
    print("Computing sFID...")
    sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    print("Computing Precision and Recall...")
    prec, recall = evaluator.compute_prec_recall_torch(ref_acts[0], sample_acts[0])

    result = {
        "Inception Score": IS,
        "FID": float(FID),
        "sFID": float(sFID),
        "Precision": prec,
        "Recall": recall,
    }
    del evaluator
    return result
