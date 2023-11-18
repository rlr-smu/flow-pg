import torch as th
from dataclasses import dataclass
from typing import List, Tuple, Any

@dataclass
class BaseDataLoader:
    batch_size: int
    batch_count: int
    dim: int
    device: str
    seed: int

    def __post_init__(self):
        self._generator = th.Generator(self.device)
        self._generator.manual_seed(self.seed)

@dataclass
class GaussianDataLoader(BaseDataLoader):
    mu: float = 0
    sigma: float = 1
    random_seed: int = 0
    dtype: Any = th.float64

    def __iter__(self):
        for batch in range(self.batch_count):
            yield th.randn(self.batch_size, self.dim, dtype=self.dtype, device=self.device)*self.sigma + self.mu


@dataclass
class UniformDataLoader(BaseDataLoader):
    range: Tuple[int, int]
    dtype: Any = th.float64

    def __iter__(self):
        h, l = self.range
        for batch in range(self.batch_count):
            yield th.rand(self.batch_size, self.dim, dtype=self.dtype, device=self.device)*(h-l) + l

@dataclass
class CombinedDataLoader(BaseDataLoader):
    data_loaders: List[BaseDataLoader]

    def __iter__(self):
        for data in zip(*self.data_loaders):
            yield th.concat(data, dim=1)
        
    def __is_list_equal(self, l):
        return l.count(l[0]) == len(l)

    def __post_init__(self):
        # Check whether batch sizes match and dimensions add up
        assert self.__is_list_equal([d.batch_size for d in self.data_loaders]+[self.batch_size]), "Batch sizes should be equal"
        assert self.__is_list_equal([d.batch_count for d in self.data_loaders]+[self.batch_count]), "Batch counts should be equal"
        assert sum([d.dim for d in self.data_loaders]) == self.dim, f"Dimensions should add up {','.join([str(d.dim) for d in self.data_loaders])} vs {self.dim}"

            
def test_data_loaders():
    uni = UniformDataLoader(100, 10, 1, 'cpu', 0, range=[-1, 1])
    gaus = GaussianDataLoader(100, 10, 1, 'cpu', 0, mu=1, sigma=0.01)
    comb = CombinedDataLoader(100, 10, 2, 'cpu', 0, data_loaders=[uni, gaus])

    for batch in comb:
        for d in batch:
            assert -1 <= d[0] <= 1 # since uniform
            # print(d)