from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Tuple
import numpy as np
import networkx as nx
from transformers import TrainerCallback, Trainer
from sklearn.metrics import roc_auc_score
import torch

from typing import Iterable, Literal, Optional, List, Dict
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value

GraphDict = Dict[str, object]  # edge_index, num_nodes, y, node_feat, edge_attr

LabelType = Literal["int", "float", "multitask_int", "multitask_float"]

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import math

from utils import *


class MyTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)


def _unwrap_to_numpy(x):
    # peel nested (tuple/list/…)-> take first element repeatedly
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def compute_metrics(eval_pred):
    # eval_pred is EvalPrediction(predictions=..., label_ids=...)
    preds, labels = eval_pred

    logits = _unwrap_to_numpy(preds)
    y = _unwrap_to_numpy(labels)

    if logits is None or y is None:
        return {"roc_auc": float("nan")}

    # Ensure 1D label vector of ints
    y = np.asarray(y).astype(float).ravel()
    # Some OGB tasks have -1 for missing labels; keep MOLHIV simple:
    mask = ~np.isnan(y)
    y = y[mask]

    # Convert logits -> prob of class 1
    logits = np.asarray(logits)
    if logits.ndim == 2 and logits.shape[1] == 2:
        # softmax[:,1]
        # (use stable softmax)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs1 = (e[:, 1] / e.sum(axis=1))[mask]
    else:
        # single logit -> sigmoid
        z = logits.squeeze()
        probs1 = (1.0 / (1.0 + np.exp(-z)))[mask]

    # Guard tiny/degenerate batches
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan")}

    return {"roc_auc": roc_auc_score(y.astype(int), probs1)}


# print a neat line at each epoch end
class EpochPrinter(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # last log entry has running loss; eval loss will appear after evaluate()
        last = state.log_history[-1] if state.log_history else {}
        cur = {k: v for k, v in last.items() if k in ("loss", "eval_loss", "roc_auc", "learning_rate")}
        print(f"[epoch {state.epoch:.0f}] {cur}")


@dataclass
class GraphBuildConfig:
    corr_threshold: float = 0.9999          # connect i<->j if (abs(corr) if use_abs else corr) < corr_threshold
    use_abs: bool = False          # use |corr| comparison by default
    bidirectional: bool = False           # add both i->j and j->i
    add_self_loops: bool = False         # normally False
    clip_corr: bool = True              # clip numerical noise to [-1, 1]
    scalePatient: bool = True
    normPatient: bool = False

# TODO edge connection on a corr threshold condiction

class GraphDataset:
    def __init__(self, X: np.ndarray, Y: np.ndarray, 
                 #train_idx: list, test_idx: list, val: bool = False, 
                 n_bins: int = 64, make_bins: bool = False,
                 seed: int = 42,
                 task: str = 'binary', pos_class_label: int = 0, 
                 cfg: Optional[GraphBuildConfig] = None,
                 purpose: str = 'train') -> None: 
        #self.train.ids = train_idx
        #self.test_idx = test_idx
        self.X = X
        self.seed = seed
        self.task = task
        assert self.X.ndim == 3, f"X must have shape (S, R, N); got {X.shape}"
        self.len_data = self.X.shape[0]
        assert self.len_data == len(Y), f"len(y)={len(Y)} does not match number of samples S={self.len_data}"
        #assert self.len_data <= len(train_idx) + len(test_idx), \
        #    f"number of samples S={self.len_data} is less than train and test combined length N={len(train_idx) + len(test_idx)}"

        if self.task == 'binary':
            label = [0 if t == pos_class_label else 1 for t in Y]
            self.Y = label
        else:
            raise NotImplementedError('task can only be binary for now')
        
        self.preprocessed = False
        self.numFeats = X.shape[1]
        self.cfg = cfg

        self.purpose = purpose
    
        self.n_bins, self.make_bins = n_bins, make_bins

        if make_bins:
            print(f'Will split values into {n_bins} bins')

        self.limit_dict = {
            "node_min": np.full(9, np.inf),
            "node_max": np.full(9, -np.inf),
            "edge_min": np.full(3, np.inf),
            "edge_max": np.full(3, -np.inf),
        }
        self.bins_dict = {}

        self.d_types = [np.int64, 'int64'] if make_bins else [np.float32, 'float32']
    
        #self.corrCoefs = np.empty((len(train_idx)), dtype=np.float32)

    def preprocess_everething(self, scalePatient: bool = False, normPatient: bool = True):

        assert not (scalePatient and normPatient)

        if scalePatient:
                self.X = scale_patient(self.X)
        if normPatient:
                self.X = normalize_data(self.X)

        self.preprocessed = True
        print('Successfully preprocessed dataset', f'scalePatient: {scalePatient}, normPatient: {normPatient}')

    def get_bins_dict(self,) -> dict:
        return self.bins_dict
    
    def _calculate_bins(self,) -> dict:

        for elem, size in zip(['edge', 'node'], [3, 9]):
            int_lims = np.empty((size, self.n_bins))
            for i in range(size):
                min_v, max_v = self.limit_dict[f'{elem}_min'][i], self.limit_dict[f'{elem}_max'][i], 
                lim = np.arange(min_v, max_v, (max_v - min_v ) / self.n_bins)
                int_lims[i] = lim
            self.bins_dict[elem] = int_lims
    
    def _corr_matrix(self, Xi: np.ndarray, split:str = 'train') -> np.ndarray:
        """
        Compute R x R correlation matrix across N time points for R channels.
        X shape: (R, N). Returns corr of shape (R, R).
        """
        # Handle constant channels robustly to avoid NaNs in correlation.
        # np.corrcoef handles this but returns nan if std==0; replace with 0 in that case.
        C = np.corrcoef(Xi)
        C = np.asarray(C, dtype=np.float64)
        np.nan_to_num(C, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # get cor coeffs for train batch for train val test batch normalization 0;1
        if split == 'train':
            lower_diagonal_mask = np.tril(np.ones_like(C), k=-1).astype(bool)
            lower_diagonal_elements = C[lower_diagonal_mask]
            self.corrCoefs = np.append(self.corrCoefs, lower_diagonal_elements)

        return C
    
    def _get_edge_attr(self, ind, Ri: int, Rj: int) -> List[float]:
                
        series_i, series_j = self.X[ind, Ri], self.X[ind, Rj]
        min_i, min_j, max_i, max_j = np.min(series_i), np.min(series_j), np.max(series_i), np.max(series_j)
        min_diff = np.abs(min_i - min_j)
        max_diff = np.abs(max_i - max_j)
        std_i, std_j = np.std(series_i, ddof=1), np.std(series_j, ddof=1)
        std_diff = std_i**2/std_j**2

        res = [min_diff, max_diff, std_diff]
        self.limit_dict['edge_min'] = np.min((self.limit_dict['edge_min'], res), axis=0)
        self.limit_dict['edge_max'] = np.max((self.limit_dict['edge_max'], res), axis=0)

        return res
        #return [0]*3

    def _get_node_feat(self, ind, Ri: int) -> List[float]:
        
        series_i = self.X[ind, Ri]
        n_points = len(series_i)
        min_i, max_i, mean_i, std_i = np.min(series_i), np.max(series_i), np.mean(series_i), np.std(series_i)
        t50, k = featurize_series(series_i, ind, method='logfit_2')
        a0, a1, a2 = featurize_series(series_i, ind, method='poly_3')

        res = [max_i, min_i, mean_i, std_i, k, t50/n_points, a0, a1, a2]
        self.limit_dict['node_min'] = np.min((self.limit_dict['node_min'], res), axis=0)
        self.limit_dict['node_max'] = np.max((self.limit_dict['node_max'], res), axis=0)

        return res
        #return list(np.clip(np.abs(np.array([max_i, min_i, mean_i, std_i, k, t50/n_points, a0, a1, a2])*5), 0, 100))
        #return list(np.random.randint(1, 5, 9))

    # TODO edge connection on a connectCondition = 'correlation'

    def _edge_lists_from_corr(self, ind: int,
        connectCondition: str = 'always',
        C: np.ndarray = None,
    ) -> Tuple[List[int], List[int], List[List[float]]]:
        """
        Build edge lists (sources, targets) and edge_attr ([[corr], ...]) from correlation matrix.
        """
        R = self.numFeats
        src: List[int] = []
        dst: List[int] = []
        edge_attr: List[List[float]] = []

        if connectCondition == 'always':
            for i in range(R):
                for j in range(R):
                    if i < j:  # single direction choice to avoid duplicates
                        local_attr = self._get_edge_attr(ind, i, j)
                        src.append(i); dst.append(j); edge_attr.append(local_attr)

        elif connectCondition == 'correlation':
            def passes_threshold(val: float) -> bool:
                v = abs(val) if self.cfg.use_abs else val
                return v < self.cfg.corr_threshold

            for i in range(R):
                for j in range(R):
                    if i == j and not self.cfg.add_self_loops:
                        continue
                    if i == j and self.cfg.add_self_loops:
                        cij = C[i, j]
                        if self.cfg.clip_corr:
                            cij = float(np.clip(cij, -1.0, 1.0))
                        if passes_threshold(cij):
                            src.append(i); dst.append(j); edge_attr.append([cij])
                        continue

                    if i < j:  # single direction choice to avoid duplicates
                        cij = C[i, j]
                        if self.cfg.clip_corr:
                            cij = float(np.clip(cij, -1.0, 1.0))
                            cij_l = [cij]*3
                        if passes_threshold(cij):
                            src.append(i); dst.append(j); edge_attr.append(cij_l)

        return src, dst, edge_attr

    def build_graph_from_sample(
        self, ind: int, sample,
        label: int = None,
    ) -> GraphDict:
        """
        Convert a single sample (R x N) into a graph dictionary in the exact format you specified.

        Parameters
        ----------
        sample : np.ndarray
            Shape (R, N): R channels (nodes), each with N scalar measurements over time.
        label : Optional[int | float | List[int] | List[float]]
            Classification label (int for multiclass), regression target (float), or multi-task list.
            If None, y will be an empty list [].
        cfg : GraphBuildConfig
            Controls thresholding, absolute corr, directionality, etc.

        Returns
        -------
        graph : dict
            {
            "edge_index": [[src...], [dst...]],
            "num_nodes": R,
            "y": [label] or label-list-as-is,
            "node_feat": [[...N features per node...], ...],
            "edge_attr": [[corr], [corr], ...]
            }
        """

        R, N = self.X[ind].shape

        # Node features: each node's feature vector is the channel vector (length N)
        # Ensure pure Python lists of floats for strict downstream compatibility.
        node_feat: List[List[float]] = [self._get_node_feat(ind, Ri) for Ri in range(R)]

        # Correlation matrix across channels
        #C = _corr_matrix(sample)

        # Edge lists & attributes
        src, dst, edge_attr = self._edge_lists_from_corr(ind)

        # y formatting
        y_val = [int(label)]

        graph: GraphDict = {
            "edge_index": [src, dst],            # list of 2 lists of integers
            "num_nodes": int(R),                 # integer
            "y": y_val,                          # list with one element (or multi-task list)
            "node_feat": node_feat,              # list of lists (each length N)
            "edge_attr": edge_attr               # list of [corr] in edge order
        }
        return graph


    def build_graph_dataset(
        self,
        bins_dict: dict = None
    ) -> Dataset:
        """
        Convert a dataset of samples into a list of graph dictionaries.

        Parameters
        ----------
        X : array-like
            - If np.ndarray: shape (S, R, N) for S samples.
            - If iterable: yields (R, N) arrays.
        y : iterable or None
            Labels per sample. If None, creates empty labels.
        cfg : GraphBuildConfig
            Graph construction controls.

        Returns
        -------
        graphs : List[GraphDict]
        """
        assert not (self.purpose != 'train' and bins_dict is None), 'a dict of bin intervals should be specified for val or test'
        assert self.preprocessed == True, 'you should call preprocess_everything class method first'
        
        # Normalize X to an iterable of (R, N)
        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (S, R, N); got {self.X.shape}")
        samples_iter = (self.X[i] for i in range(self.X.shape[0]))
        S = self.X.shape[0]

        labels_list = list(self.Y)
        if len(labels_list) != S:
            raise ValueError(f"len(y)={len(labels_list)} does not match number of samples S={S}")
        labels_iter = iter(labels_list)

        graphs: List[GraphDict] = []
        for ind, (sample, label) in enumerate(zip(samples_iter, labels_iter)):
            graphs.append(self.build_graph_from_sample(ind, sample, label))

        if self.make_bins:
            if self.purpose == 'train':
                self._calculate_bins()
                bins_dict = self.bins_dict
            else:
                bins_dict = bins_dict

            graphs_res = [to_types(graph_to_bins(gr, bins_dict), node_and_edge_type=self.d_types[0]) for gr in graphs]
        else:
            graphs_res = [to_types(gr, node_and_edge_type=self.d_types[0]) for gr in graphs]

        return Dataset.from_list(graphs_res, features=hf_features(node_and_edge_type=self.d_types[1]))
    
"""    def build_split(self) -> Dataset:
        records: List[Dict] = []
        for i in range(self.len_data):
            graph = self.build_graph_from_sample(
                sample=self.X[i],
                label=self.Y[i],
                cfg=cfg,
            )
            records.append(to_types(graph))
        return Dataset.from_list(records, features=hf_features())"""


def make_graphormer_datasetdict(
    X: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], # for train, test and optinally val sets
    Y: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    make_bins: bool = False,
    cfg: Optional[GraphBuildConfig] = GraphBuildConfig,
    #val_size: float = 0.2,
    #shuffle: bool = False,
    #seed: int = 42,
    label_type: LabelType = "int",
) -> DatasetDict:
    """
    Build a Hugging Face DatasetDict with 'train' and 'validation' splits
    from (S, R, N) data using the graph format required by your Graphormer code.

    Parameters
    ----------
    X: np.ndarray | Iterable[np.ndarray]
        - np.ndarray with shape (S, R, N), or iterable of (R, N) arrays.
    y: iterable or None
        Labels per sample. If None, empty lists [] are stored.
        Choose label_type to control dtype/shape expectations.
    cfg: GraphBuildConfig
        Correlation-threshold config for graph construction.
    val_size: float
        Fraction of samples to hold out for validation.
    shuffle: bool
        Shuffle before splitting.
    seed: int
        RNG seed for reproducible split.
    label_type: Literal["int", "float", "multitask_int", "multitask_float"]
        Determines dtype/shape of 'y' in Features.

    Returns
    -------
    DatasetDict with 'train' and 'validation'.
    """
    # Normalize X and y to arrays for indexing/splitting

    n_megabatch = len(X)

    assert n_megabatch == len(Y), 'X and Y must have identical parts'
    if n_megabatch == 2:
        keys = ['train', 'validation']
    elif n_megabatch == 3:
        keys = ['train', 'validation', 'test']
    else:
        raise ValueError("X and Y should have 2 (train, val) or 3 (train, val, test) elements")

    megabatch = []
    bins_dict = None 

    for split_X, split_Y, purpose in zip(X, Y, keys):
        
        S = split_X.shape[0]

        if len(split_Y) != S:
            raise ValueError(f"len(y)={len(split_Y)} does not match number of samples S={S}")
        
        ds = GraphDataset(split_X, split_Y, purpose=purpose, make_bins=make_bins)
        ds.preprocess_everething(normPatient=cfg.normPatient, scalePatient=cfg.scalePatient)
        batch = ds.build_graph_dataset(bins_dict=bins_dict) #for train bins_dict is calculated internally
        megabatch.append(batch)
        if purpose == 'train': # for val and test bins dict from train will be used
            bins_dict = ds.get_bins_dict()

    dataset_dict = {k:v for k, v in zip(keys, megabatch)}

    return DatasetDict(dataset_dict)


def hf_features(label_type: LabelType = "int", 
                node_and_edge_type: str = 'int64') -> Features:
        if label_type == "int":
            y_feat = Sequence(Value("int64"))
        elif label_type == "float":
            y_feat = Sequence(Value("float32"))
        elif label_type == "multitask_int":
            y_feat = Sequence(Value("int64"))
        elif label_type == "multitask_float":
            y_feat = Sequence(Value("float32"))
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

        return Features({
            "edge_index": Sequence(Sequence(Value("int64"))),     # [[src...], [dst...]]
            "num_nodes": Value("int64"),
            "y": y_feat,
            "node_feat": Sequence(Sequence(Value(node_and_edge_type))),    # [[N feats], ...]
            "edge_attr": Sequence(Sequence(Value(node_and_edge_type))),    # [[corr], ...]
        })

def to_types(graph: Dict, 
             label_type: LabelType = "int", 
             node_and_edge_type: np.dtype = np.int64) -> Dict:
    """Ensure consistent dtypes for HF storage."""

    ei0 = np.asarray(graph["edge_index"][0], dtype=np.int64).tolist()
    ei1 = np.asarray(graph["edge_index"][1], dtype=np.int64).tolist()
    num_nodes = int(graph["num_nodes"])
    y = [int(v) for v in graph["y"]] 
    node_feat = np.asarray(graph["node_feat"], dtype=node_and_edge_type).tolist()
    edge_attr = np.asarray(graph["edge_attr"], dtype=node_and_edge_type).tolist()

    return {
        "edge_index": [ei0, ei1],
        "num_nodes": num_nodes,
        "y": y,
        "node_feat": node_feat,
        "edge_attr": edge_attr,
    }
        
def graph_to_bins(graph: GraphDict, 
                  bins_dict: dict) -> GraphDict:
    
    node_feats = np.array(graph["node_feat"])
    for i in range(9):
        node_feats[:, i] = np.digitize(node_feats[:, i], bins_dict["node"][i])
    graph["node_feat"] = node_feats.tolist()

    edge_feats = np.array(graph["edge_attr"])
    for i in range(3):
        edge_feats[:, i] = np.digitize(edge_feats[:, i], bins_dict["edge"][i])
    graph["edge_attr"] = edge_feats.tolist()

    return graph
    
    