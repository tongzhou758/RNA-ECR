# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from typing import Sequence, Tuple, List, Union
import pickle
import re
import shutil
import torch
from pathlib import Path
from .constants import proteinseq_toks, rnaseq_toks
import math
import random
from copy import deepcopy
import numpy as np
import _pickle as cPickle
import collections
from .utils_ss import *
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch
from itertools import permutations, product
import pdb
from collections import defaultdict


RawMSA = Sequence[Tuple[str, str]]


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<pad>", "<eos>", "<unk>"), # "<null_0>",
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"), # 
        prepend_bos: bool = True,
        append_eos: bool = True,
        use_msa: bool = False,
        mask_prob: float = 0.15, ###---
    ):
        self.mask_prob = mask_prob ###---
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
#         for i in range((8 - (len(self.all_toks) % 8)) % 8):
#             self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
#         print(self.tok_to_idx)
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<pad>', '<mask>'] # , '<unk>', '<cls>'
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self):
        if self.use_msa:
            return MSABatchConverter(self)
        else:
            return BatchConverter(self)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"] ###---rnaseq
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]

class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs, mask_prob = 0.15):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        self.mask_prob = mask_prob
        
    @classmethod
    def from_file(cls, fasta_file, mask_prob = 0.15):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_strs)) == len(
            sequence_strs
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs, mask_prob)

    def __len__(self):
        return len(self.sequence_labels)
    
    def mask_sequence(self, seq): ###---
        length = len(seq)
#         print(self.mask_prob)
        max_length = math.ceil(length * self.mask_prob)
        rand = random.sample(range(0, length), max_length)
        res = ''.join(['<mask>' if idx in rand else ele for idx, ele in enumerate(seq)])
        #print(seq, rand, res)
        return rand, res
    
    def __getitem__(self, idx):
        sequence_str = self.sequence_strs[idx]
        sequence_label = self.sequence_labels[idx]
        masked_indices, masked_sequence_str = self.mask_sequence(sequence_str)
        return sequence_label, sequence_str, masked_sequence_str, masked_indices

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches

class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]): 
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list, masked_seq_str_list, masked_indices_list = zip(*raw_batch)
        
        masked_seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in masked_seq_str_list] ###---
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list] ###---
#         print('====', seq_str_list)
#         print('----', masked_seq_str_list)
#         print('++++', masked_seq_encoded_list)
#         print('****', seq_encoded_list)
        
        max_len = max(len(seq_encoded) for seq_encoded in masked_seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        masked_tokens = deepcopy(tokens)
        
        labels = []
        strs, masked_strs = [], []
        masked_indices = []
#         print('=================')
        for i, (label, seq_str, masked_seq_str, seq_encoded, masked_seq_encoded, indices_mask) in enumerate(
            zip(batch_labels, seq_str_list, masked_seq_str_list, seq_encoded_list, masked_seq_encoded_list, masked_indices_list) ###---
        ):
            labels.append(label)
            strs.append(seq_str)
            masked_strs.append(masked_seq_str)
            masked_indices.append(indices_mask)
            
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
                masked_tokens[i, 0] = self.alphabet.cls_idx
                
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            masked_seq = torch.tensor(masked_seq_encoded, dtype=torch.int64)
#             print(tokens, masked_tokens)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            
            masked_tokens[
                i,
                int(self.alphabet.prepend_bos) : len(masked_seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = masked_seq
#             print(tokens, masked_tokens)
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                masked_tokens[i, len(masked_seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
#             print(tokens, masked_tokens)
        return labels, strs, masked_strs, tokens, masked_tokens, masked_indices


class MSABatchConverter(BatchConverter):
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens


def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip()
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)


class ESMStructuralSplitDataset(torch.utils.data.Dataset):
    """
    Structural Split Dataset as described in section A.10 of the supplement of our paper.
    https://doi.org/10.1101/622803

    We use the full version of SCOPe 2.07, clustered at 90% sequence identity,
    generated on January 23, 2020.

    For each SCOPe domain:
        - We extract the sequence from the corresponding PDB file
        - We extract the 3D coordinates of the Carbon beta atoms, aligning them
          to the sequence. We put NaN where Cb atoms are missing.
        - From the 3D coordinates, we calculate a pairwise distance map, based
          on L2 distance
        - We use DSSP to generate secondary structure labels for the corresponding
          PDB file. This is also aligned to the sequence. We put - where SSP
          labels are missing.

    For each SCOPe classification level of family/superfamily/fold (in order of difficulty),
    we have split the data into 5 partitions for cross validation. These are provided
    in a downloaded splits folder, in the format:
            splits/{split_level}/{cv_partition}/{train|valid}.txt
    where train is the partition and valid is the concatentation of the remaining 4.

    For each SCOPe domain, we provide a pkl dump that contains:
        - seq    : The domain sequence, stored as an L-length string
        - ssp    : The secondary structure labels, stored as an L-length string
        - dist   : The distance map, stored as an LxL numpy array
        - coords : The 3D coordinates, stored as an Lx3 numpy array

    """

    base_folder = "structural-data"
    file_list = [
        #  url  tar filename   filename      MD5 Hash
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz",
            "splits.tar.gz",
            "splits",
            "456fe1c7f22c9d3d8dfe9735da52411d",
        ),
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz",
            "pkl.tar.gz",
            "pkl",
            "644ea91e56066c750cd50101d390f5db",
        ),
    ]

    def __init__(
        self,
        split_level,
        cv_partition,
        split,
        root_path=os.path.expanduser("~/.cache/torch/data/esm"),
        download=False,
    ):
        super().__init__()
        assert split in [
            "train",
            "valid",
        ], "train_valid must be 'train' or 'valid'"
        self.root_path = root_path
        self.base_path = os.path.join(self.root_path, self.base_folder)

        # check if root path has what you need or else download it
        if download:
            self.download()

        self.split_file = os.path.join(
            self.base_path, "splits", split_level, cv_partition, f"{split}.txt"
        )
        self.pkl_dir = os.path.join(self.base_path, "pkl")
        self.names = []
        with open(self.split_file) as f:
            self.names = f.read().splitlines()

    def __len__(self):
        return len(self.names)

    def _check_exists(self) -> bool:
        for (_, _, filename, _) in self.file_list:
            fpath = os.path.join(self.base_path, filename)
            if not os.path.exists(fpath) or not os.path.isdir(fpath):
                return False
        return True

    def download(self):

        if self._check_exists():
            print("Files already downloaded and verified")
            return

        from torchvision.datasets.utils import download_url

        for url, tar_filename, filename, md5_hash in self.file_list:
            download_path = os.path.join(self.base_path, tar_filename)
            download_url(url=url, root=self.base_path, filename=tar_filename, md5=md5_hash)
            shutil.unpack_archive(download_path, self.base_path)

    def __getitem__(self, idx):
        """
        Returns a dict with the following entires
         - seq : Str (domain sequence)
         - ssp : Str (SSP labels)
         - dist : np.array (distance map)
         - coords : np.array (3D coordinates)
        """
        name = self.names[idx]
        pkl_fname = os.path.join(self.pkl_dir, name[1:3], f"{name}.pkl")
        with open(pkl_fname, "rb") as f:
            obj = pickle.load(f)
        return obj


class RNASSDataGenerator(object):
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
                                             'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s' % self.split), 'rb') as f:
            self.data = cPickle.load(f, encoding='iso-8859-1')
        if self.upsampling:
            self.data = self.upsampling_data_new()
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        self.pairs = [instance[-1] for instance in self.data]  # np.array([instance[-1] for instance in self.data])
        # pdb.set_trace()
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        self.data_name = np.array([instance[3] for instance in self.data])
        # self.matrix_rep = np.array(list(p.map(creatmat, self.seq)))
        # self.matrix_rep = np.zeros([self.len, len(self.data_x[0]), len(self.data_x[0])])

    def upsampling_data(self):  # 8 种 RNA 家族
        pdb.set_trace()
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('/')[2], name)))
        data = np.array(self.data)  # 提取 RNA 家族类别
        max_num = max(Counter(list(d_type)).values())  # 寻找“最大类别”并对数据分组
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]  # 按字母顺序排序
            data_list.append(data[index])
        final_d_list = list()
        # for d in data_list:
        #     index = np.random.choice(d.shape[0], max_num)
        #     final_d_list += list(d[index])
        for i in [0, 1, 5, 7]:  # 将它们的数量强行扩充（或缩减）到 max_num
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num)
            final_d_list += list(d[index])

        for i in [2, 3, 4]:  # 把它们的数量扩充到了最大数量的两倍
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num * 2)
            final_d_list += list(d[index])

        d = data_list[6]
        index = np.random.choice(d.shape[0], int(max_num / 2))  # 对于第 6 个家族，只抽取了最大数量的一半
        final_d_list += list(d[index])

        shuffle(final_d_list)
        return final_d_list

    def upsampling_data_new(self):
        name = [instance.name for instance in self.data]  # 提取类别标签
        d_type = np.array(list(map(lambda x: x.split('_')[0], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()  # 按类别对数据进行分组
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]  # 按字母顺序排序
            data_list.append(data[index])
        final_d_list = list()  # 扩充
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300 - d.shape[
                    0])  # 如果某个类别的样本数少于 300 个，代码会使用 np.random.choice（有放回地随机抽样）抽取缺少的数量（300 - 当前数量）
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                print('processing PDB seq...')
                index = np.random.choice(d.shape[0], d.shape[
                    0] * 4)  # 只要某个类别的样本数刚好等于 652 个，那就认定它是高质量的 PDB（Protein Data Bank）数据集。认定后，随机抽取该类别数据量 4 倍的样本加进去。加上原本的基础盘，等于把这批数据的权重强行放大了 5 倍。目的是强迫神经网络多学习这些高质量的三维结构解析数据
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list


    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact


    def get_one_sample(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        # data_len = np.nonzero(self.data_x[index].sum(axis=2))[0].max()
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact = self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name

perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1,3],[3,1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

class Dataset_Cut_concat_new_merge_multi(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_list):
        'Initialization'
        self.data2 = data_list[0]
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def merge_data(self, data_list):

        self.data2.data_x = np.concatenate((data_list[0].data_x, data_list[1].data_x), axis=0)
        self.data2.data_y = np.concatenate((data_list[0].data_y, data_list[1].data_y), axis=0)
        self.data2.seq_length = np.concatenate((data_list[0].seq_length, data_list[1].seq_length), axis=0)
        self.data2.pairs = np.concatenate((data_list[0].pairs, data_list[1].pairs), axis=0)
        self.data2.data_name = np.concatenate((data_list[0].data_name, data_list[1].data_name), axis=0)
        for item in data_list[2:]:
            self.data2.data_x = np.concatenate((self.data2.data_x, item.data_x), axis=0)
            self.data2.data_y = np.concatenate((self.data2.data_y, item.data_y), axis=0)
            self.data2.seq_length = np.concatenate((self.data2.seq_length, item.seq_length), axis=0)
            self.data2.pairs = np.concatenate((self.data2.pairs, item.pairs), axis=0)
            self.data2.data_name = np.concatenate((self.data2.data_name, item.data_name), axis=0)

        self.data2.len = len(self.data2.data_name)
        return self.data2

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)  # 计算边长 l
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:  # 扩充画布
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):  # 二维矩阵
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                          data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len,])  # 引入物理热力学先验
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len))
        feature = np.concatenate((data_fcn, feature), axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name

class Dataset_Cut_concat_new_canonicle(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        #contact, data_seq, matrix_rep, data_len, data_name, data_pair = self.data.get_one_sample_addpairs(index)
        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        #data_nc = np.zeros((2, l, l))
        data_nc = np.zeros((10, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_nc = data_nc.sum(axis=0).astype(np.bool)
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        #zero_mask = z_mask(data_len)[None, :, :, None]
        #label_mask = l_mask(data_seq, data_len)
        #temp = data_seq[None, :data_len, :data_len]
        #temp = np.tile(temp, (temp.shape[1], 1, 1))
        #feature[:,:data_len,:data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1,data_len,data_len))
        #feature = np.concatenate((data_fcn,feature),axis=0)
        #return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        #return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)
        #return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc, data_pair
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc,l


def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)

def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)

def creatmat(data, device=None):
    if device==None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        data = ''.join(['AUCG'[list(d).index(1)] for d in data])
        paired = defaultdict(int, {'AU':2, 'UA':2, 'GC':3, 'CG':3, 'UG':0.8, 'GU':0.8})

        mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
        n = len(data)

        # i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing=None)
        i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing='ij')
        t = torch.arange(30).to(device)
        m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n), mat[torch.clamp(i[:,:,None]-t, 0, n-1), torch.clamp(j[:,:,None]+t, 0, n-1)], 0)
        m1 *= torch.exp(-0.5*t*t)

        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad==0).to(int), dim=2)
        to0indices = t[None,None,:]>first0[:,:,None]
        m1[to0indices] = 0
        m1 = m1.sum(dim=2)

        t = torch.arange(1, 30).to(device)
        m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0), mat[torch.clamp(i[:,:,None]+t, 0, n-1), torch.clamp(j[:,:,None]-t, 0, n-1)], 0)
        m2 *= torch.exp(-0.5*t*t)

        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad==0).to(int), dim=2)
        to0indices = torch.arange(29).to(device)[None,None,:]>first0[:,:,None]
        m2[to0indices] = 0
        m2 = m2.sum(dim=2)
        m2[m1==0] = 0

        return (m1+m2).to(torch.device('cpu'))

