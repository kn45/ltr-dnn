# -*- coding=utf-8 -*-
import copy
import numpy as np
import sys


class DictTable(object):
    def __init__(self, dict_file, UNK=None):
        self.table = {}
        self.rev_table = {}
        self.UNK = UNK
        if isinstance(dict_file, basestring):
            with open(dict_file) as f:
                for line in f:
                    k, v = line.rstrip('\n').split('\t')
                    self.table[k] = int(v)
                    self.rev_table[int(v)] = k
                    if v == self.UNK:
                        sys.stderr.write('word index is conflict with UNK: ')
                        sys.stderr.write(k + ' ' + v + '\n')
        if isinstance(dict_file, dict):
            self.table = copy.deepcopy(dict_file)
            for k in dict_file:
                self.rev_table[dict_file[k]] = k
                if dict_file[k] == self.UNK:
                    sys.stderr.write('word index is conflict with UNK: ')
                    sys.stderr.write(str(k) + ' ' + str(v) + '\n')

    def lookup(self, words):
        ids = []
        for word in words:
            if word in self.table:
                ids.append(self.table[word])
            else:
                ids.append(self.UNK)
        return ids

    def lookup_rev(self, ids):
        ids = map(int, list(ids))
        words = []
        for idx in ids:
            if idx in self.rev_table:
                words.append(self.rev_table[idx])
            else:
                words.append(None)
        return words


class BatchReader(object):
    """Get batch data recurrently from a file.
    """
    def __init__(self, file_name, max_epoch=None):
        self.fname = file_name
        self.max_epoch = max_epoch
        self.nepoch = 0
        self.fp = None

    def __del__(self):
        if self.fp:
            self.fp.close()

    def get_batch(self, batch_size, out=None):
        if out is None:
            out = []
        if not self.fp:
            if (not self.max_epoch) or self.nepoch < self.max_epoch:
                # if max_epoch not set or num_epoch not reach the limit
                self.fp = open(self.fname)
                self.nepoch += 1
            else:  # reach max_epoch limit
                return out
        for line in self.fp:
            out.append(line.rstrip('\n'))
            if len(out) >= batch_size:
                break
        else:
            self.fp.close()
            self.fp = None
            return self.get_batch(batch_size, out)
        return out


def sparse2dense(ids, ndim):
    out = np.zeros((ndim), dtype=np.int32)
    for idx in ids:
        out[idx] = 1
    return out


def zero_padding(inp, seq_len):
    out = np.zeros((seq_len), dtype=np.int32)
    for i, v in enumerate(inp):
        if i >= seq_len:
            break
        out[i] = v
    return out


def draw_progress(iteration, total, pref='Progress:', suff='',
                  decimals=1, barlen=50):
    """Call in a loop to create terminal progress bar
    """
    formatStr = "{0:." + str(decimals) + "f}"
    pcts = formatStr.format(100 * (iteration / float(total)))
    filledlen = int(round(barlen * iteration / float(total)))
    bar = '█' * filledlen + '-' * (barlen - filledlen)
    out_str = '\r%s |%s| %s%s %s' % (pref, bar, pcts, '%', suff)
    out_str = '\x1b[0;34;40m' + out_str + '\x1b[0m'
    sys.stderr.write(out_str),
    if iteration == total:
        sys.stderr.write('\n')
    sys.stderr.flush()


if __name__ == '__main__':
    freader = BatchReader('../run.sh', 2)
    for i in range(5):
        print freader.get_batch(6)
