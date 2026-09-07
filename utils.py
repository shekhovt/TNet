# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import relimport


import sys
import builtins
import traceback
import os
import math
import pickle
import numbers
from collections import OrderedDict
import scipy.stats
import numpy as np

import torch
import threading

from .tools import dotdict
from .functional import *

threadLocal = threading.local()
threadLocal.AttributeError = None


odict = dotdict

def divup(x, y):
    return (x + y - 1) // y


def format_digits(x, sig_digits):
    # digits = -int(math.floor(math.log10(abs(x)))) + sig_digits - 1
    # return '{num:.{digits}f}'.format(num=x, digits=digits)
    return '{num:.{digits}g}'.format(num=x, digits=sig_digits)


def format_std(x, std=0.0, std1=0.0, units=''):
    """ given a value and its std generate string like 0.23 ± 0.01 with significant digits determined by std"""
    ci = math.sqrt(std ** 2 + std1 ** 2)
    stddigits = 1
    if abs(x) == 0.0 or math.isnan(x) or math.isinf(x):
        digit0 = 0
    else:
        digit0 = int(math.floor(math.log10(abs(x))))  # decimal digit of x
    if ci > 0:
        digit = int(math.floor(math.log10(ci))) - 1
        if std >= 10:
            digit -=1
            stddigits += 1
        sdigits = digit0 - digit  # take digits to precision of ci
    else:
        sdigits = 3
        # take 3 significant digits of x
    sdigits = max(sdigits, 0)  # otherwise format does not work
    # sdigits = max(sdigits, digit0)  # otherwise format does not work
    s = '{num:.{digits}g}'.format(num=x, digits=sdigits)
    l = 8
    if std > 0:
        s += ' ±{num:.{digits}g}'.format(num=std, digits=stddigits)
        l += 6
    if std1 > 0:
        s += ' ±{num:.{digits}g}'.format(num=std1, digits=stddigits)
        l += 6
    s = s + units
    s = s.ljust(l)
    return s


# def hasattr(o, a, orig_hasattr=hasattr):
#     try:
#         getattr(o, a)
#         return True
#     except AttributeError:
#         threadLocal.AttributeError = None
#         return False
#     return r


# builtins.hasattr = hasattr


# class SAttributeError(AttributeError):
#     """ An attribute error that keeps a list of stack frames and the original AttributeError"""

#     def __init__(self, e: AttributeError = None):
#         self.e = e
#         self.outer = []

#     def __str__(self):
#         """ Method to print it """
#         s = 'Aux stack trace:\n'
#         # print the stack outer frames
#         lines = traceback.StackSummary.extract(
#             [(f, f.f_lineno) for f in reversed(self.outer)]).format()
#         s += ''.join(lines)
#         # print the stack inner frames
#         lines = traceback.format_tb(self.e.__traceback__.tb_next)
#         s += ''.join(lines)
#         # print the original error message
#         s += str(self.e)
#         return s


# class ModuleBase(torch.nn.Module):
#     def register_buffer(self, name, tensor):
#         """register anything suitable"""
#         if hasattr(self, name) and name not in self._buffers:
#             raise KeyError("attribute '{}' already exists".format(name))
#         elif '.' in name:
#             raise KeyError("buffer name can't contain \".\"")
#         elif name == '':
#             raise KeyError("buffer name can't be empty string \"\"")
#         else:
#             self._buffers[name] = tensor

#     def __setattr__(self, name, value):
#         """ exclude properties from nn.Module's __setattr__"""
#         if isinstance(getattr(type(self), name, None), property):  # check the class has this name, not accessing the getter
#             return object.__setattr__(self, name, value)
#         else:
#             return torch.nn.Module.__setattr__(self, name, value)

#     # def __getattr__(self, name): todo: leaks memory through remembering stack frames and their locals
#     #     """ less intrusive fix of nn.Module __getattr__ so that we at least see which atribute has failed"""
#     #     try:
#     #         return super().__getattr__(name)
#     #     except AttributeError as e:
#     #         if threadLocal.AttributeError is None:
#     #             threadLocal.AttributeError = SAttributeError(e)
#     #         tb = e.__traceback__
#     #         threadLocal.AttributeError.outer.append(tb.tb_frame.f_back)
#     #         raise threadLocal.AttributeError from None


# # todo: running stat update with a r.v (takeƒ into account variance), or with a batch of data

# # class odict(OrderedDict):
# #     def __getattr__(self, name):
# #         if name not in self:
# #             return None
# #         else:
# #             return self[name]
# #
# #     def __setattr__(self, name, value):
# #         self[name] = value
# #
# #     def __delattr__(self, name):
# #         del self[name]


# # def save_object(filename, obj):
# #     with open(filename, 'wb') as output:
# #         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class RunningStat:
    """
    Weighted mean and variance of a sample, estimated online.
    With each new sample the previous estimate is weighted with q and the new sample is weighted with (1-q)
    For fixed q this gives exponential weights q^{n-1}, q^{n-2}(1-q),..., (1-q)
    q can be changed with each samle, giving weights q_0*q_1*..*q_{n-1}, ..., 1-q_{n-1}, where q_0=1
    A useful reference is http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, Section 9
    """

    def __init__(self, q=0.95):
        # current momentum, i.e. (1-q) is the current weight for a new point
        self.q = q
        self.n = 0  # number of points seen
        self.m = 0  # running mean
        self.v = 0  # running variance (biased)
        self.Q2 = 0  # running sum of w_i^2

    def update(self, x, new_q=None):
        if new_q is None:
            new_q = self.q
        if self.n == 0:
            self.q = 0.0  # first sample must have 1
        m = x * (1 - self.q) + self.m * self.q
        self.v = (x - m) * (x - self.m) * (1 - self.q) + self.v * self.q
        self.m = m
        self.Q2 = (1 - self.q) ** 2 + self.Q2 * self.q ** 2
        #
        self.n += 1
        self.q = new_q

    def update_batch(self, x, keepdims=None, new_q=None):
        """insert elements of x sequentially, keepdims defines dimensions of the avarage while the remaining dimensions are averaged over with weights"""
        # squeeze dimensions to average over, run a loop
        raise NotImplementedError()

    @property
    def mean(self):
        """ unbiased mean estimate """
        return self.m

    @property
    def std_of_mean(self):
        """ variance of the mean estimate using sample variance """
        if isinstance(self.var, numbers.Number):
            return math.sqrt(self.var) * math.sqrt(self.Q2)
        else:
            return self.var.sqrt() * math.sqrt(self.Q2)

    @property
    def var(self):
        """ unbiased estimate of variance """
        """ https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance """
        if self.n < 2:
            return 0
        return self.v / (1 - self.Q2)


class RunningStatIdx:
    def __init__(self, N, q=0.95):
        # current momentum, i.e. (1-q) is the current weight for a new point
        self.q = q
        self.n = np.zeros(N, dtype='int')  # number of points seen
        self.m = np.zeros(N)  # running mean
        self.v = np.zeros(N)  # running variance (biased)
        self.Q2 = np.zeros(N)  # running sum of w_i^2

    def update(self, x, idx, new_q=None):
        if new_q is None:
            new_q = self.q
        mask = self.n[idx] == 0
        self.m[idx][mask] = x
        m[idx] = x * (1 - self.q) + self.m * self.q
        self.v = (x - m) * (x - self.m) * (1 - self.q) + self.v * self.q
        self.m = m
        self.Q2 = (1 - self.q) ** 2 + self.Q2 * self.q ** 2
        #
        self.n += 1
        self.q = new_q

    @property
    def mean(self):
        """ unbiased mean estimate """
        return self.m

    @property
    def std_of_mean(self):
        """ variance of the mean estimate using sample variance """
        if isinstance(self.var, numbers.Number):
            return math.sqrt(self.var) * math.sqrt(self.Q2)
        else:
            return self.var.sqrt() * math.sqrt(self.Q2)

    @property
    def var(self):
        """ unbiased estimate of variance """
        """ https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance """
        if self.n < 2:
            return 0
        return self.v / (1 - self.Q2)


class RunningStatAdaptive(RunningStat):
    def __init__(self, alpha, speed=1):
        RunningStat.__init__(self)
        self.alpha = alpha
        self.speed = speed

    def update(self, x):
        b = min(max((self.n + 1) / 10, 1), 100 / self.speed)
        new_q = pow(self.alpha, 1 / b)
        RunningStat.update(self, x, new_q)

    def update_batch(self, x, keepdims=None):
        """insert elements of x sequentially, keepdims defines dimensions of the avarage while the remaining dimensions are averaged over with weights"""
        # squeeze dimensions to average over, run a loop
        raise NotImplementedError()


class RunningStatAvg(RunningStat):
    def update(self, x):
        new_q = (self.n + 1) / (self.n + 2)
        RunningStat.update(self, x, new_q)

    def update_stat(self, mean, var, n2):
        self.m = (self.n * self.m + n2 * mean) / (self.n + n2)
        self.v = (self.n * self.v + n2 * var) / (self.n + n2)
        self.n = self.n + n2
        self.Q2 = 1 / self.n
        self.q = (self.n) / (self.n + 1)  # for adding one point

    @property
    def std_conf_interval(self, alpha=0.05):
        df = self.n - 1
        scale = math.sqrt(df * self.var)
        return [scale / math.sqrt(scipy.stats.chi2.ppf(1 - alpha / 2, df)), scale / math.sqrt(scipy.stats.chi2.ppf(alpha / 2, df))]


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def samples_to_batch(data, n_samples):
    # copy on dimension 0
    rep = [n_samples] + [1] * (data.dim() - 1)
    return data.repeat(rep)  # repeat on the batch dimension


def batch_to_samples(data, n_samples):
    # reshape on dimension 0
    sz = [n_samples, data.size(0) // n_samples] + list(data.size()[1:])
    return data.view(sz)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if __run__:
    print(format_std(991,13))