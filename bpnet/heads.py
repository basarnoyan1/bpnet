"""Head modules
"""
import numpy as np
from bpnet.utils import dict_prefix_key
from bpnet.metrics import ClassificationMetrics, RegressionMetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import os
import abc


class BaseHead:

    # loss
    # weight -> loss weight (1 by default)
    # kwargs -> kwargs for the model
    # name -> name of the module
    # _model -> gets setup in the init

    @abc.abstractmethod
    def get_target(self, task):
        pass

    @abc.abstractmethod
    def __call__(self, inp, task):
        """Useful for writing together the model
        Returns the output tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_preact_tensor(self, graph=None):
        """Return the single pre-activation tensors
        """
        pass

    @abc.abstractmethod
    def intp_tensors(self, preact_only=False, graph=None):
        """Dictionary of all available interpretation tensors
        for `get_interpretation_node`
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def get_intp_tensor(self, which=None):
    #     """Returns a target tensor which is a scalar
    #     w.r.t. to which to compute the outputs

    #     Args:
    #       which [string]: If None, use the default
    #       **kwargs: optional kwargs for the interpretation method

    #     Returns:
    #       scalar tensor
    #     """
    #     raise NotImplementedError

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


class BaseHeadWBias(BaseHead):

    @abc.abstractmethod
    def get_bias_input(self, task):
        pass

    @abc.abstractmethod
    def neutral_bias_input(self, task, length, seqlen):
        pass


def id_fn(x):
    return x


def named_tensor(x, name):
    """PyTorch version of named tensor - just return the tensor"""
    return x


# --------------------------------------------
# Head implementations

@gin.configurable
class ScalarHead(BaseHeadWBias):

    def __init__(self, target_name,  # "{task}/scalar"
                 net,  # function that takes a keras tensor and returns a keras tensor
                 activation=None,
                 loss='mse',
                 loss_weight=1,
                 metric=RegressionMetrics(),
                 postproc_fn=None,  # post-processing to apply so that we are in the right scale
                 # bias input
                 use_bias=False,
                 bias_net=None,
                 bias_input='bias/{task}/scalar',
                 bias_shape=(1,),
                 ):
        self.net = net
        self.loss = loss
        self.loss_weight = loss_weight
        self.metric = metric
        self.postproc_fn = postproc_fn
        self.target_name = target_name
        self.activation = activation
        self.bias_input = bias_input
        self.bias_net = bias_net
        self.use_bias = use_bias
        self.bias_shape = bias_shape

    def get_target(self, task):
        return self.target_name.format(task=task)

    def __call__(self, inp, task):
        o = self.net(inp)

        # remember the tensors useful for interpretation (referred by name)
        self.pre_act = o

        # add the target bias
        if self.use_bias:
            # Create bias input tensor
            bias_shape = [inp.shape[0]] + list(self.bias_shape)  # Add batch dimension
            bias_input = torch.zeros(bias_shape, device=inp.device)
            bias_inputs = [bias_input]

            # add the bias term
            if self.bias_net is not None:
                bias_x = self.bias_net(bias_input)
            else:
                # Don't use the nn 'bias' so that when the measurement bias = 0,
                # this term vanishes
                bias_layer = nn.Linear(self.bias_shape[0], 1, bias=False)
                if inp.is_cuda:
                    bias_layer = bias_layer.cuda()
                bias_x = bias_layer(bias_input)
            o = o + bias_x
        else:
            bias_inputs = []

        if self.activation is not None:
            if isinstance(self.activation, str):
                if self.activation == 'sigmoid':
                    o = torch.sigmoid(o)
                elif self.activation == 'relu':
                    o = torch.relu(o)
                elif self.activation == 'tanh':
                    o = torch.tanh(o)
                # Add more activations as needed
            else:
                o = self.activation(o)

        self.post_act = o

        # label the target op so that we can use a dictionary of targets
        # to train the model
        return named_tensor(o, name=self.get_target(task)), bias_inputs

    def get_preact_tensor(self, graph=None):
        return self.pre_act

    def intp_tensors(self, preact_only=False, graph=None):
        """Return the required interpretation tensors
        """
        if self.activation is None:
            # the post-activation doesn't
            # have any specific meaning when
            # we don't use any activation function
            return {"pre-act": self.pre_act}

        if preact_only:
            return {"pre-act": self.pre_act}
        else:
            return {"pre-act": self.pre_act,
                    "output": self.post_act}

    # def get_intp_tensor(self, which='pre-act'):
    #     return self.intp_tensors()[which]

    def get_bias_input(self, task):
        return self.bias_input.format(task=task)

    def neutral_bias_input(self, task, length, seqlen):
        """Create dummy bias input

        Return: (k, v) tuple
        """
        shape = tuple([x if x is not None else seqlen
                       for x in self.bias_shape])
        return (self.get_bias_input(task), np.zeros((length, ) + shape))


@gin.configurable
class BinaryClassificationHead(ScalarHead):

    def __init__(self, target_name,  # "{task}/scalar"
                 net,  # function that takes a keras tensor and returns a keras tensor
                 activation='sigmoid',
                 loss='binary_crossentropy',
                 loss_weight=1,
                 metric=ClassificationMetrics(),
                 postproc_fn=None,
                 # bias input
                 use_bias=False,
                 bias_net=None,
                 bias_input='bias/{task}/scalar',
                 bias_shape=(1,),
                 ):
        # override the default
        super().__init__(target_name,
                         net,
                         activation=activation,
                         loss=loss,
                         metric=metric,
                         postproc_fn=postproc_fn,
                         use_bias=use_bias,
                         bias_net=bias_net,
                         bias_input=bias_input,
                         bias_shape=bias_shape)

        # TODO - mabye override the way we call outputs?


@gin.configurable
class ProfileHead(BaseHeadWBias):
    """Deals with the case where the output are multiple tracks of
    total shape (L, C) (L = sequence length, C = number of channels)

    Note: Since the contribution score will be a single scalar, the
    interpretation method will have to aggregate both across channels
    as well as positions
    """

    def __init__(self, target_name,  # "{task}/profile"
                 net,  # function that takes a keras tensor and returns a keras tensor
                 activation=None,
                 loss='mse',
                 loss_weight=1,
                 metric=RegressionMetrics(),
                 postproc_fn=None,
                 # bias input
                 use_bias=False,
                 bias_net=None,
                 bias_input='bias/{task}/profile',
                 bias_shape=(None, 1),
                 ):
        self.net = net
        self.loss = loss
        self.loss_weight = loss_weight
        self.metric = metric
        self.postproc_fn = postproc_fn
        self.target_name = target_name
        self.activation = activation
        self.bias_input = bias_input
        self.bias_net = bias_net
        self.use_bias = use_bias
        self.bias_shape = bias_shape

    def get_target(self, task):
        return self.target_name.format(task=task)

    def __call__(self, inp, task):
        o = self.net(inp)

        # remember the tensors useful for interpretation (referred by name)
        self.pre_act = o

        # add the target bias
        if self.use_bias:
            # Create bias input tensor
            bias_shape = [inp.shape[0]] + list(self.bias_shape)  # Add batch dimension
            # Handle None dimensions
            bias_shape = [s if s is not None else inp.shape[i+1] for i, s in enumerate(bias_shape)]
            bias_input = torch.zeros(bias_shape, device=inp.device)
            bias_inputs = [bias_input]

            # add the bias term
            if self.bias_net is not None:
                bias_x = self.bias_net(bias_input)
            else:
                # Don't use the nn 'bias' so that when the measurement bias = 0,
                # this term vanishes
                if len(bias_shape) == 3:  # Conv1D case
                    bias_layer = nn.Conv1d(bias_shape[-1], 1, kernel_size=1, bias=False)
                else:  # Linear case
                    bias_layer = nn.Linear(bias_shape[-1], 1, bias=False)
                if inp.is_cuda:
                    bias_layer = bias_layer.cuda()
                bias_x = bias_layer(bias_input)
            o = o + bias_x
        else:
            bias_inputs = []

        if self.activation is not None:
            if isinstance(self.activation, str):
                if self.activation == 'sigmoid':
                    o = torch.sigmoid(o)
                elif self.activation == 'relu':
                    o = torch.relu(o)
                elif self.activation == 'tanh':
                    o = torch.tanh(o)
                # Add more activations as needed
            else:
                o = self.activation(o)

        self.post_act = o

        # label the target op so that we can use a dictionary of targets
        # to train the model
        return named_tensor(o, name=self.get_target(task)), bias_inputs

    def get_preact_tensor(self, graph=None):
        return self.pre_act

    @staticmethod
    def profile_contrib(p):
        """Summarizing the profile for the contribution scores

        wn: Normalized contribution (weighted sum of the contribution scores)
          where the weighted sum uses softmax(p) to weight it
        w2: Simple sum (p**2)
        w1: sum(p)
        winf: max(p)
        """
        # Normalized contribution
        softmax_weights = F.softmax(p, dim=-2)
        wn = torch.mean(torch.sum(softmax_weights * p, dim=-2), dim=-1)

        # Squared weight
        w2 = torch.mean(torch.sum(p * p, dim=-2), dim=-1)

        # W1 weight
        w1 = torch.mean(torch.sum(p, dim=-2), dim=-1)

        # Winf
        # 1. max across the positional axis, average the strands
        winf = torch.mean(torch.max(p, dim=-2)[0], dim=-1)

        return {"wn": wn,
                "w1": w1,
                "w2": w2,
                "winf": winf
                }

    def intp_tensors(self, preact_only=False, graph=None):
        """Return the required interpretation tensors (scalars)

        Note: Since we are predicting a track,
            we should return a single scalar here
        """
        preact = self.pre_act
        postact = self.post_act if hasattr(self, 'post_act') else None

        # Construct the profile summary ops
        preact_tensors = self.profile_contrib(preact)
        
        if postact is not None:
            postact_tensors = dict_prefix_key(self.profile_contrib(postact), 'output_')
        else:
            postact_tensors = {}

        if self.activation is None:
            # the post-activation doesn't
            # have any specific meaning when
            # we don't use any activation function
            return preact_tensors

        if preact_only:
            return preact_tensors
        else:
            return {**preact_tensors, **postact_tensors}

    # def get_intp_tensor(self, which='wn'):
    #     return self.intp_tensors()[which]

    def get_bias_input(self, task):
        return self.bias_input.format(task=task)

    def neutral_bias_input(self, task, length, seqlen):
        """Create dummy bias input

        Return: (k, v) tuple
        """
        shape = tuple([x if x is not None else seqlen
                       for x in self.bias_shape])
        return (self.get_bias_input(task), np.zeros((length, ) + shape))
