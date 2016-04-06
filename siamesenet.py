__authors__='Matt Tudor'
import pylearn2
import functools
import itertools
import os, cPickle
import numpy as np
import math
import h5py
import multiprocessing
import pylearn2.utils.iteration
from collections import defaultdict
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.compat import OrderedDict
import pylearn2.models.mlp
from pylearn2.utils.rng import make_np_rng
import pdb
import theano
from theano import tensor as T
from pylearn2.utils import wraps
from pylearn2.expr.nnet import (compute_precision,compute_recall,compute_f1)


#################################################siamese network###################################################################


class GenerateAdjacentPairsDataset(dense_design_matrix.DenseDesignMatrix):
    '''
    read individual examples along with grouping variable from file,
    generate paris labeled as 1/0 if derived from same/different groups
    '''
    def __init__(self,xfile=None,grpfile=None,which_set='train',start=0,stop=None,validfrac=0.1,testfrac=0.1,repfrac=.33,balanceclasses=True):
      ''' 
      xfile: csv file with one row per sample
      grpfile: single column file with grouping as numeric ids, hierarchical annotation possible:
      	train/test/valid split on int(grp), while replicates annotated on basis of str(grp)
      whichset: 'train', 'valid', or 'test'
      start and stop: indicies to truncate iteration of samples
      validfrac and testfrac: fraction of samples to be used for validation and testing
      repfrac: (min) fraction of replicates (i.e. derived from same group) desired in ouput dataset, with balance being nonreplicates
      balanceclasses: if True, replicates are repeated to match number of nonreplicates
      '''

      self.dtype='float32'
      self.rng=np.random.RandomState(self._default_seed)
      self.which_set=which_set
      assert xfile is not None
      self.xfile=xfile
      if grpfile is None: #grouping is first column of x
          grpinx=True
          print "using first column of "+xfile+" as grouping variable"
      else:
          self.grpfile=y=grpfile
          grpinx=False
      #read data in
      self.Xin=np.loadtxt(xfile,dtype='float32',delimiter=',')
      if grpinx:
          self.grp=self.Xin[:,0].astype('str')
          self.Xin=self.Xin[:,1:]
      else:
          self.grp=np.loadtxt(grpfile,dtype='str',delimiter=',')    
      #use which_set to select slice of groups
      uniquegroups=np.unique([int(float(x)) for x in self.grp])
      lu=len(uniquegroups)
      numtest=int(lu*testfrac)
      numvalid=int(lu*validfrac)
      numtrain=lu-numtest-numvalid
      np.random.seed(313)
      shufgroups=np.random.permutation(len(uniquegroups))
      if which_set=='train':
          ugrpsamp=shufgroups[:numtrain]
      elif which_set=='test':
          ugrpsamp=shufgroups[numtrain:(numtrain+numtest)]
      elif which_set=='valid':
          ugrpsamp=shufgroups[(numtrain+numtest):]
      else:
          raise ValueError('which_set must be "train", "test", or "valid"')
      #update X,grp
      ugrp=uniquegroups[ugrpsamp]
      self.dataidx=[i for i,j in enumerate(self.grp) if int(float(j)) in ugrp]
      self.Xin=self.Xin[self.dataidx,:]
      self.grp=self.grp[self.dataidx]
      #cartesian product of replicates (i.e. samples sharing grp id)
      ugrpcnt=defaultdict(int)
      for i in self.grp:
          ugrpcnt[i]+=1
      grphasreps=[k for k,v in ugrpcnt.items() if v>1]
      reppairs=[]
      for r in grphasreps:
          inds=[i for i,j in enumerate(self.grp) if j==r]
          if len(inds)>10:  #don't oversample large groups
              samp=np.random.random_integers(len(inds),size=10)
              inds=[i for i in inds if i in samp]
          prs=[k for k in itertools.permutations(inds,2)]
          reppairs.extend(prs)
      numreppairs=len(reppairs)
      #nonreplicates
      numnonreppairs=int(numreppairs*(1-repfrac)/(repfrac))
      #balance class sizes 
      if balanceclasses:
        reppairmultiple=max(1,numnonreppairs/numreppairs -1)
      else: 
        reppairmultiple=1
      nonreppairs=[tuple(np.random.random_integers(len(self.grp)-1,size=2)) for i in range(2*numnonreppairs) ]
      #eliminate within-group pairs
      repdict={i:0 for i in reppairs} #for fast lookup of reppairs
      nonreppairs=[i for i in nonreppairs if i not in repdict]  #not a reppair
      #eliminate pairs with same int(grp) 
      nonreppairs=[i for i in nonreppairs if int(float(self.grp[i[0]]))!=int(float(self.grp[i[1]])) ]
      nonreppairs=nonreppairs[:numnonreppairs]
      numnonreppairs=len(nonreppairs)
      #repeat reppairs to balance number of nonreppairs, concatenate replicates and nonreplicates, generate corresponding 0/1 response
      pairs=reppairs*reppairmultiple+nonreppairs
      resps=np.append(np.ones(numreppairs*reppairmultiple,dtype=self.dtype),np.zeros(numnonreppairs,dtype=self.dtype))
      #shuffle pairs and response, truncate to round number to match batch sizes
      shuffidx=np.random.permutation(len(resps)) 
      shuffidx=shuffidx[:(500*int(len(shuffidx)/500))]
      pairs=[pairs[i] for i in shuffidx]
      resps=resps[shuffidx]
      #flatten X pairs, duplicate responses to match
      self.Xidx=list(itertools.chain.from_iterable(pairs))
      self.y=np.array(list(itertools.chain.from_iterable([(x,x) for x in resps])))
      assert len(self.Xidx)==len(self.y)
      
      #start,stop
      if start>0 or stop is not None:
        assert stop>start
        if stop > len(self.Xidx):
           raise ValueError('stop='+str(stop)+'>'+'m='+str(len(self.Xidx)))
        self.Xidx=[j for i,j in enumerate(self.Xidx) if i>=start and i<stop]
        self.y=self.y[start:stop]
      '''
      with some work, one could avoid carrying the entire (highly 
      redundant) X matrix in memory and instead generate batches of 
      data on demand by using chunks of Xidx to index Xin...
      TODO
      '''
      self.X=self.Xin[self.Xidx]
      print "dataset "+which_set+" size: "+str(self.X.shape)    
      self.X_topo_space = None
      X_space = VectorSpace(dim=self.X.shape[1],dtype=self.dtype)
      X_source = 'features'

      assert self.y.ndim == 1
      #densedesignmatrix expects 2d matrix, cast y as nx1 matrix
      self.y=self.y.reshape((-1,1))
      y_space = VectorSpace(dim=1,dtype=self.dtype) 
      y_source = 'targets'
      space = CompositeSpace((X_space, y_space))
      source = (X_source, y_source)
      self.data_specs = (space, source)
      self.X_space = X_space
      self._iter_mode = 'randompairs'
      self._iter_subset_class = self._iter_mode
      self._iter_topo = False
      self._iter_targets = False
      self._iter_data_specs = (self.X_space, 'features')
    def iterator(self,mode=None,batch_size=None,num_batches=None,topo=None,targets=None,rng=None,data_specs=None,return_tuple=None,convert=None):
      return pylearn2.utils.iteration.FiniteDatasetIterator(self,RandomPairsIterator(self.X.shape[0],batch_size,num_batches,rng),data_specs=data_specs,return_tuple=return_tuple,convert=convert)  

class AdjacentPairsDataset(dense_design_matrix.DenseDesignMatrix):
    #reads data from xfile (features) and yfile (targets), 
    #data is organized to have pairs on adjacent lines, i.e. lines 0 and 1 represent the first pair, 2 and 3 second, etc.
    def __init__(self,xfile=None,yfile=None,which_set='train',start=0,stop=None):
      self.dtype='float32'
      self.rng=np.random.RandomState(self._default_seed)
      self.which_set=which_set

      if xfile is None:
          if which_set=='train':
             xfile='adjacentpairsTrX.csv'
          elif which_set=='valid':
             xfile='adjacentpairsValX.csv'
          elif which_set=='test':
             xfile='adjacentpairsTstX.csv'
          else:
              raise ValueError('only "train"ing, "test"ing, and "valid"ation datasets available')
      if yfile is None:
          if(which_set=='train'):
             yfile='adjacentpairsTrY.csv'
          elif which_set=='valid':
             yfile='adjacentpairsValY.csv'
          elif which_set=='test':
             yfile='adjacentpairsTstY.csv'
      self.xfile=xfile
      self.yfile=yfile
      #read data in
      self.X=np.loadtxt(xfile,dtype='float32',delimiter=',')
      self.y=np.loadtxt(yfile,dtype='float32',delimiter=',')    
      #target values should be repeated for every pair of lines:
      assert np.all(self.y[::2]==self.y[1::2])  
      if start>0 or stop is not None:
        assert stop>start
        if stop > self.X.shape[0]:
           raise ValueError('stop='+str(stop)+'>'+'m='+str(self.X.shape[0]))
        self.X=self.X[start:stop,:]
        self.y=self.y[start:stop]
          
      self.X_topo_space = None
      X_space = VectorSpace(dim=self.X.shape[1],dtype=self.dtype)
      X_source = 'features'

      assert self.y.ndim == 1
      #densedesignmatrix expects 2d matrix, cast y as nx1 matrix
      self.y=self.y.reshape((-1,1))
      y_space = VectorSpace(dim=1,dtype=self.dtype) 
      y_source = 'targets'
      space = CompositeSpace((X_space, y_space))
      source = (X_source, y_source)
      self.data_specs = (space, source)
      self.X_space = X_space
      self._iter_mode = 'randompairs'
      self._iter_subset_class = self._iter_mode
      self._iter_topo = False
      self._iter_targets = False
      self._iter_data_specs = (self.X_space, 'features')
    def iterator(self,mode=None,batch_size=None,num_batches=None,topo=None,targets=None,rng=None,data_specs=None,return_tuple=None,convert=None):
      return pylearn2.utils.iteration.FiniteDatasetIterator(self,RandomPairsIterator(self.X.shape[0],batch_size,num_batches,rng),data_specs=data_specs,return_tuple=return_tuple,convert=convert)  
        
class SiameseMetric(pylearn2.models.mlp.Linear):
  """
  distance metric learning based on siamese network.  rather than explicit siamese network, a single network is trained
  using adjacent examples to represent pairs with 0/1 target for dissimilar/similar
  refs:  
    cauchy & gaussian metrics: Liu, C.  'A probabilistic siamese network for learning representations.'  Masters thesis, University of Toronto, 2013.
    margin: ...
  TODO details, refs
  """
  def __init__(self,layer_name,costfn='margin',costparam=2,constw=False,**kwargs):
    self.dtype='float32'
    self.costfn=costfn
    self.costparam=theano.shared(costparam)
    self.output_space=VectorSpace(dim=1,dtype=self.dtype) 
    self.use_bias=False
    self.layer_name=layer_name
    self.dim=1
    self.constw=constw
    super(SiameseMetric,self).__init__(self.dim,layer_name,**kwargs)
  
  @wraps(pylearn2.models.mlp.Layer.fprop)  
  def fprop(self,state_below):
    self.input_space.validate(state_below)
    x1=state_below
    x2ind=np.arange(self.mlp.batch_size)
    x2ind=[a for b in zip(x2ind[1::2],x2ind[::2]) for a in b]
    x2=T.set_subtensor(x1[x2ind],x1)  #reorder pairs to register rows 1:2, 2:1, 3:4, 4:3, etc.
    difs=T.sqr(x1-x2)
    W, = self.transformer.get_params()
    #dists=T.dot(difs,T.sqr(W))   #nonnegative weights           
    dists=T.dot(difs,W)              
    return dists
  
  @wraps(pylearn2.models.mlp.Layer.cost)
  def cost(self,Y,Y_hat):
    #Y is 0/1 dissim/sim indicator, Y_hat is squared distance metric
    if self.costfn=='margin':
      Y=2.*Y-1.  #0/1 dissim/sim to -1/1  for loss calc
      cost=T.nnet.softplus(1.-Y*(self.costparam-Y_hat))
      #cost=cost*(cost>0)
    elif self.costfn=='cauchy':
      Y_hat=2./(1.+T.exp(Y_hat))
      cost=T.sqr(Y-Y_hat)
    elif self.costfn=='gaussian':
      Y_hat=T.exp(-Y_hat)
      cost=T.sqr(Y-Y_hat)
    elif self.costfn=='linear':
      Y=1.-Y     #0/1 dissim/sim to 1/0 distance for loss calc
      cost=T.sqr(Y-Y_hat)
    return cost.mean()
    
  @wraps(pylearn2.models.mlp.Linear._modify_updates)
  def _modify_updates(self,updates):
    W, = self.transformer.get_params()
    if W in updates:
      updates[W]=T.clip(updates[W],0,np.Inf)
      #updates[W]=T.nnet.softplus(updates[W])
    if self.constw:  # a hack, probably better to freeze layer?
      updates[W]=T.ones_like(updates[W])
    
  def modify_updates(self,updates):  #necessary?
    self._modify_updates(updates)

  def get_layer_monitoring_channels(self,state_below=None,state=None,target=None):
    rval=OrderedDict()
    W,=self.transformer.get_params()
    rval['norm']=T.sqrt(T.sqr(W).sum())
    if(target is not None) and ((state_below is not None) or (state is not None)):
        if state is None:
            state=self.fprop(state_below)
        target=1.-target  #0/1 dissim/sim to 1/0 distances
        rmse=T.sqrt(T.mean(T.sqr(state-target)))
        rval['rmse']=rmse.mean()
        if self.costfn=='margin':
            thresh=self.costparam
        elif self.costfn=='cauchy':
            thresh=2./(1.+T.exp(self.costparam))
        else:
            thresh=0.5
        yhat=state<thresh
        y=target<0.5
        wrong_bit=T.cast(T.neq(y,yhat),state.dtype)
        rval['01_loss']=wrong_bit.mean()

        y=T.cast(y,state.dtype)
        yhat=T.cast(yhat,state.dtype)
        tp=(y*yhat).sum()
        fp=((1-y)*yhat).sum()
        prec=compute_precision(tp,fp)
        rec=compute_recall(y,tp)
        f1=compute_f1(prec,rec)
        rval['neg_precision']=-prec
        rval['neg_recall']=-rec
        rval['neg_f1']=-f1
        return rval

class BinarySigmoid(pylearn2.models.mlp.Sigmoid):
    '''
    thresholded sigmoid unit
    '''
    def fprop(self,state_below):
        p=super(BinarySigmoid,self).fprop(state_below)
        p=T.cast(p>0.5,p.dtype)
        return p

class NoisySigmoid(pylearn2.models.mlp.Sigmoid):
    '''
    noisy sigmoid unit
    '''
    def __init__(self,noisesd=1.,**kwargs):
        super(NoisySigmoid,self).__init__(**kwargs)
        self.noisesd=noisesd
        self.noiserng=T.shared_randomstreams.RandomStreams(seed=1234)

    def fprop(self,state_below):
        z=super(NoisySigmoid,self)._linear_part(state_below)
        z+=self.noiserng.normal((self.mlp.batch_size,1))*self.noisesd
        return T.nnet.sigmoid(z)

    def get_layer_monitoring_channels(self,state_below=None,state=None,targets=None):
        rval=OrderedDict()
        if state is None and state_below is not None:
            state=self.fprop(state_below)
        if state is not None:
            rval=super(NoisySigmoid,self).get_layer_monitoring_channels(state_below=state_below,state=state,targets=targets)
            lt=T.cast(state<0.1,state.dtype).mean(axis=0)
            gt=T.cast(state>0.9,state.dtype).mean(axis=0)
            mid=T.cast(T.abs_(state-0.5)<0.2,state.dtype).mean(axis=0)
            rval['fract_lt0.1_mean']=lt.mean()
            rval['fract_gt0.9_mean']=gt.mean()
            rval['fract_mid0.4_mean']=mid.mean()
            rval['fract_mid0.4_sd']=mid.std()
        return rval

class RandomPairsIterator(pylearn2.utils.iteration.SequentialSubsetIterator):
    """
    Selects minibatches of examples by shuffling data, keeping adjacent pairs together

    """
    stochastic=False  #as far as pylearn2 is concerned...
    fancy=True
    uniform_batch_size=False

    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        self._rng=make_np_rng(rng,which_method=['random_integers','shuffle'])
        if dataset_size % 2 > 0:
            raise ValueError('dataset size must be even number ')
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        if num_batches is None:
            num_batches=int(np.ceil((1.*dataset_size)/batch_size))
        self._num_batches = num_batches
        self._batch=0
        self._idx=0
        self._next_batch_no = 0
        evens=np.arange(dataset_size/2)*2
        self._rng.shuffle(evens)
        odds=evens+1
        self._shuffled=[a for b in zip(evens,odds) for a in b]

    @wraps(pylearn2.utils.iteration.SubsetIterator.next)
    def next(self):
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fixes the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            rval = self._shuffled[self._idx: self._dataset_size]
            self._idx = self._dataset_size
            return rval
        else:
            rval = self._shuffled[self._idx: self._idx + self._batch_size]
            self._idx += self._batch_size
            self._batch += 1
            return rval

    def __next__(self):
        return self.next()

  
from pylearn2.utils.iteration import _iteration_schemes
_iteration_schemes['randompairs']=RandomPairsIterator

