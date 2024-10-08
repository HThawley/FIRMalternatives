# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 05:21:25 2024

@author: u6942852
"""

import numpy as np 
import pandas as pd
from numba import njit, prange, float64, int64, objmode
from numba.experimental import jitclass
import datetime as dt
from tqdm import tqdm
import warnings
from csv import writer
from multiprocessing import cpu_count
from time import sleep 
import shutil
import os 


spec = [
    ('centre', float64[:]),
    ('f', float64),
    ('extras', float64[:]),
    ('parent_f', float64),
    ('half_length', float64[:]),
    ('generation', int64),
    ('cuts', int64),
    ]

@jitclass(spec)
class hyperrectangle():
    def __init__(self, centre, f, generation, cuts, extras, half_length):
        self.centre = centre
        self.f = f
        self.half_length = half_length
        self.generation = generation
        self.cuts = cuts
        self.extras = extras

class Result:
    def __init__(self, x, extras, f, nfev, nit, half_length):
        self.x = x
        self.extras = extras
        self.f = f 
        self.nfev = nfev
        self.nit = nit
        self.half_length = half_length
        self.lb, self.ub = x-half_length, x+half_length
        self.volume = (half_length * 2).prod()

class Spacepartition:
    def __init__(self, 
                 func, 
                 bounds, 
                 vectorizable=False, 
                 nextras=0, 
                 restart='', 
                 printfile='', 
                 disp=True,
                 
                 f_args=(),
                 max_iter=np.inf,
                 max_fev=np.inf,
                 max_dims=-1,
                 max_pop=1,
                 max_res=-np.inf,
                 near_optimal=np.inf,
                 ):
        
        self.func = func
        self.bounds = self.lb , self.ub = bounds
        self.vectorizable = vectorizable
        self.nextras = nextras
        self.restart = restart
        self.printfile = printfile
        self.disp = disp
        
        self.ndim = len(self.lb)
        self.maxparents = int(3_000_000/self.ndim)
        
        self.f_args = f_args
        self.max_iter = max_iter
        self.max_fev = max_fev
        self.max_dims = max_dims if max_dims != -1 else self.ndim
        self.max_pop = max_pop
        self.max_res = max_res
        self.near_optimal = near_optimal

        # thresholds at which we will estimate time of long steps
        self.cpu = cpu_count()
        self.cmtt = self.cpu * 1500 / self.ndim # cpu-multiple timer threshhold
        self.cctt = 50e10 / self.ndim # comparison count timer threshold
        self.i, self.conv_count, self.miter_adj, self.mfev_adj, self.np, self.fev = 0, 0, 0, 0, 0, 1
        self.noptimal_threshold = np.inf
        
        if self.vectorizable:
            if self.nextras > 0:
                self._dividefunc = _divide_vec_extra
            else: 
                self._dividefunc = _divide_vec
        else: 
            if self.nextras > 0:
                self._dividefunc = _divide_mp_extra
            else:
                self._dividefunc = _divide_mp
        
        centre = 0.5*(self.ub - self.lb) + self.lb
        
        if self.vectorizable: 
            f = self.func(np.atleast_2d(centre).T, *self.f_args)[0]
        else: 
            f = self.func(centre, *self.f_args)
        if self.nextras > 0:
            f, extras = f[0], f[1:]
        else:
            extras = np.array([], np.float64)
        
        self.elite = hyperrectangle(centre, f, -1, 0, extras, (self.ub-self.lb)/2)
        self.childless = np.array([self.elite], dtype=hyperrectangle)
        
        self.new_resolved = np.array([], dtype=hyperrectangle)
        self.edge_resolved = np.array([], dtype=hyperrectangle)
        self.ll_resolved = np.array([], dtype=hyperrectangle)
    
    def Initiate(self):
        if self.restart == '':
            for file in ('parents', 'children', 'resolved'):
                self._printout(np.array([]), file, 'w')
        else:
            self._restart()

    def Step(self, step_dict):
        self._parse_step_dict(step_dict)
               
        self._iterate()
        
    def Polish(self, step_dict={}):
        if 'max_res' in step_dict.keys():
            assert (step_dict['max_res'] == self.max_res).all()
        self._parse_step_dict(step_dict)
        
        self._polish()
        
    def ReturnElite(self):
        return Result(self.elite.centre, self.elite.extras, self.elite.f, 
                      self.fev, self.i, self.elite.half_length)       
        
        
    def _parse_step_dict(self, step_dict):
        keys = step_dict.keys()
        setdiff = set(keys) - set(('f_args', 'max_iter', 'max_fev', 'max_dims', 
                                   'max_pop', 'max_res', 'near_optimal'))
        assert setdiff == set(), f"invalid keywords: {list(setdiff)}"

        self.f_args   = step_dict['f_args']   if 'f_args'   in keys else self.f_args
        self.max_iter = step_dict['max_iter'] if 'max_iter' in keys else self.max_iter
        self.max_fev  = step_dict['max_fev']  if 'max_fev'  in keys else self.max_fev
        self.max_dims = step_dict['max_dims'] if 'max_dims' in keys else self.max_dims
        self.max_pop  = step_dict['max_pop']  if 'max_pop'  in keys else self.max_pop
        self.near_optimal = step_dict['near_optimal'] if 'near_optimal' in keys else self.near_optimal

        self.max_dims = self.max_dims if self.max_dims != -1 else self.ndim        
        assert self.max_dims <= self.ndim
        
        self.dims = np.arange(self.max_dims, dtype=np.int64)
        self.max_rot = self.ndim // self.max_dims + min(1, self.ndim % self.max_dims)

        recalc = False if 'max_res' not in keys else not (self.max_res == step_dict['max_res']).all()
        if recalc or self.i == 0:
            self.max_res = step_dict['max_res']  if 'max_res' in keys else self.max_res
            self.min_half_length = self.max_res/2.

            # only update if necessary (can be slow)
            self._update_resolution()
        
        
    def _update_resolution(self):
        self.childless = np.concatenate((self.childless, 
                                         self.edge_resolved, 
                                         self.new_resolved,
                                         self.ll_resolved))
        self.new_resolved = np.array([], dtype=hyperrectangle)
        if len(self.childless) > 0:
            resolved_mask = semibarren_speedup(list(self.childless), np.ones(self.ndim, dtype=np.bool_), self.min_half_length) 
            self.new_resolved = self.childless[resolved_mask]
            self.childless = self.childless[~resolved_mask]
            
        self.ll_resolved = np.array([], dtype=hyperrectangle)
        self.edge_resolved = np.array([], dtype=hyperrectangle)

        self._printout(self.childless, 'children', 'w')
        self._printout(self.new_resolved, 'resolved', 'w')
        
    
        
    def _iterate(self):
        miter = self.max_iter + self.i
        mfev = self.max_fev + self.fev
        self.nrotate = 0
        while (self.i < miter and self.fev < mfev and self.nrotate < self.max_rot and len(self.childless)>0):
            it_start = dt.datetime.now()
            
            self._sort_resolved()
            self._choose_parents()

            # evaluate new rectangles 
            self.new_hrects = np.array([hrect for parent in 
                                        tqdm(self.parents, desc=f'it {self.i} - #hrects: {self.np}. Evaluating Rectangles', leave=False)
                                        for hrect in self._dividefunc(self.func, parent, self.dims, self.f_args, self.min_half_length, self.nextras)])
            
            self._sort_new_children()
            self._printout(self.parents, 'parents', 'a')
            self._printout(self.new_resolved, 'resolved', 'a')
            self._printout(self.childless, 'children', 'w')
            
            it_time = dt.datetime.now() - it_start
            print(' '*160, end='\r', flush=True)
            if self.disp is True: 
                print(f'it {self.i} - #hrects: {self.np}. Took: {it_time}. Best value: {self.elite.f}.', flush=True)

            self._rotate_axes()
            self.i += 1
            
    def _rotate_axes(self):
        # rotate splitting axes
        self.dims += self.max_dims
        self.dims %= self.ndim
    
    def _sort_resolved(self):
        """Sort resolved rectangles into edge and landlocked (ll)"""
        if len(self.new_resolved) == 0:
            return
        self.edge_resolved = np.concatenate((self.edge_resolved, 
                                             self.new_resolved))
        self.all_resolved = np.concatenate((self.edge_resolved, 
                                            self.ll_resolved))
        self.new_resolved = np.array([], dtype=hyperrectangle)
        
        non_res = np.uint64(len(self.childless))
        nresolved = np.uint64(len(self.all_resolved))
        nedge = np.uint64(len(self.edge_resolved))
        
        print(' '*160, '\r', f'it {self.i} - Sorting resolved points. Estimated time: ', sep='', end='', flush=True)  
        # select classification method based on approx no. of comparisons required
        if nedge * nresolved < nedge * non_res or len(self.childless)==0: 
            _sort_func = self._sort_by_sum
        else: 
            _sort_func = self._sort_by_contra
    
        ll_mask = self._time_long_func(_sort_func, np.ones(len(self.edge_resolved), dtype=np.bool_), nedge, nedge*nresolved)
        self.ll_resolved = np.concatenate((self.ll_resolved,
                                           self.edge_resolved[ll_mask]))
        self.edge_resolved = self.edge_resolved[~ll_mask]
        self.all_resolved = np.array([], dtype=hyperrectangle) #ease memory burden
    
    def _sort_noptimal(self):
        """Sort noptimal rectangles into edge and landlocked (ll)"""
        if len(self.new_noptimal) == 0:
            return
        
        self.edge_noptimal = np.concatenate((self.edge_noptimal, 
                                             self.new_noptimal))
        self.all_noptimal = np.concatenate((self.edge_noptimal, 
                                            self.ll_noptimal))
        self.new_noptimal = np.array([], dtype=hyperrectangle)
        
        nnoptimal = np.uint64(len(self.all_noptimal))
        nedge = np.uint64(len(self.edge_noptimal))
        
        print(' '*160, '\r', f'it {self.i} - Sorting near-optimal points. Estimated time: ', sep='', end='', flush=True)  

        _sort_func = self._sort_nop_by_contra # by sum not available as bounds not known like that
        ll_mask = self._time_long_func(_sort_func, np.ones(len(self.edge_noptimal), dtype=np.bool_), nedge, nedge*nnoptimal)
        self.ll_noptimal = np.concatenate((self.ll_noptimal, self.edge_noptimal[ll_mask]))
        
        self.edge_noptimal = self.edge_noptimal[~ll_mask]
        self.all_noptimal = np.array([], dtype=hyperrectangle) #ease memory burden
        
    def _get_noptimal(self):
        fs = np.array([h.f for h in self.childless], dtype=np.float64)
        self.childless = self.childless[fs.argsort()]
        
        nearoptimalcount = (fs <= self.noptimal_threshold).sum()
        
        best = np.concatenate((np.ones(nearoptimalcount, dtype=np.bool_), 
                               np.zeros(len(self.childless) - nearoptimalcount, dtype=np.bool_)))
        if best.sum() > 0:
            # only rectangles which can be split on current splitting axes
            best = ~semibarren_speedup(list(self.childless[best]), self.dims, self.min_half_length)
            # append 0s to best to match length of childless array
            best = np.concatenate((best, np.zeros(len(self.childless) -len(best), dtype=np.bool_)))
        return best

    def _get_nearoptimal_neighbours(self):
        self.noptimal_resolved = np.array([h.f < self.noptimal_threshold for h in self.edge_resolved])
        if self.noptimal_resolved.sum() > 0 and len(self.childless) > 0:
            # rectangles which cannot be split on the axes are ineligible
            self.eligible = ~semibarren_speedup(list(self.childless), self.dims, self.min_half_length)
            if self.eligible.sum() > 0:
                # rectangles failing beyond maximal extent of near-optimal resolved are ineligible
                self.eligible[self.eligible] = ~_borderheuristic(list(self.childless[self.eligible]), 
                                                                 list(self.edge_resolved[self.noptimal_resolved]))
            print(' '*160, '\r', f'it {self.i} - Identifying near-optimal neighbours. Estimated time: ', sep='', end='', flush=True)  
        
            best = self._time_long_func(self._find_neighbours_parent, self.eligible, self.eligible.sum(), self.eligible.sum()*self.noptimal_resolved.sum())
            self.eligible[self.eligible] = best
            return self.eligible
        else:
            return np.zeros(len(self.childless), np.bool_)
    
    def _get_polishing_neighbours(self):
        if len(self.edge_noptimal) > 0 and len(self.childless) > 0:
            # rectangles which cannot be split on the axes are ineligible
            self.eligible = ~semibarren_speedup(list(self.childless), self.dims, self.min_half_length)
            if self.eligible.sum() > 0:
                # rectangles failing beyond maximal extent of near-optimal resolved are ineligible
                self.eligible[self.eligible] = ~_borderheuristic(list(self.childless[self.eligible]), 
                                                                 list(self.edge_noptimal))
            print(' '*160, '\r', f'it {self.i} - Identifying near-optimal neighbours. Estimated time: ', sep='', end='', flush=True)  
        
            best = self._time_long_func(self._find_neighbours_polish, self.eligible, self.eligible.sum(), self.eligible.sum()*len(self.edge_noptimal))
            self.eligible[self.eligible] = best
            return self.eligible
        else:
            return np.zeros(len(self.childless), np.bool_)
    
    def _choose_parents(self):
        print(' '*160, '\r', f'it {self.i} - Identifying parents...', end='\r', sep='', flush=True)
        # boolean mask of near-optimal rectangles
        best = self._get_noptimal()
            
        # If no near-optimal rectangles to be split, find neighbours
        if best.sum() == 0:
            best = self._get_nearoptimal_neighbours()
        
        # limit number of rectangles 
        best[find_bool_indx(best, min(self.max_pop, self.maxparents)):] = False

        if best.sum() == 0: # no near-optimal rectangles to be split
            self.nrotate +=1 # as this accumulates, it will stop iterating when nothing to be split
        else:
            self.nrotate = 0
        
        self.parents = self.childless[best]
        self.childless = self.childless[~best]
        self.np = len(self.parents)
        
    def _update_elite(self):
        #update elite
        fs = np.array([h.f for h in self.new_hrects])
        if fs.min() < self.elite.f:
            self.elite = self.new_hrects[fs.argmin()]
        del fs
        self.noptimal_threshold = self.elite.f*self.near_optimal
        
        
    def _sort_new_children(self):
        lnh = len(self.new_hrects)
        self.fev += lnh
        
        if lnh > 0:
            print(' '*160, '\r', f'it {self.i} - Sorting new children...', sep='', end='\r', flush=True)

            self._update_elite()

            # identify resolved rectangles 
            self.resolved_mask = semibarren_speedup(list(self.new_hrects), np.ones(self.ndim, dtype=np.bool_), self.min_half_length)
            
            # sort new into childless and resolved
            self.new_resolved = self.new_hrects[self.resolved_mask]
            self.childless = np.concatenate((self.new_hrects[~self.resolved_mask], 
                                             self.childless))
            self.new_hrects = np.array([], dtype=hyperrectangle)
            
        else:
            self.resolved_mask = np.array([], dtype=np.bool_)
        
    def _find_neighbours_parent(self, mask):
        return find_neighbours(
            list(self.childless[self.eligible * mask]), 
            list(self.edge_resolved[self.noptimal_resolved])
            )
    
    def _find_neighbours_polish(self, mask):
        return find_neighbours(
            list(self.childless[self.eligible * mask]), 
            list(self.edge_noptimal)
            )
    
    def _sort_by_sum(self, mask):
        return landlocked_bysum(
            list(self.edge_resolved[mask]),
            list(self.all_resolved),
            self.bounds
            )
        
    def _sort_by_contra(self, mask): 
        return landlocked_bycontra(
            list(self.edge_resolved[mask]),
            list(self.childless)
            )  
        
    def _sort_nop_by_contra(self, mask): 
        return landlocked_bycontra(
            list(self.edge_noptimal[mask]),
            list(self.childless)
            )  
        
    def _time_long_func(self, long_func, base_mask, _cmtt = -1, _cctt = -1):
        if base_mask.sum() == 0:
            return np.zeros(len(base_mask), dtype=np.bool_)
        # determine whether to time or not (default is to time)
        if _cmtt < self.cmtt or _cctt < self.cctt or base_mask.sum() == 1:
            print('< a few minutes. ', end ='\r', flush=True)
            return_mask = long_func(np.ones(len(base_mask), dtype=np.bool_))
        else: 
            # no. of samples to test to estimate time
            ntime = int(self.cmtt/2)
            # this mask is multiplied with the base_mask to select the first {ntime} samples 
            # the inverse is multiple with base_mask to select the remaining samples
            time_mask = np.zeros(len(base_mask), dtype=np.bool_)
            time_mask[:find_bool_indx(base_mask, ntime)] = True
            
            sort_start = dt.datetime.now()
            # evaluate first {ntime}
            return_mask = long_func(time_mask)
            
            sort_time = (dt.datetime.now() - sort_start) * (base_mask.sum() - ntime) / ntime
            print(f'{sort_time}. Estimated end time: {dt.datetime.now() + sort_time}. ', end='\r', flush=True)
            
            # evaluate remaining values
            return_mask = np.concatenate((
                return_mask, 
                long_func(~time_mask)
                ))
        
        return return_mask
        

        
    def _printout(self, arr, suffix, mode='w'):
        """Print out an array of hyperrectangles."""
        if self.printfile != '':
            print(' '*160, '\r', f'it {self.i} - #hrects: {self.np}. Writing out to file. Do not Interrupt.', sep='', end='\r', flush=True)
            path, temppath = f'{self.printfile}-{suffix}.csv', f'{self.printfile}-{suffix}-temp.csv'
            if mode == 'a':
                shutil.copyfile(path, temppath)
            
            with open(temppath, mode, newline='') as csvfile:
                if len(arr) > 0:
                    printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in arr]), 
                                               np.array([h.extras for h in arr]), 
                                               np.array([h.centre for h in arr])),
                                              axis=1)
                    writer(csvfile).writerows(printout)
            shutil.copyfile(temppath, path)
            os.remove(temppath)
        
    def _restart(self):
        if self.disp:
            print('Restarting optimisation where',self.restart,'left off.')
    
        history = pd.read_csv(self.restart+'-children.csv', header=None).to_numpy() # signficantly faster and lower memory than np.genfromtxt
        try: 
            resolved = pd.read_csv(self.restart+'-resolved.csv', header=None).to_numpy()
            if len(resolved) != 0:
                history = np.vstack((history, np.atleast_2d(resolved)))
            del resolved
        except (FileNotFoundError, pd.errors.EmptyDataError):
            warnings.warn("Warning: No resolved file found.", UserWarning)
        try: 
            parents = pd.read_csv(self.restart+'-parents.csv', header=None).to_numpy()
            pmin, pminidx = parents[:,0].min(), parents[:,0].argmin()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pmin, pminidx= np.inf, None
            warnings.warn("Warning: No parents file found.", UserWarning)
        fs, exs, xs = history[:,:3], history[:,3:3+self.nextras], history[:,3+self.nextras:]
        
        xs, hls = _reconstruct_from_centre(xs, self.bounds)
        lxs = len(xs)
        del history
    
        fmin, fmini = fs[:,0].min(), fs[:,0].argmin()
        self.childless = np.array([hyperrectangle(xs[i], *fs[i,:], exs[i], hls[i]) for i in range(lxs)])
        
        if fmin < pmin:
            self.elite = self.childless[fmini]
            del xs, fs, exs, hls
        else: 
            fps, exps, xps = parents[:,:3], parents[:,3:3+self.nextras:], parents[:,3+self.nextras:]
            xps, hlps = _reconstruct_from_centre(np.atleast_2d(xps[pminidx, :]), self.bounds)
            self.elite = hyperrectangle(xps[0,:], *fps[pminidx,:], exps[pminidx,:], hlps.flatten())
            del fps, xps, hlps, parents
    
        if self.disp:
            print(f'Restart: read in {lxs} rectangles. Best value: {self.elite.f}.')
        
    def _polish(self):
        self.min_half_length/=2
        self.nrotate, i = 0, 0
        self.i = f'Polishing {i}'
        self.max_pop = np.inf
        self.max_dims = 1
        self.max_rot = self.ndim // self.max_dims + min(1, self.ndim % self.max_dims)
        self.dims = np.array([0], np.int64)
        self.nrotate = 0
        
        for file in ('polished', 'pol-child'):
            self._printout(np.array([]), file, 'w')

        # Polish should be last step and the near-optimal space + neighbours should be fully resolved. 
        self.childless = np.concatenate((self.edge_resolved, self.ll_resolved, self.new_resolved))
        self.noptimal_threshold = self.elite.f*self.near_optimal
        
        # Polish should be last step, therefore we don't need to remember these  
        # delete for memory burden
        self.edge_resolved = np.array([], dtype=hyperrectangle) 
        self.ll_resolved = np.array([], dtype=hyperrectangle) 
        
        while len(self.childless) > 0:
            it_start = dt.datetime.now()
            
            noptimal_mask = self._get_noptimal()
            self.new_noptimal = self.childless[noptimal_mask]
            self.childless = self.childless[~noptimal_mask]
            
            self.edge_noptimal = np.array([], dtype=hyperrectangle) 
            self.ll_noptimal = np.array([], dtype=hyperrectangle) 
            
            self._sort_noptimal()
            self.parents = np.concatenate((self.edge_noptimal, self.ll_noptimal, self.childless[self._get_polishing_neighbours()]))
            self.np = len(self.parents)
            
            self.new_hrects = np.array([hrect for parent in 
                                        tqdm(self.parents, desc=f'{self.i} - #hrects: {len(self.parents)}. Evaluating Rectangles', leave=False)
                                        for hrect in _divide_polish(self.func, parent, self.dims, self.f_args, self.nextras)])
            self.new_hrects = np.concatenate((self.new_hrects, 
                                              np.array([_adjust_polish_parent(parent, self.dims) for parent in self.parents])))
            self._update_elite()
            
            resolved_mask = semibarren_speedup(list(self.new_hrects), np.ones(self.ndim, dtype=np.bool_), self.min_half_length)
            self.new_resolved = self.new_hrects[resolved_mask]
            self.childless = self.new_hrects[~resolved_mask]
            
            self._printout(self.childless, 'pol-child', mode='w')
            self._printout(self.new_resolved, 'polished', mode='a')
            
            it_time = dt.datetime.now() - it_start
            print(' '*160, end='\r', flush=True)
            if self.disp is True: 
                print(f'{self.i} - #hrects: {self.np}. Took: {it_time}. Best value: {self.elite.f}.', flush=True)
            i+=1
            self.i = f'Polishing {i}'
            self._rotate_axes()
                
        print('Completed.')
        
#%% Heavy duty helper functions
@njit
def _common_divide(hrect, dims, min_half_length):
    # pre-processing common to all _divide_... functions
    dims = dims[(hrect.half_length >= min_half_length)[dims]] 

    l_dim = len(dims)
    n_new = 2**l_dim

    centres = generate_centres(hrect, dims)
    hls = hrect.half_length.copy()
    hls[dims] /= 2 
    gen, cuts = hrect.generation + 1, hrect.cuts + l_dim
    
    return centres, hls, gen, cuts, n_new, dims

@njit
def _divide_vec(func, hrect, dims, f_args, min_half_length, nextras):
    centres, hls, gen, cuts, n_new, dims = _common_divide(hrect, dims, min_half_length)
    
    f_values = func(centres.T, *f_args)
        
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, np.array([], np.float64), hls[k]) 
        for k in range(n_new)]

    return hrects

@njit
def _divide_vec_extra(func, hrect, dims, f_args, min_half_length, nextras, lb, ub):
    centres, hls, gen, cuts, n_new, dims = _common_divide(hrect, dims, min_half_length)
    
    f_values = func(centres.T, *f_args)
    f_values, extras = f_values[:,0], f_values[:,1:]
        
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, extras[k], hls[k]) 
        for k in range(n_new)]

    return hrects

@njit(parallel=True)
def _divide_mp(func, hrect, dims, f_args, min_half_length, nextras):
    centres, hls, gen, cuts, n_new, dims = _common_divide(hrect, dims, min_half_length)
    
    f_values = np.empty(n_new, dtype=np.float64)
    for i in prange(n_new):
        f_values[i] = func(centres[i,:], *f_args)
    
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, np.array([], np.float64), hls) 
        for k in range(n_new)]
    return hrects

@njit(parallel=True)
def _divide_mp_extra(func, hrect, dims, f_args, min_half_length, nextras):
    centres, hls, gen, cuts, n_new, dims = _common_divide(hrect, dims, min_half_length)
    
    f_values = np.empty((n_new, nextras+1), dtype=np.float64)
    for i in prange(n_new):
        f_values[i] = func(centres[i,:], *f_args)
    f_values, extras = f_values[:,0], f_values[:, 1:]
    
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, extras[k], hls) 
        for k in range(n_new)]
    return hrects

@njit(parallel=True)
def _divide_polish(func, hrect, dims, f_args, nextras):
    centres, hrect = generate_polish_centres(hrect, dims)
    n_new = len(centres)
    
    hls = hrect.half_length.copy()
    hls[dims] /= 2
    
    f_values = np.empty((n_new, nextras+1), dtype=np.float64)
    for i in prange(n_new):
        f_values[i] = func(centres[i,:], *f_args)
    f_values, extras = f_values[:,0], f_values[:, 1:]
    
    hrects = [hyperrectangle(
        centres[k], f_values[k], -1, -1, extras[k], hls) 
        for k in range(n_new)]
    
    return hrects

@njit
def _adjust_polish_parent(hrect, dims):
    hrect.cuts = -1 
    hrect.generation = -1
    hrect.half_length[dims] /= 2
    return hrect

@njit(parallel=True)
def _reconstruct_from_centre(centres, bounds, maxres=2**31):
    lb, ub = bounds
    centres = normalise(centres, lb, ub)   
    incs = np.round((centres*maxres)).astype(np.uint64)
    incs1 = np.empty_like(incs)
    for i in prange(len(centres)):
        for j in range(centres.shape[1]):
            incs1[i,j] = factor2(incs[i,j])
    half_lengths = incs1 / maxres
    
    centres = unnormalise_c(centres, lb, ub) 
    half_lengths = unnormalise_hl(half_lengths, lb, ub) 
    return centres, half_lengths
    
#%% Hyperrectangle helper funcctions

@njit
def hrect_semibarren(h, dims, min_half_length):
    """Returns True if {h} is at maximum resolution along {dims} axes """
    return (h.half_length <= min_half_length)[dims].all()

@njit(parallel=True)
def semibarren_speedup(rects, dims, min_half_length):
    """Returns boolean array like rects where Trues are at maximum resolution along {dims} axes """
    accepted = np.empty(len(rects), dtype=np.bool_)
    for i in prange(len(rects)):
        accepted[i] = hrect_semibarren(rects[i], dims, min_half_length)
    return accepted

@njit(parallel=True)
def _borderheuristic(rects, best):
    """Returns boolean array like rects where Trues are definitely non-adjacent to a rectangle in best. 
    Determined by having more than 1 dimension with upper bound less than minimal lower bound of best or 
    lower bound greater than maximal upper bound of best"""
    minlb =  np.inf*np.ones(len(best[0].centre), dtype=np.float64)
    maxub = -np.inf*np.ones(len(best[0].centre), dtype=np.float64)
    
    for i in range(len(best)): 
        minlb = np.minimum(minlb, best[i].centre-best[i].half_length)
        maxub = np.maximum(maxub, best[i].centre+best[i].half_length)
    
    rejected = np.empty(len(rects), dtype=np.bool_)
    for i in prange(len(rects)):
        rejected[i] = ((rects[i].centre-rects[i].half_length >= maxub).sum() + 
                       (rects[i].centre+rects[i].half_length <= minlb).sum() > 1)

    return rejected

@njit(parallel=True)
def find_neighbours(eligible, members):
    accepted=np.zeros(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        h = eligible[i]
        for r in members:
            if hrects_border(h, r):
                accepted[i] = True
                break
    return accepted

@njit 
def _sub_landlocked_bysum(h, members, bounds):
    """ Returns True if h is landlocked by pool """
    """ Assumes h is of the smallest resolution in archive """
    faces = len(h.centre)*2 - (h.centre-h.half_length == bounds[0]).sum() - (h.centre+h.half_length == bounds[1]).sum()
    for h2 in members:
        if hrects_border(h, h2):
            faces -= 1 
        if faces == 0:
            return True
    return False
            
@njit(parallel=True)
def landlocked_bysum(eligible, members, bounds):
    accepted = np.empty(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        accepted[i] = _sub_landlocked_bysum(eligible[i], members, bounds)
    return accepted 

@njit 
def _sub_landlocked_bycontra(h, nonmembers):
    """ Returns True if h is landlocked by pool """
    for h2 in nonmembers: 
        if hrects_border(h, h2):
            return False
    return True

@njit(parallel=True)
def landlocked_bycontra(eligible, nonmembers):
    accepted = np.empty(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        accepted[i] = _sub_landlocked_bycontra(eligible[i], nonmembers)
    return accepted

@njit #parallel is slower for ndim range
def generate_centres(hrect, dims):
    l = (hrect.centre - hrect.half_length/2)[dims]
    u = (hrect.centre + hrect.half_length/2)[dims]
    
    indcs = generate_boolmatrix(len(dims))
    
    centres = np.repeat(hrect.centre, len(indcs)).reshape((len(hrect.centre), len(indcs))).T
    for i in prange(len(indcs)):
        base = np.empty(len(dims), dtype=np.float64)
        ind = indcs[i,:]
        inv = ~ind
        base[ind], base[inv] = l[ind], u[inv]
        centres[i, dims] = base
    return centres

@njit #parallel is slower for ndim range
def generate_polish_centres(hrect, dims):
    centres = generate_centres(hrect, dims) 
    n_new = 2**len(dims)
    
    hl_adj = hrect.half_length[dims] / 2
    oob = np.empty(n_new, dtype=np.bool_)
    for i in range(n_new):
        centres[i, dims] -= hl_adj
        oob[i] = (centres[i] == hrect.centre).all()
    
    centres = centres[~oob, :]
    return centres, hrect

@njit
def hrects_border(h1, h2, tol = 1e-10):
    lb1, ub1 = h1.centre - h1.half_length, h1.centre + h1.half_length
    lb2, ub2 = h2.centre - h2.half_length, h2.centre + h2.half_length
    ndim = len(lb1)
    
    # axes in which domain of h1 wholly contains or is wholly inside of domain of h2 
    overlaps = (lb1 >= lb2 - tol) * (ub1 <= ub2 + tol) + (lb1 <= lb2 + tol) * (ub1 >= ub2 - tol)
    if overlaps.sum() != ndim-1:
        return False
    
    # adjacent (ub=lb or lb=ub) (higher OR lower) in exactly one dimension
    adjacency = ((np.abs(ub2 - lb1) < tol) +
                 (np.abs(lb2 - ub1) < tol)) 
    if adjacency.sum() != 1:
        return False
    
    # adjacent on dimensions which do not overlap
    if not (adjacency == ~overlaps).all():
        return False
    
    return True

#%% Light duty helper functions
@njit
def find_bool_indx(mask, count):
    """returns the index of the boolean mask such that there are {count} Trues 
    before it. If there are less than {count} Trues in the array, returns -1. 
    Not valid for count<=0 """
    if count >= len(mask) or count >= mask.sum():
        return len(mask)
    _mask_indx, _counter = -1, 0
    while _counter < count:
        _mask_indx+=1
        if mask[_mask_indx]:
            _counter += 1
    return _mask_indx + 1

@njit
def normalise(arr, lb, ub):
    return (arr-lb)/(ub-lb)

@njit
def unnormalise_c(arr, lb, ub):
    return arr*(ub-lb) + lb

@njit
def unnormalise_hl(arr, lb, ub):
    return arr*(ub-lb)

@njit
def factor2(n):
    if n==0: 
        return 0
    i=0
    while n%(2**(i+1)) != 2**i:
        i+=1
    return 2**i
    
@njit #parallel is slower for ndim range
def generate_boolmatrix(ndim):
    _2ndim = 2**ndim
    z = np.empty((_2ndim, ndim), dtype=np.bool_)
    for i in prange(ndim):
        base = np.zeros(_2ndim, dtype=np.bool_)
        i2 = 2**i
        for j in range(0, _2ndim, i2*2):
            base[j:j+i2] = True
        z[:,i] = base
    return z

@njit
def signs_of_array(arr, tol=1e-10):
    arr[np.abs(arr)<tol] = 0 
    return np.sign(arr)