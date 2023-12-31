from multiprocessing import Pool
from pprint import pprint
import gurobipy as grb
import multiprocessing
import torch.nn as nn
import numpy as np
import contextlib
import random
import torch
import time
import copy
import math
import re
import os
import pickle
from contextlib import contextmanager
import gc

from batch_processing import deeppoly, domain, gradient_abstractor
from dnn_solver.symbolic_network import SymbolicNetwork

from utils.timer import Timers
from abstract.crown import arguments
from utils.misc import MP
import settings

from utils.data_collector import collector

class DNNTheoremProverGurobi:

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec

        # self.hidden_nodes = sum([len(v) for k, v in self.layers_mapping.items()])

        self.decider = decider

        # with contextlib.redirect_stdout(open(os.devnull, 'w')):
        if 1:
            self.model = grb.Model()
            self.model.setParam('Threads', 1)
            self.model.setParam('OutputFlag', False)
            # self.model.setParam('FeasibilityTol', 1e-8)
            

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE, device=net.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE, device=net.device)

        # print(self.lbs_init.shape)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) for i in range(self.net.n_input)
        ]
        self.mvars = grb.MVar(self.gurobi_vars)

        self.count = 0 # debug

        self.solution = None

        self.transformer = SymbolicNetwork(net)

        self.deeppoly = deeppoly.BatchDeepPoly(net, back_sub_steps=0 if net.dataset == 'test' else 1e6)
        self.ga = gradient_abstractor.GradientAbstractor(net, spec, n_iters=10, lr=0.2)

        # self.concrete = self.net.get_concrete((self.lbs_init + self.ubs_init) / 2.0)
        # self.reversed_layers_mapping = {n: k for k, v in self.layers_mapping.items() for n in v}

        # self.last_assignment = {}        

        # # pgd attack 
        # self.backsub_cacher = BacksubCacher(self.layers_mapping, max_caches=10)

        # Timers.tic('Randomized attack')
        # self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)
        # stat, adv = self.rf.eval(timeout=settings.FALSIFICATION_TIMEOUT)
        # if settings.DEBUG:
        #     print('Randomized attack:', stat)
        # if stat == 'violated':
        #     self.solution = adv[0]
        # Timers.toc('Randomized attack')

        # self.crown = CrownWrapper(net)
        # self.deepzono = deepzono.DeepZono(net)

        # # self.glpk_solver = GLPKSolver(net.n_input)
        self.verified = False
        # self.optimized_layer_bounds = {}
        self.next_iter_implication = False

        self.batch = arguments.Config["general"]["batch"]
        self.domains = {}

        os.makedirs('gurobi', exist_ok=True)
        # self.cac_queue = multiprocessing.Queue()

        # self.grads = self.estimate_grads(self.lbs_init, self.ubs_init, steps=3)
        # print(self.grads)
        # exit()
        self.initialized = False
        self.initial_splits = 1 if net.dataset == 'test' else 4


    def _find_unassigned_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes

    def _get_equation(self, coeffs):
        expr = grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]
        return expr

    def add_domains(self, domains):
        for d in domains:
            if d.unsat:
                continue
                

            var = d.get_next_variable()
            if var in d.assignment:
                print('duplicated:', var)
                raise
            if var is None:
                continue
            # f_assignment = copy.deepcopy(d.assignment)
            # f_assignment[var] = False
            new_d1 = d.clone(var, False)
            new_d2 = d.clone(var, True)

            # print('Add next var:', var)
            # print(1, '===========>', d.bounds_mapping[151], d.bounds_mapping[119])
            # print(var, new_d1.assignment, new_d1.bounds_mapping[84])
            # print(var, new_d2.assignment, new_d2.bounds_mapping[84])
            assert hash(frozenset(new_d1.assignment.items())) not in self.domains
            assert hash(frozenset(new_d2.assignment.items())) not in self.domains

            self.domains[hash(frozenset(new_d1.assignment.items()))] = new_d1
            self.domains[hash(frozenset(new_d2.assignment.items()))] = new_d2


    def get_domains(self, cur_domain, batch=1):
        # print(f'get {batch} domains')
        ds = [cur_domain]
        if batch == 1:
            return ds
        idx = 1
        for k, v in self.domains.items():
            if v.valid:
                ds.append(v)
                idx += 1
            if idx == batch:
                break
        return ds

    
    # def build_temp_model(self, assignment):

    #     output_mat, backsub_dict = self.transformer(assignment)
    #     # print('full assignment')
    #     # raise

    #     lhs = np.zeros([len(assignment), len(self.gurobi_vars)])
    #     rhs = np.zeros(len(assignment))
    #     for i, (node, status) in enumerate(assignment.items()):
    #         if node not in backsub_dict:
    #             continue
    #         if status:
    #             lhs[i] = -1 * backsub_dict[node][:-1]
    #             rhs[i] = backsub_dict[node][-1] - 1e-6
    #         else:
    #             lhs[i] = backsub_dict[node][:-1]
    #             rhs[i] = -1 * backsub_dict[node][-1]

    #     self.model.remove(self.model.getConstrs())
    #     self.model.addConstr(lhs @ self.mvars <= rhs) 
    #     self.model.update()



    # @torch.no_grad()
    def __call__(self, assignment, info=None, full_assignment=None, use_implication=True):
        # debug
        self.count += 1
        cc = frozenset()
        implications = {}

        if self.solution is not None:
            return True, {}, None


        Timers.tic('Find node')
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False
        Timers.toc('Find node')

        # print('\t', self.count, 'full_assignment:', full_assignment, self.next_iter_implication)

        if not self.initialized:
            assert len(full_assignment) == 0
            # print('first time', assignment)
            self.initialized = True

            # (lbs, ubs), invalid_batch, hidden_bounds = self.deeppoly(self.lbs_init.unsqueeze(0), self.ubs_init.unsqueeze(0), return_hidden_bounds=True)
            (lbs, ubs), invalid_batch, hidden_bounds = self.compute_abstraction(self.lbs_init.unsqueeze(0), self.ubs_init.unsqueeze(0), [assignment], initial_splits=self.initial_splits)
            # print(lbs.shape)
            # print(lbs.shape)
            # print(len(hidden_bounds))
                
            if len(invalid_batch):
                # print('unsat')
                collector.add(unsat_by_abs=True)
                return False, cc, None

            stat, _ = self.spec.check_output_reachability(lbs[0], ubs[0])

            # print('reachable:', stat)
            if not stat: # conflict
                # print('unsat')
                collector.add(unsat_by_abs=True)
                return False, cc, None
            # for bound in hidden_bounds:
            #     print(bound.shape)

            bounds_mapping = {}
            for idx, (lb, ub) in enumerate([b[0] for b in hidden_bounds]):
                b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
                assert len(b) == len(self.layers_mapping[idx])
                bounds_mapping.update(dict(zip(self.layers_mapping[idx], b)))
                
            self.decider.update(bounds_mapping=bounds_mapping)

            d = domain.ReLUDomain(self.net, self.lbs_init, self.ubs_init, assignment, bounds_mapping, optimize_input_flag=True)
            d.init_optimizer()
            # d.valid = False

            # self.add_domains([d])
            # print('add_domain:', d.assignment)

            self.domains[hash(frozenset(d.assignment.items()))] = d


            # implication
            for node, (l, u) in bounds_mapping.items():
                if u <= 1e-6:
                    implications[node] = {'pos': False, 'neg': True}
                elif l >= -1e-6:
                    implications[node] = {'pos': True, 'neg': False}

            if len(implications):
                self.next_iter_implication = True

            return True, implications, is_full_assignment
        
        if is_full_assignment:


            output_mat, backsub_dict = self.transformer(assignment)
            print('full assignment')
            # raise

            lhs = np.zeros([len(assignment), len(self.gurobi_vars)])
            rhs = np.zeros(len(assignment))
            for i, (node, status) in enumerate(assignment.items()):
                if node not in backsub_dict:
                    continue
                if status:
                    lhs[i] = -1 * backsub_dict[node][:-1]
                    rhs[i] = backsub_dict[node][-1] - 1e-6
                else:
                    lhs[i] = backsub_dict[node][:-1]
                    rhs[i] = -1 * backsub_dict[node][-1]

            self.model.remove(self.model.getConstrs())
            self.model.addConstr(lhs @ self.mvars <= rhs) 
            self.model.update()

            
            flag_sat = False
            output_constraint = self.spec.get_output_property(
                [self._get_equation(output_mat[i]) for i in range(self.net.n_output)]
            )
            for cnf in output_constraint:
                ci = [self.model.addLConstr(_) for _ in cnf]
                self._optimize()
                self.model.remove(ci)
                if self.model.status == grb.GRB.OPTIMAL:
                    if self.check_solution(self.get_solution()):
                        flag_sat = True
                        break

            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            
            collector.add(unsat_by_lp=True)
            return False, cc, None


        # print(f'\t[{self.count}] assignment:', assignment)

        last_domain = self.domains.get(hash(frozenset(full_assignment.items())), None)

        if self.next_iter_implication:
            self.next_iter_implication = False
            # print('implication')
            # todo: haven't known yet
            if last_domain is not None:
                # print(f'\t[{self.count}] last literal:', info[0])
                return True, implications, is_full_assignment
            else:
                # conflict dectected by BCP
                pass

        if last_domain is not None: # BCP
            if last_domain.unsat:
                return False, cc, None
            return True, implications, is_full_assignment


        last_assignment = {}
        last_assignment.update(full_assignment)
        del last_assignment[info[0]]
        last_domain = self.domains.get(hash(frozenset(last_assignment.items())), None)

        if last_domain is None:
            print(f'\t[{self.count}] last_assignment:', last_assignment)
            print(f'\t[{self.count}] assignment:', assignment)
            print(f'\t[{self.count}] full_assignment:', full_assignment)
            print(f'\t[{self.count}] last literal:', info[0])
            print(f'\t[{self.count}] key:', hash(frozenset(last_assignment.items())) in self.domains)
            raise KeyError()
            
        last_domain.valid = False
        cur_domain = last_domain.clone(info[0], assignment[info[0]], assignment)
        cur_domain.optimize_input_bounds()
        # cur_domain.valid = False

        if cur_domain.unsat:
            # print('unsat')
            # del self.domains[hash(frozenset(cur_domain.assignment.items()))]
            return False, cc, None

        batch_bound = cur_domain.get_input_bounds().unsqueeze(0)
        batch_lower = batch_bound[:, 0]
        batch_upper = batch_bound[:, 1]
        batch_assignment = [assignment]
        assert (batch_lower <= batch_upper).all()
        
        Timers.tic('Abstraction')
        (lbs, ubs), invalid_batch, hidden_bounds = self.compute_abstraction(batch_lower.clone(), batch_upper.clone(), batch_assignment, initial_splits=self.initial_splits)
        Timers.toc('Abstraction')
        if len(invalid_batch) > 0:
            # del self.domains[hash(frozenset(cur_domain.assignment.items()))]
            collector.add(unsat_by_abs=True)
            return False, cc, None

        cur_domain.update_output_bounds(lbs[0], ubs[0])

        stat, _ = self.spec.check_output_reachability(cur_domain.output_lower, cur_domain.output_upper)
        if not stat: 
            # del self.domains[hash(frozenset(cur_domain.assignment.items()))]
            collector.add(unsat_by_abs=True)
            return False, cc, None

        # print(lbs)
        # print(ubs)
        # print(invalid_batch, len(invalid_batch))

        bounds_mapping = {}
        for idx, hb in enumerate(hidden_bounds):
            lb, ub = hb.squeeze(0)
            b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
            assert len(b) == len(self.layers_mapping[idx])
            assert (lb <= ub).all()
            bounds_mapping.update(dict(zip(self.layers_mapping[idx], b)))

        # print(bounds_mapping)
        for node, status in assignment.items():
            l, u = bounds_mapping[node]
            if l < 0 < u:
                if status:
                    bounds_mapping[node] = (max(0, l), u)
                else:
                    bounds_mapping[node] = (l, min(0, u))
            
        cur_domain.update_bounds_mapping(bounds_mapping)
        
        

        if settings.DEBUG:
            print(f'\n- Hidden bounds before: ', cur_domain.bounds_mapping)
        
        Timers.tic('Optimize hidden bounds')
        tic_ = time.time()
        cur_domain.optimize_bounds(assignment)
        collector.add(theory_implication_time=time.time()-tic_)
        Timers.toc('Optimize hidden bounds')


        if settings.DEBUG:
            print(f'\n- Hidden bounds after: ', cur_domain.bounds_mapping)
            
        if cur_domain.unsat:
            # del self.domains[hash(frozenset(cur_domain.assignment.items()))]
            return False, cc, None

        # print(f'\t[{self.count}] add:', cur_domain.assignment)
        self.domains[hash(frozenset(cur_domain.assignment.items()))] = cur_domain


        self.decider.update(bounds_mapping=cur_domain.bounds_mapping)

        # print(2, '===========>', cur_domain.bounds_mapping[151], cur_domain.bounds_mapping[119])
        # implication
        # print(self.count, unassigned_nodes)


        # nodes = []
        tic_ = time.time()
        for node in cur_domain.bounds_mapping:
            if node in assignment:
                continue
            l, u = cur_domain.bounds_mapping[node]
            if u <= 1e-6:
                # nodes.append(node)
                implications[node] = {'pos': False, 'neg': True}
            elif l >= -1e-6:
                # nodes.append(node)
                implications[node] = {'pos': True, 'neg': False}

        collector.add(theory_implication_time=time.time()-tic_)

        if len(implications):
            collector.add(implication=len(implications))
            # print(self.count, 'implication:', implications.keys())
            self.next_iter_implication = True

        return True, implications, is_full_assignment
        

    def _optimize(self):
        self.model.update()
        self.model.reset()
        self.model.optimize()


    def get_solution(self):
        if self.model.status == grb.GRB.LOADED:
            self._optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            return torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)
        return None


    def check_solution(self, solution):
        if torch.any(solution < self.lbs_init.view(self.net.input_shape)) or torch.any(solution > self.ubs_init.view(self.net.input_shape)):
            return False
        # if self.spec.check_solution(self.net(solution)):
        #     return True
        return True


    def estimate_grads(self, lower, upper, steps=3):
        # print(lower.device)
        inputs = [(((steps - i) * lower + i * upper) / steps) for i in range(steps + 1)]
        diffs = torch.zeros(len(lower), dtype=settings.DTYPE, device=lower.device)

        for sample in range(steps + 1):
            pred = self.net(inputs[sample].unsqueeze(0))
            for index in range(len(lower)):
                if sample < steps:
                    l_input = [m if i != index else u for i, m, u in zip(range(len(lower)), inputs[sample], inputs[sample+1])]
                    l_input = torch.tensor(l_input, dtype=settings.DTYPE, device=lower.device).unsqueeze(0)
                    l_i_pred = self.net(l_input)
                else:
                    l_i_pred = pred
                if sample > 0:
                    u_input = [m if i != index else l for i, m, l in zip(range(len(lower)), inputs[sample], inputs[sample-1])]
                    u_input = torch.tensor(u_input, dtype=settings.DTYPE, device=lower.device).unsqueeze(0)
                    u_i_pred = self.net(u_input)
                else:
                    u_i_pred = pred
                diff = [abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)][0]
                diffs[index] += diff.sum()
        return diffs / steps



    def split_multi_bound(self, multi_bound, dim=0, d=2):
        if isinstance(d, int):
            di = d
        else:
            di = d[dim]
        new_multi_bound = []
        for idx, (lower, upper) in enumerate(multi_bound):
            d_lb = lower[dim].clone()
            d_ub = upper[dim].clone()

            d_range = d_ub-d_lb
            d_step = d_range/di
            for i in range(di):
                # print(idx, dim, len(multi_bound), d_step, d_lb, d_ub)
                lower[dim] = d_lb + i*d_step
                upper[dim] = d_lb + (i+1)*d_step
                new_multi_bound.append((lower.clone(), upper.clone()))
                # print('new lower:', new_multi_bound[-1][0])
                # print('new upper:', new_multi_bound[-1][1])
            # print()
        # print('--')
        if dim + 1 < len(upper):
            return self.split_multi_bound(new_multi_bound, dim=dim+1, d=d)
        else:
            return new_multi_bound




    def split_multi_bounds(self, lower, upper, initial_splits=10):
        if initial_splits <= 1:
            return ([(lower, upper)])
        self.grads = self.estimate_grads(lower, upper, steps=3)
        smears = (self.grads.abs() + 1e-6) * (upper - lower + 1e-6)
        # print(smears)
        # print(smears.argmax())
        split_multiple = initial_splits / smears.sum()
        # print(split_multiple)
        num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]
        # num_splits = [5 if i >= 5 else i for i in num_splits]
        # print(f'\t[{self.count}] num_splits 1:', num_splits)
        # exit()
        num_splits = self.balancing_num_splits(num_splits)
        # num_splits = [i-1 if i > 2 else i-3 if i > 3 else i for i in num_splits ]
        # for i in num_splits:
        #     if i == 1: i = 2
        #     if i
        # num_splits[-2] = 2
        # num_splits = [5 if i >= 5 else i for i in num_splits]
        # num_splits = [i+1 if i < 2 else i for i in num_splits]
        # num_splits = [3] * 5
        # num_splits[-1] = 1
        # print(f'\t[{self.count}] num_splits 2:', num_splits)
        # num_splits = [i+1 if i==1 else i for i in num_splits ]
        # num_splits[-1] = 1
        # print('num_splits:', num_splits)
        # exit()
        assert all([x>0 for x in num_splits])
        return self.split_multi_bound([(lower, upper)], d=num_splits)


    def restore_input_bounds(self):
        pass


    def compute_abstraction(self, lowers, uppers, assignments, initial_splits=8):
        # pass
        # print(lowers.shape)
        # # print(multi_bounds)
        
        if settings.DEBUG:
            print(f'\n- Input bounds: ', [f'{b_:.02f}' for b_ in lowers.flatten()], [f'{b_:.02f}' for b_ in uppers.flatten()])

        batch_idx = []
        new_lowers = []
        new_uppers = []
        new_assignments = []
        for i in range(len(lowers)):
            multi_bounds = self.split_multi_bounds(lowers[i].clone(), uppers[i].clone(), initial_splits)
            new_lowers += [mb[0] for mb in multi_bounds]
            new_uppers += [mb[1] for mb in multi_bounds]
            batch_idx += [i] * len(multi_bounds)
            new_assignments += [assignments[i]] * len(multi_bounds)

            # print(len(multi_bounds))
        # print(len(new_lowers))
        new_lowers = torch.stack(new_lowers)
        new_uppers = torch.stack(new_uppers)
        batch_idx = torch.tensor(batch_idx, dtype=torch.int16)

        # print(new_lowers.amin(dim=0))
        # print(new_uppers.amax(dim=0))
        # print(batch_idx)
        assert (new_lowers <= new_uppers).all()


        # print(f'\t[{self.count}] ===================> batch', len(new_lowers))

        # new_batch_lower = torch.stack([mb[0] for mb in multi_bounds])
        # new_batch_upper = torch.stack([mb[1] for mb in multi_bounds])

        # new_batch_assignment = batch_assignment*len(new_batch_lower)
        # # print(new_batch_lower.shape)
        # # print((new_batch_lower <= new_batch_lower).all())
        # # print(batch_assignment * 28)
        # for i in range(len(new_lowers)):
        #     print('======== lower', i, new_lowers[i])
        if 1:
            with torch.no_grad():
                (lbs, ubs), invalid_batch, hidden_bounds = self.deeppoly(new_lowers, new_uppers, assignment=new_assignments, return_hidden_bounds=True, reset_param=True, adaptive=self.net.dataset!='test')
        else:    
            (lbs, ubs), invalid_batch, hidden_bounds = self.ga.get_optimized_bounds_from_input(new_lowers, new_uppers, assignment=new_assignments)

        if settings.DEBUG:
            print(f'\n- Abstraction:', lbs.detach(), ubs.detach())
            
        # for i in range(len(lbs)):
        #     print('======== lower', i, lbs[i])

            
        # print(invalid_batch, len(invalid_batch))
        # print(len(hidden_bounds), len(hidden_bounds[0]))
        valid_bidx = torch.ones(len(new_lowers)).to(torch.bool)
        valid_bidx[invalid_batch] = False
        # print(valid_bidx, len(valid_bidx))
        lbs = lbs[valid_bidx]
        ubs = ubs[valid_bidx]
        batch_idx = batch_idx[valid_bidx]
        # print(lbs.shape)

        new_hidden_bounds = []
        new_invalid_batch = []
        for hb in hidden_bounds:
            hb = hb[valid_bidx]
            new_hb_i = torch.empty((len(lowers), 2, hb.shape[-1]), dtype=lowers.dtype, device=lowers.device)
            # print(hb.shape, new_hb_i.shape)
            for idx in range(len(lowers)):
                bidx = torch.where(batch_idx == idx)[0]
                if len(bidx) == 0:
                    if idx not in new_invalid_batch:
                        new_invalid_batch.append(idx)
                    continue
                batch_hb = hb[bidx]
                # print(idx, bidx, len(bidx), batch_hb.shape, batch_hb[:, 0].shape)
                # exit()

                new_hb_i[idx, 0, :] = batch_hb[:, 0].amin(dim=0)
                new_hb_i[idx, 1, :] = batch_hb[:, 1].amax(dim=0)

                # print(hb_l.shape) 
            new_hidden_bounds.append(new_hb_i)

        # if len(invalid_batch2):
            # print(f'\t[{self.count}] ===================> invalid_batch', len(new_batch_upper), invalid_batch2)
        #     # exit()
        # for cac in new_hidden_bounds:
        #     print(cac.shape)
        #     break
        # print(f'\t[{self.count}] output lower 2:', len(lbs2), lbs2.amin(dim=0))
        # print(f'\t[{self.count}] output upper 2:', len(ubs2), ubs2.amax(dim=0))
        # print(f'\t[{self.count}] stat 2:', self.spec.check_output_reachability(lbs2.amin(dim=0), ubs2.amax(dim=0))[0])

        # print(invalid_batch, new_invalid_batch)
        new_lbs = torch.empty((len(lowers), lbs.shape[-1]), dtype=lowers.dtype, device=lowers.device)
        new_ubs = torch.empty((len(lowers), ubs.shape[-1]), dtype=lowers.dtype, device=lowers.device)
        # print(lbs.shape, )
        for idx in range(len(lowers)):
            if idx in new_invalid_batch:
                continue
            bidx = torch.where(batch_idx == idx)
            new_lbs[idx] = lbs[bidx].amin(dim=0)
            new_ubs[idx] = ubs[bidx].amax(dim=0)
        
        # print(new_lbs)
        # print(new_ubs)

        return (new_lbs, new_ubs), new_invalid_batch, new_hidden_bounds

    def balancing_num_splits(self, num_splits, max_batch=8):
        # num_add = math.floor(max_batch / math.prod(num_splits))
        # # print(num_add, num_splits)
        # count = 0
        # if 1 < num_add <= 3:
        #     for i in range(len(num_splits)):
        #         if num_splits[i] == 1:
        #             num_splits[i] += 1
        #             count += 1
        #         if count == 1:
        #             break
        # elif 3 < num_add <= 5:
        #     for i in range(len(num_splits)):
        #         if num_splits[i] == 1:
        #             num_splits[i] += 1
        #             count += 1
        #         if count == 2:
        #             break
        # print(num_add, num_splits)
        # exit()
        # return [3] * len(num_splits)
        num_splits = np.array(num_splits)
        while True:
            # print(num_splits)
            idx = np.argmin(num_splits)
            num_splits[idx] += 1
            # print(num_splits)
            # exit()
            if math.prod(num_splits) > max_batch:
                num_splits[idx] -= 1
                break

        return num_splits.tolist()