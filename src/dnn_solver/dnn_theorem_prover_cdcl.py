from pprint import pprint
import gurobipy as grb
import torch.nn as nn
import numpy as np
import random
import torch
import math
import time
import copy

from batch_processing import deeppoly, domain, gradient_abstractor
from dnn_solver.symbolic_network import SymbolicNetwork

from utils.timer import Timers
import settings

MAX_BATCH_ABSTRACTION = 4000


class DNNTheoremProverCDCL:

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec

        self.decider = decider

        self.model = grb.Model()
        # self.model.setParam('Threads', 1)
        self.model.setParam('OutputFlag', False)
        # self.model.setParam('FeasibilityTol', 1e-8)


        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE, device=net.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE, device=net.device)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) for i in range(self.net.n_input)
        ]
        self.mvars = grb.MVar(self.gurobi_vars)

        self.count = 0 # debug

        self.solution = None

        self.transformer = SymbolicNetwork(net)

        self.deeppoly = deeppoly.BatchDeepPoly(net, back_sub_steps=100)
        # self.ga = gradient_abstractor.GradientAbstractor(net, spec, n_iters=10, lr=0.2)

        self.verified = False

        # os.makedirs('gurobi', exist_ok=True)

        self.initialized = False
        self.initial_splits = 60
        self.last_assignment = {}


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


    def tighten_input_bounds(self, assignment):
        _, backsub_dict = self.transformer(assignment)
        self.model.remove(self.model.getConstrs())
        
        for i, v in enumerate(self.model.getVars()):
            v.lb = self.lbs_init[i]
            v.ub = self.ubs_init[i]
        self.model.update()
        
        for node, status in assignment.items():
            if node not in backsub_dict:
                continue
            cstr = self._get_equation(backsub_dict[node])
            if status:
                self.model.addLConstr(cstr >= 1e-6)
            else:
                self.model.addLConstr(cstr <= 0)
                
        self.model.update()
        
        new_input_lowers = self.lbs_init.clone().unsqueeze(0)
        new_input_uppers = self.ubs_init.clone().unsqueeze(0)
        
        for i, v in enumerate(self.model.getVars()):
            self.model.setObjective(v, grb.GRB.MINIMIZE)
            self.model.optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                return None, None
            
            if self.model.status != grb.GRB.OPTIMAL:
                continue
                
            v.lb = self.model.objval
            # upper bound
            self.model.setObjective(v, grb.GRB.MAXIMIZE)
            self.model.optimize()
            v.ub = self.model.objval
            
            if new_input_lowers[0][i] < v.lb:
                new_input_lowers[0][i] = v.lb
            if new_input_uppers[0][i] > v.ub:
                new_input_uppers[0][i] = v.ub
        self.model.update()
                
        return new_input_lowers, new_input_uppers
        

    def tighten_hidden_bounds(self, assignment, bounds_mapping):
        print('tighten_hidden_bounds')
        _, backsub_dict = self.transformer(assignment)
        for node in backsub_dict:
            lb, ub = bounds_mapping[node]
            if lb < 0 < ub:
                # print(node, bounds_mapping[node])
                obj = self._get_equation(backsub_dict[node])
                # print(obj)
                self.model.setObjective(obj, grb.GRB.MINIMIZE)
                self.model.optimize()
                new_lb = self.model.objval
                
                self.model.setObjective(obj, grb.GRB.MAXIMIZE)
                self.model.optimize()
                new_ub = self.model.objval
                
                # print(new_lb, new_ub)
                bounds_mapping[node] = (max(lb, new_lb), min(ub, new_ub))
                
    
    def __call__(self, assignment, info=None, full_assignment=None):

        # debug
        self.count += 1
        print('assignment:', full_assignment)
        
            
        cc = frozenset()
        implications = {}

        if self.solution is not None:
            return True, {}, None

        #
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False

        # full 
        if is_full_assignment:

            output_mat, backsub_dict = self.transformer(assignment)
            print('full assignment')
            # raise

            lhs = np.zeros([len(assignment), len(self.gurobi_vars)])
            rhs = np.zeros(len(assignment))
            for i, (node, status) in enumerate(assignment.items()):
                if node not in backsub_dict:
                    continue
                coeffs = backsub_dict[node].detach().cpu().numpy()
                if status:
                    lhs[i] = -1 * coeffs[:-1]
                    rhs[i] = coeffs[-1] - 1e-6
                else:
                    lhs[i] = coeffs[:-1]
                    rhs[i] = -1 * coeffs[-1]

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
            return False, cc, None


        if len(full_assignment) and (self.last_assignment == full_assignment):
            return True, {}, None

        # partial assignment
        # if len(full_assignment):
        if len(assignment):
            # tighten input bounds
            # print('assignment:', assignment)
            input_lowers, input_uppers = self.tighten_input_bounds(assignment)
            # print('\tinit:', self.lbs_init)
            # print('\tinit:', self.ubs_init)
            if input_lowers is None:
                return False, cc, None
                
            # raise
        else:
            input_lowers = self.lbs_init.clone().unsqueeze(0)
            input_uppers = self.ubs_init.clone().unsqueeze(0)
        
        print('\tinit     :', self.lbs_init)
        print('\tinit     :', self.ubs_init)
        print('\ttightened:', input_lowers)
        print('\ttightened:', input_uppers)
        
        # approximation
        # print('\t', self.count, 'full_assignment:', full_assignment)
        tic = time.time()
        (lbs, ubs), invalid_batch, hidden_bounds = self.compute_abstraction(
            lowers=input_lowers, 
            uppers=input_uppers, 
            assignments=[assignment], 
            initial_splits=self.initial_splits,
        )
        print('approximation time:', time.time() - tic)
        
        print(lbs)
        print(ubs)
        
        if len(invalid_batch):
            print('invalid bounds')
            return False, cc, None
            
        stat, _ = self.spec.check_output_reachability(lbs[0], ubs[0])
        print('stat:', stat)
        if not stat: # conflict
            return False, cc, None
        
        # approx hidden bounds
        bounds_mapping = {}
        for idx, (lb, ub) in enumerate([b[0] for b in hidden_bounds]):
            b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
            assert len(b) == len(self.layers_mapping[idx])
            bounds_mapping.update(dict(zip(self.layers_mapping[idx], b)))
            
            
        # tighten hidden bounds
        if len(assignment):
        # if len(full_assignment):
            self.tighten_hidden_bounds(assignment, bounds_mapping)
            
        # print('bounds_mapping')
        # print(bounds_mapping)
        self.decider.update(bounds_mapping=bounds_mapping)
        
        
        # implication
        for node, (l, u) in bounds_mapping.items():
            if node in assignment:
                continue
            if u <= 1e-6:
                implications[node] = {'pos': False, 'neg': True}
            elif l >= -1e-6:
                implications[node] = {'pos': True, 'neg': False}

        print('implications:', len(implications))
        print()
        # if len(full_assignment):
        #     exit()
        self.last_assignment = copy.deepcopy(full_assignment)

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
        
        if not hasattr(self, 'num_splits') or 1:
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
            self.num_splits = self.balancing_num_splits(num_splits, max_batch=MAX_BATCH_ABSTRACTION)
        # print(num_splits)
        
        # exit()
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
        assert all([x>0 for x in self.num_splits])
        return self.split_multi_bound([(lower, upper)], d=self.num_splits)


    def restore_input_bounds(self):
        pass


    def compute_abstraction(self, lowers, uppers, assignments, initial_splits=8):
        # pass
        print(lowers.shape, lowers.device)
        # # print(multi_bounds)
        batch_idx = []
        new_lowers = []
        new_uppers = []
        new_assignments = []
        tic = time.time()
        for i in range(len(lowers)):
            multi_bounds = self.split_multi_bounds(lowers[i].clone(), uppers[i].clone(), initial_splits)
            new_lowers += [mb[0] for mb in multi_bounds]
            new_uppers += [mb[1] for mb in multi_bounds]
            batch_idx += [i] * len(multi_bounds)
            new_assignments += [assignments[i]] * len(multi_bounds)

        print('split input time:', time.time() - tic)
            # print(len(multi_bounds))
        print(len(new_lowers))
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
        tic = time.time()
        if 1:
            with torch.no_grad():
                (lbs, ubs), invalid_batch, hidden_bounds = self.deeppoly(new_lowers, new_uppers, assignment=new_assignments, return_hidden_bounds=True, reset_param=True)
        else:    
            (lbs, ubs), invalid_batch, hidden_bounds = self.ga.get_optimized_bounds_from_input(new_lowers, new_uppers, assignment=new_assignments)
        print('bounding time:', time.time() - tic)

        
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
        num_splits = np.array(num_splits)
        while math.prod(num_splits) >= max_batch:
            idx = np.argsort(num_splits)[-1 * random.choice([1, 2])]
            num_splits[idx] -= 1
            
        while math.prod(num_splits) <= max_batch:
            idx = np.argsort(num_splits)[-1 * random.choice([1, 2])]
            num_splits[idx] += 1
            
        return num_splits.tolist()