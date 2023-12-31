import gurobipy as grb
import torch.nn as nn
import numpy as np
import contextlib
import random
import torch
import time
import copy
import re
import os

from dnn_solver.symbolic_network import SymbolicNetwork
from utils.timer import Timers
import settings

from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from auto_LiRPA.utils import *
from abstract.crown import *
from abstract.crown import arguments

from utils.data_collector import collector


class DNNTheoremProverCrown:

    def __init__(self, net, spec, decider=None):

        torch.manual_seed(arguments.Config["general"]["seed"])
        random.seed(arguments.Config["general"]["seed"])
        np.random.seed(arguments.Config["general"]["seed"])
        self.batch = arguments.Config["general"]["batch"]

        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec
        self.verified = False
        self.decider = decider


        self.crown_decision_mapping = {}
        for lid, lnodes in self.layers_mapping.items():
            for jj, node in enumerate(lnodes):
                self.crown_decision_mapping[(lid, jj)] = node

        ##########################################################################################

        # with contextlib.redirect_stdout(open(os.devnull, 'w')):
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE, device=net.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE, device=net.device)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
            for i in range(self.net.n_input)
        ]
        self.mvars = grb.MVar(self.gurobi_vars)
        self.transformer = SymbolicNetwork(net)
        self.last_assignment = {}        
        
        ##########################################################################################
        prop_mat, prop_rhs = spec.mat[0]
        if len(prop_rhs) > 1:
            raise NotImplementedError
            output = self.net(((self.lbs_init + self.ubs_init)/2).view(-1, *self.net.input_shape[1:])).detach().cpu().numpy().flatten()
            vec = prop_mat.dot(output)
            selected_prop = prop_mat[vec.argmax()]
            y = int(np.where(selected_prop == 1)[0])  # true label
            target = int(np.where(selected_prop == -1)[0])  # target label
            decision_thresh = prop_rhs[vec.argmax()]
        else:
            assert len(prop_mat) == 1
            y = np.where(prop_mat[0] == 1)[0]
            if len(y) != 0:
                y = int(y)
            else:
                y = None
            target = np.where(prop_mat[0] == -1)[0]  # target label
            target = int(target) if len(target) != 0 else None  # Fix constant specification with no target label.
            if y is not None and target is None:
                y, target = target, y  # Fix vnnlib with >= const property.
            decision_thresh = prop_rhs[0]
        arguments.Config["bab"]["decision_thresh"] = decision_thresh

        if y is not None:
            if net.n_output > 1:
                c = torch.zeros((1, 1, net.n_output), dtype=settings.DTYPE, device=net.device)  # we only support c with shape of (1, 1, n)
                c[0, 0, y] = 1
                c[0, 0, target] = -1
            else:
                # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
                c = (float(y) - 0.5) * 2 * torch.ones(size=(1, 1, 1), dtype=settings.DTYPE, device=net.device)
        else:
            # if there is no ture label, we only verify the target output
            c = torch.zeros((1, 1, net.n_output), dtype=settings.DTYPE, device=net.device)  # we only support c with shape of (1, 1, n)
            c[0, 0, target] = -1

        if arguments.Config["general"]["verbose"]:
            print(f'[+] Label: {y}, Test: {target}')
        # if target != 9:
        #     self.verified = True

        input_shape = net.input_shape
        x_range = torch.tensor(spec.bounds, dtype=settings.DTYPE, device=net.device)
        data_min = x_range[:, 0].reshape(input_shape)
        data_max = x_range[:, 1].reshape(input_shape)
        data = x_range.mean(1).reshape(input_shape)
        # print(x_range.shape, data_max.shape)

        # print((data_max - data_min).sum(), net.n_output)
        # print(c)
        # print(net.layers)

        self.lirpa = LiRPAConvNet(net.layers, y, target, device=net.device, in_size=input_shape, deterministic=False, conv_mode=arguments.Config['general']['conv_mode'], c=c)

        # print('Model prediction is:', self.lirpa.net(data))

        self.decision_thresh = decision_thresh

        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=data_min, x_U=data_max)
        self.x = BoundedTensor(data, ptb).to(net.device)



        self.branching_reduceop = arguments.Config['bab']['branching']['reduceop']
        # exit()
        self.count = 0

        self.domains = {}

        self.solution = None

        # self.last_dl = None
        # self.last_domain = None

        self.next_iter_implication = False


    def _get_equation(self, coeffs):
        expr = grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]
        return expr

    def _find_unassigned_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes


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
        if self.spec.check_solution(self.net(solution)):
            return True
        return False


    def __call__(self, assignment, info=None, full_assignment=None, use_implication=True):
        # print('\n\n-------------------------------------------------\n')
        
        self.count += 1
        cc = frozenset()
        implications = {}
        # cur_var, dl, branching_decision = info

        # print('\n\nstart loop', self.count, len(assignment), full_assignment, len([d for _, d in self.domains.items() if d.valid]))

        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False

        # initialize
        if len(assignment) == 0:
            Timers.tic('build_the_model')
            global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = self.lirpa.build_the_model(
                None, self.x, stop_criterion_func=stop_criterion_sum(self.decision_thresh))
            Timers.toc('build_the_model')

            self.pre_relu_indices = pre_relu_indices

            if global_lb >= self.decision_thresh:
                collector.add(unsat_by_abs=True)
                return False, cc, None

            # We only keep the alpha for the last layer.
            new_slope = defaultdict(dict)
            output_layer_name = self.lirpa.net.final_name
            for relu_layer, alphas in slope.items():
                new_slope[relu_layer][output_layer_name] = alphas[output_layer_name]
            slope = new_slope

            candidate_domain = ReLUDomain(lA, global_lb, global_ub, lower_bounds, upper_bounds, slope, history=history, depth=0, primals=primals, assignment_mapping=self.crown_decision_mapping).to_device(self.net.device, partial=True)

            mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(candidate_domain)
            history = [sd.history for sd in selected_domains]

            # print('=========================>', self.count, orig_lbs[0].flatten()[92], orig_ubs[0].flatten()[92])
            # print(unassigned_nodes)
            # print(orig_lbs)
            # print(orig_lbs[0].shape)
            # print(orig_lbs[1].shape)
            # print(lower_bounds)

            if use_implication:
                tic_ = time.time()
                count = 1
                for lbs, ubs in zip(orig_lbs[:-1], orig_ubs[:-1]):
                    # print(lbs)
                    if (lbs - ubs).max() > 1e-6:
                        collector.add(unsat_by_abs=True)
                        return False, cc, None

                    for jj, (l, u) in enumerate(zip(lbs.flatten(), ubs.flatten())):
                        if (count + jj) in assignment:
                            continue
                        if u <= 0:
                            implications[count + jj] = {'pos': False, 'neg': True}
                            # print('u', count+jj, u)
                        elif l > 0:
                            implications[count + jj] = {'pos': True, 'neg': False}
                            # print('l', count+jj, l)
                    count += lbs.numel()
                collector.add(theory_implication_time=time.time()-tic_)

            # print(orig_lbs[1].flatten()[161], orig_ubs[1].flatten()[161])
            # exit()
            # print(len(implications))

            if self.decider is not None:
                crown_params = orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, slopes, betas, history
                self.decider.update(crown_params=crown_params)

            # candidate_domain.valid = True
            self.domains[hash(frozenset(assignment.items()))] = candidate_domain

            # self.domains.append(candidate_domain)
            # self.last_domain.valid = False
            if len(implications):
                collector.add(implication=len(implications))
                self.next_iter_implication = True

            return True, implications, is_full_assignment

        if is_full_assignment:
            print('\t\t\tfull assignment')
            Timers.tic('check full assignment')
            # output_mat, backsub_dict = self.transformer(assignment)
            # backsub_dict_expr = self.backsub_cacher.get_cache(assignment)

            # if backsub_dict_expr is not None:
            #     backsub_dict_expr.update({k: self._get_equation(v) for k, v in backsub_dict.items() if k not in backsub_dict_expr})
            # else:
            #     backsub_dict_expr = {k: self._get_equation(v) for k, v in backsub_dict.items()}

            # self.backsub_cacher.put(assignment, backsub_dict_expr)
            # constrs = []
            # for node, status in assignment.items():
            #     if status:
            #         constrs.append(self.model.addLConstr(backsub_dict_expr[node] >= 1e-6))
            #     else:
            #         constrs.append(self.model.addLConstr(backsub_dict_expr[node] <= 0))
            
            # flag_sat = False
            # output_constraint = self.spec.get_output_property(
            #     [self._get_equation(output_mat[i]) for i in range(self.net.n_output)]
            # )
            # for cnf in output_constraint:
            #     ci = [self.model.addLConstr(_) for _ in cnf]
            #     self._optimize()
            #     self.model.remove(ci)
            #     if self.model.status == grb.GRB.OPTIMAL:
            #         if self.check_solution(self.get_solution()):
            #             flag_sat = True
            #             break

            # self.model.remove(constrs)

            output_mat, backsub_dict = self.transformer(assignment)
            # print('full assignment')
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

            Timers.toc('check full assignment')

            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            collector.add(unsat_by_lp=True)
            return False, cc, None


        if self.next_iter_implication:

            self.next_iter_implication = False
            # print('\t', self.count, 'implication')
            # implication iteration
            # TODO: haven't known yet
            return True, implications, is_full_assignment


        domain_key = hash(frozenset(full_assignment.items()))
        cur_domain = self.domains.get(domain_key, None)
        # print('\tfull_assignment:', full_assignment)
        # print('\t', 'full_assignment:', full_assignment)
        # print('\t', 'cur_domain:', cur_domain.get_assignment() if cur_domain is not None else None)
        # print('\t', 'dl:', info[1])

        if cur_domain is None:
            cur_var, _, crown_decision = info
            # print('\tcac:', cur_var, crown_decision)
            last_assignment = copy.deepcopy(full_assignment)
            del last_assignment[cur_var]
            # print('\tlast_assignment:', last_assignment)
            cur_domain = self.domains[hash(frozenset(last_assignment.items()))]
            # del self.domains[hash(frozenset(last_assignment.items()))]
            mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(cur_domain, batch=self.batch)
            # cur_domain.valid = False


            for d in selected_domains:
                d.valid = False # mark as processed
                # print('cac', d.get_assignment())
                # del self.domains[hash(frozenset(d.get_assignment().items()))]

            batch = len(selected_domains)
            history = [sd.history for sd in selected_domains]

            if len(selected_domains) > 1:
                # assert cur_domain == selected_domains[0]

                branching_decision = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, branching_reduceop=self.branching_reduceop, slopes=slopes, betas=betas, history=history)
                # branching_decision = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, batch=batch, branching_reduceop=self.branching_reduceop)
                # print(crown_decision)
                # print(branching_decision)
                # print(len(selected_domains))
                # print(len(branching_decision))
                # assert crown_decision[0] in branching_decision
                # if branching_decision[0] != crown_decision[0]:
                #     raise
                branching_decision[0] = crown_decision[0]
                for idx, d in enumerate(selected_domains):
                    d.next_var = self.crown_decision_mapping[tuple(branching_decision[idx])]

            else:
                branching_decision = crown_decision
                selected_domains[0].next_var = branching_decision[0]

            # print('\t --->', self.count, batch, branching_decision)

            split_history = [sd.split_history for sd in selected_domains]
            split = {}
            split["decision"] = [[bd] for bd in branching_decision]
            split["coeffs"] = [[1.] for i in range(len(branching_decision))]
            split["diving"] = 0

            Timers.tic('get_lower_bound')
            ret = self.lirpa.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history, split_history=split_history, layer_set_bound=True, betas=betas, single_node_split=True, intermediate_betas=intermediate_betas)
            Timers.toc('get_lower_bound')


            dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals = ret

            Timers.tic('add_domain')
            domain_list = add_domain_parallel(lA=lAs[:2*batch], lb=dom_lb[:2*batch], ub=dom_ub[:2*batch], lb_all=dom_lb_all[:2*batch], up_all=dom_ub_all[:2*batch],
                                             domains=None, selected_domains=selected_domains[:batch], slope=slopes[:2*batch], beta=betas[:2*batch],
                                             growth_rate=0, branching_decision=branching_decision, decision_thresh=self.decision_thresh,
                                             split_history=split_history[:2*batch], intermediate_betas=intermediate_betas[:2*batch],
                                             check_infeasibility=False, primals=primals[:2*batch] if primals is not None else None)
            Timers.toc('add_domain')

            Timers.tic('save_domain')
            for d in domain_list:
                # print('\t---> add:', 'assignment', d.get_assignment(), d.valid)
                key = hash(frozenset(d.get_assignment().items()))
                if key not in self.domains:
                    self.domains[key] = d
                else:
                    # print(d.get_assignment)
                    # print(d.get_assignment() == self.domains[key].get_assignment())
                    # print(d.history)
                    # print(self.domains[key].history)
                    if d.history != self.domains[key].history:
                        # FIXME: 03/10/22: same key, different order of history
                        pass
                        
            Timers.toc('save_domain')

            # print(len(domain_list), full_assignment == domain_list[0].get_assignment())
            # print(len(domain_list), full_assignment == domain_list[1].get_assignment())

            # del self.domains[hash(frozenset(cur_domain.get_assignment().items()))]
            # print('\tfull_assignment:', full_assignment)
            cur_domain = self.domains[hash(frozenset(full_assignment.items()))]

        else:
            mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(None, batch=self.batch)

            if len(selected_domains) > 0:
                for d in selected_domains:
                    d.valid = False # mark as processed

                batch = len(selected_domains)
                history = [sd.history for sd in selected_domains]

                branching_decision = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, branching_reduceop=self.branching_reduceop, slopes=slopes, betas=betas, history=history)

                split_history = [sd.split_history for sd in selected_domains]
                split = {}
                split["decision"] = [[bd] for bd in branching_decision]
                split["coeffs"] = [[1.] for i in range(len(branching_decision))]
                split["diving"] = 0

                Timers.tic('get_lower_bound')
                ret = self.lirpa.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history, split_history=split_history, layer_set_bound=True, betas=betas, single_node_split=True, intermediate_betas=intermediate_betas)
                Timers.toc('get_lower_bound')


                dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals = ret

                Timers.tic('add_domain')
                domain_list = add_domain_parallel(lA=lAs[:2*batch], lb=dom_lb[:2*batch], ub=dom_ub[:2*batch], lb_all=dom_lb_all[:2*batch], up_all=dom_ub_all[:2*batch],
                                                domains=None, selected_domains=selected_domains[:batch], slope=slopes[:2*batch], beta=betas[:2*batch],
                                                growth_rate=0, branching_decision=branching_decision, decision_thresh=self.decision_thresh,
                                                split_history=split_history[:2*batch], intermediate_betas=intermediate_betas[:2*batch],
                                                check_infeasibility=False, primals=primals[:2*batch] if primals is not None else None)
                Timers.toc('add_domain')

                Timers.tic('save_domain')
                for d in domain_list:
                    key = hash(frozenset(d.get_assignment().items()))
                    if key not in self.domains:
                        self.domains[key] = d
                    else:
                        if d.history != self.domains[key].history:
                            # FIXME: 03/10/22: same key, different order of history
                            pass
                Timers.toc('save_domain')

        # if len(selected_domains) >= 10:
        #     self.generate_samples(selected_domains)

        mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(cur_domain)

        history = [sd.history for sd in selected_domains]
        if self.decider:
            crown_params = orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, slopes, betas, history
            # print('\t', self.count, 'update crown_params')
            self.decider.update(crown_params=crown_params)
            self.decider.next_var = cur_domain.next_var

        if cur_domain.lower_bound >= self.decision_thresh:
            # del self.domains[hash(frozenset(cur_domain.get_assignment().items()))]
            # print('\t===============>', self.count, 'unsat')
            collector.add(unsat_by_abs=True)
            return False, cc, None
        
        if use_implication:
            tic_ = time.time()
            count = 1
            for lbs, ubs in zip(orig_lbs[:-1], orig_ubs[:-1]):
                if (lbs - ubs).max() > 1e-6:
                    collector.add(unsat_by_abs=True)
                    return False, cc, None

                for jj, (l, u) in enumerate(zip(lbs.flatten(), ubs.flatten())):
                    if (count + jj) in assignment:
                        continue
                    if u <= 0:
                        implications[count + jj] = {'pos': False, 'neg': True}
                    elif l > 0:
                        implications[count + jj] = {'pos': True, 'neg': False}
                count += lbs.numel()

            collector.add(theory_implication_time=time.time()-tic_)

        if len(implications):
            collector.add(implication=len(implications))
            self.next_iter_implication = True


        return True, implications, is_full_assignment


    def restore_input_bounds(self):
        pass


    def get_domain_params(self, current_domain, batch=1):

        lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []
        device = self.net.device

        selected_domain_hashes = []

        idx = 0
        # if current_domain.valid is True and current_domain.lower_bound < self.decision_thresh:
        if current_domain is not None:
            current_domain.to_device(device, partial=True)
            # current_domain.valid = False  # set False to avoid another pop
            lAs.append(current_domain.lA)
            lower_all.append(current_domain.lower_all)
            upper_all.append(current_domain.upper_all)
            slopes_all.append(current_domain.slope)
            betas_all.append(current_domain.beta)
            intermediate_betas_all.append(current_domain.intermediate_betas)
            selected_candidate_domains.append(current_domain)
            selected_domain_hashes.append(current_domain.get_assignment())
            idx += 1

        if batch > idx:
            for k, selected_candidate_domain in self.domains.items():
                if selected_candidate_domain.lower_bound < self.decision_thresh and selected_candidate_domain.valid is True and selected_candidate_domain.get_assignment() not in selected_domain_hashes:
                    # print('--------> select:', selected_candidate_domain.get_assignment())
                    selected_candidate_domain.to_device(device, partial=True)
                    # selected_candidate_domain.valid = False  # set False to avoid another pop
                    lAs.append(selected_candidate_domain.lA)
                    lower_all.append(selected_candidate_domain.lower_all)
                    upper_all.append(selected_candidate_domain.upper_all)
                    slopes_all.append(selected_candidate_domain.slope)
                    betas_all.append(selected_candidate_domain.beta)
                    intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
                    selected_candidate_domains.append(selected_candidate_domain)
                    selected_domain_hashes.append(selected_candidate_domain.get_assignment())
                    idx += 1
                    if idx == batch:
                        break
                # selected_candidate_domain.valid = False   # set False to avoid another pop

        batch = len(selected_candidate_domains)
        if batch == 0:
            return None, None, None, None, None, None, None, []


        lower_bounds = []
        for j in range(len(lower_all[0])):
            lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
        lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

        upper_bounds = []
        for j in range(len(upper_all[0])):
            upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
        upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

        # Reshape to batch first in each list.
        new_lAs = []
        for j in range(len(lAs[0])):
            new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
        # Transfer to GPU.
        new_lAs = [t.to(device=device, non_blocking=True) for t in new_lAs]

        slopes = []
        if slopes_all[0] is not None:
            if isinstance(slopes_all[0], dict):
                # Per-neuron slope, each slope is a dictionary.
                slopes = slopes_all
            else:
                for j in range(len(slopes_all[0])):
                    slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

        # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
        lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
        upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]
        
        # if batch >= 50:
        #     self.generate_samples(lower_bounds[0], upper_bounds[0])
        
        # Recompute the mask on GPU.
        new_masks = []
        for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
            new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())
        return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains


    @torch.no_grad()
    def forward_from_1st_hidden_layer(self, x):
        idx = 0
        for layer in self.net.layers:
            if idx > 0:
                x = layer(x)
            if isinstance(layer, nn.Linear): # skip input layer, forward from 1st hidden layer
                idx += 1
        return x

    def generate_samples(self, lowers, uppers):
        # return
        # print(lowers.shape)
        lbs = self.lbs_init.repeat(500, 1)
        ubs = self.ubs_init.repeat(500, 1)
        # print(lbs.shape)
        # exit()
        # assert torch.all(lowers <= uppers)
        # s = (uppers - lowers) * torch.rand(*lowers.shape, dtype=lowers.dtype, device=lowers.device) + lowers
        s = (ubs - lbs) * torch.rand(*lbs.shape, dtype=lbs.dtype, device=lbs.device) + lbs
        # assert torch.all(s[0] == s[1])
        assert torch.all(s >= lbs) and torch.all(s <= ubs)
        # exit()
        # x = 
        # assert torch.all(s >= lowers) and torch.all(s <= uppers)
        # x = self.forward_from_1st_hidden_layer(s)
        x = self.net(s)
        # print(s.shape, x.shape)
        for prop_mat, prop_rhs in self.spec.mat:
            prop_mat = torch.tensor(prop_mat, dtype=lowers.dtype, device=lowers.device)
            prop_rhs = torch.tensor(prop_rhs, dtype=lowers.dtype, device=lowers.device)
            # print(prop_mat.shape)
            vec = (prop_mat @ x.transpose(0, 1)).squeeze()
            # print(vec.data, prop_rhs)
            if torch.any(vec <= prop_rhs):
                cex_indices = torch.where(vec <= prop_rhs)
                # print('cex', cex_indices, vec.data)
                self.solution = s[0].clone()
                # for cex_idx in cex_indices[0]:
                #     self.model.setParam('PoolSolutions', 100)
                #     self.model.setParam('PoolSearchMode', 2)
                #     self.model.optimize()

                #     self.model.remove(self.model.getConstrs())
                #     _, backsub_dict = self.transformer({})
                #     sample = s[cex_idx].flatten()
                #     # print(cex_idx, sample.shape)
                #     for i in range(len(sample)):
                #         eq = self._get_equation(backsub_dict[i+1])
                #         self.model.addLConstr( eq >= 1e-6 if sample[i] > 0 else eq <= 0)
                #         # self.model.addLConstr( eq == sample[i] if sample[i] > 0 else eq <= 0)
                #     self.model.optimize()
                #     # self.model.write('gurobi/test.lp')

                #     if self.model.status == grb.GRB.OPTIMAL:
                #         print('Number of solutions found: ' + str(self.model.SolCount))
                #         cex = torch.as_tensor([[v.X for v in self.gurobi_vars]], dtype=lowers.dtype, device=lowers.device)
                #         if self.spec.check_solution(self.net(cex)):
                #             self.solution = cex.clone()
                #             return
                #     #         print('cex')
                #     #         exit()
                    # print('spurious cex')


                    # print(cex_idx, s[cex_idx], s[cex_idx].shape)
                    # assignment = {(i+1): True if s[cex_idx][0][i] > 0 else False for i in range(len(s[cex_idx][0]))}
                    # print(assignment)
                #     print(self.forward_from_1st_hidden_layer(s[cex_idx]))
                # for i in range(len(s)):
                #     print(self.forward_from_1st_hidden_layer(s[i]).detach().cpu().numpy())
                
        # self.model.setParam('PoolSolutions', 1)
                
                # exit()
        # tic = time.time()
        # for d in selected_domains:
        #     assignment = d.get_assignment()
        #     # print(assignment)
        #     full_assignment = {}
        #     full_assignment.update(assignment)
        #     for lid, (layer_l, layer_u) in enumerate(zip(d.lower_all[:-1], d.upper_all[:-1])):
        #         for nid in range(len(layer_l[0])):
        #             status = None
        #             if layer_l[0][nid] > -1e-6:
        #                 status = True
        #                 # print(lid, nid, self.crown_decision_mapping[(lid, nid)], True)
        #             elif layer_u[0][nid] < 1e-6:
        #                 status = False
        #                 # print(lid, nid, self.crown_decision_mapping[(lid, nid)], False)

        #             if status is not None:
        #                 node = self.crown_decision_mapping[(lid, nid)]
        #                 if node in full_assignment:
        #                     if status != full_assignment[node]:
        #                         raise
        #                 full_assignment[node] = status
        #         # print(layer_l.shape, layer_l[0, :10])
        #         # print(layer_u.shape, layer_u[0, :10])
        #     # print(full_assignment)
        #     output_mat, backsub_dict = self.transformer(full_assignment)
        #     # lhs = np.zeros([min(len(backsub_dict), len(assignment)), len(self.gurobi_vars)])
        #     # rhs = np.zeros(min(len(backsub_dict), len(assignment)))
        #     # for i, (node, status) in enumerate(assignment.items()):
        #     #     if node not in backsub_dict:
        #     #         continue
        #     #     if status:
        #     #         lhs[i] = -1 * backsub_dict[node][:-1]
        #     #         rhs[i] = backsub_dict[node][-1] - 1e-6
        #     #     else:
        #     #         lhs[i] = backsub_dict[node][:-1]
        #     #         rhs[i] = -1 * backsub_dict[node][-1]

        #     self.model.remove(self.model.getConstrs())x

        #     for node, status in assignment.items():
        #         if node not in backsub_dict:
        #             continue
        #         if status:
        #             self.model.addLConstr(self._get_equation(backsub_dict[node]) >= 1e-6)
        #         else:
        #             self.model.addLConstr(self._get_equation(backsub_dict[node]) <= 0)

        #     print(assignment, len(assignment), len(self.model.getConstrs()))
        # print(time.time() - tic)
        # # exit()
