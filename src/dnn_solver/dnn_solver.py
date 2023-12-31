from pprint import pprint
import numpy as np
import torch
import time
import copy

# from dnn_solver.dnn_theorem_prover_2 import DNNTheoremProver
from dnn_solver.dnn_theorem_prover_gurobi_4 import DNNTheoremProverGurobi
from dnn_solver.dnn_theorem_prover_crown import DNNTheoremProverCrown
from sat_solver.custom_sat_solver import CustomSATSolver
from sat_solver.sat_solver import Solver
from utils.dnn_parser import DNNParser
from heuristic.decision import decider
from utils.timer import Timers
from abstract.crown import arguments
import settings

class TheorySolver(Solver):

    def __init__(self, variables, layers_mapping, decider=None):
        super().__init__()

        self._solver = CustomSATSolver(formula=None,
                                       variables=variables,
                                       layers_mapping=layers_mapping,
                                       decider=decider,
                                       theory_solver=self)

    def get_assignment(self) -> dict:
        pass

    def solve(self, timeout=None) -> bool:
        return self._solver.solve(timeout)

    def set_early_stop(self, status):
        self._solver.set_early_stop(status)


class DNNSolver(TheorySolver):

    def __init__(self, net, spec, dataset, debug=False):

        self.net = net

        layers_mapping = net.layers_mapping
        variables = [v for d in layers_mapping.values() for v in d]
        self.dataset = dataset

        if arguments.Config["general"]["n_procs"] > 1:
            torch.set_num_threads(1)
            
        self.decider = decider.Decider(net, dataset)
        if net.n_input <= 50:
            self.dnn_theorem_prover = DNNTheoremProverGurobi(net, spec=spec, decider=self.decider)
        else:
            self.dnn_theorem_prover = DNNTheoremProverCrown(net, spec=spec, decider=self.decider)

        super().__init__(variables=variables, layers_mapping=layers_mapping, decider=self.decider)
        self.start_time = time.time()
        
        self.debug = debug


    def propagate(self):
        if settings.DEBUG:
            print('- Theory propagate\n')

        conflict_clause = None
        new_assignments = []

        if self.dnn_theorem_prover.verified:
            self.set_early_stop('UNSAT')
            return conflict_clause, new_assignments
        # exit()

        assignment = {k: v['value'] for k, v in self._solver._assignment.items()}

        if settings.DEBUG:
            print('- Assignment:', assignment)

        # theory checking
        full_assignment = {k: v for k, v, is_implied in self._solver.iterable_assignment() if not is_implied}

        # Timers.reset()
        tic = time.time()
        Timers.tic('Theorem deduction')
        theory_sat, implications, is_full_assignment = self.dnn_theorem_prover(assignment, info=self._solver.get_current_assigned_node(), full_assignment=full_assignment)
        Timers.toc('Theorem deduction')

        if arguments.Config["general"]["verbose"]:
            if time.time() - tic > 0.01 or 1:
                if hasattr(self.dnn_theorem_prover, 'domains'):
                    print(f'[{time.time()-self.start_time:.02f}]', self.dnn_theorem_prover.count, 'dnn_theorem_prover:', len([v for v, _, is_implied in self._solver.iterable_assignment() if not is_implied]), f'(valid domains={len([d for _, d in self.dnn_theorem_prover.domains.items() if d.valid])}/{len(self.dnn_theorem_prover.domains)})', time.time() - tic, 'SAT' if theory_sat else 'UNSAT')
                else:
                    print(f'[{time.time()-self.start_time:.02f}]', self.dnn_theorem_prover.count, 'dnn_theorem_prover:', len([v for v, _, is_implied in self._solver.iterable_assignment() if not is_implied]), time.time() - tic, 'SAT' if theory_sat else 'UNSAT')

        # Timers.print_stats()
        # print()
        # print()
        # for d in self.dnn_theorem_prover.domains.values():
        #     print('\t', d.lower_bound, d.get_assignment())
        # if self.dnn_theorem_prover.count >= 35:
        #     exit()

        if self.get_solution() is not None:
            self.set_early_stop('SAT')
            return conflict_clause, new_assignments

        # if self.dnn_theorem_prover.count == 50:
        #     exit()

        if not theory_sat:
            self.dnn_theorem_prover.restore_input_bounds()

            if hasattr(self.dnn_theorem_prover, 'optimized_layer_bounds'):
                self.dnn_theorem_prover.optimized_layer_bounds = {}

            if hasattr(self.dnn_theorem_prover, 'next_iter_implication'):
                self.dnn_theorem_prover.next_iter_implication = False

            conflict_clause  = set(implications)
            if len(conflict_clause):
                return conflict_clause, new_assignments
            # new_ccs = implications
            conflict_clause = set()
    
            # cac = set()
            for variable, value, is_implied in self._solver.iterable_assignment():
                if not is_implied:
                # if True:
                    conflict_clause.add(-variable if value else variable)
                # cac.add(-variable if value else variable)
            conflict_clause = frozenset(conflict_clause)
            if settings.DEBUG:
                print(f'    - Conflict clause: `{list(conflict_clause)}`')
                print()

            # print('cac:', list(cac))
            # print(self.dnn_theorem_prover.count, 'cc :', list(conflict_clause))
            # print()
            return conflict_clause, new_assignments

        if hasattr(self.dnn_theorem_prover, 'domains'):
            vd = [d for _, d in self.dnn_theorem_prover.domains.items() if d.valid]
            if len(vd) == 0 and self.dnn_theorem_prover.batch > 1 and not self.debug:
                self.set_early_stop('UNSAT')
                return conflict_clause, new_assignments

            if len(vd) > 100000:
                self.set_early_stop('TIMEOUT')
                return conflict_clause, new_assignments



        if is_full_assignment:
            return conflict_clause, new_assignments

        # deduce next layers
        if settings.DEBUG:
            print(f'\n- Implication')
            
        for node in implications:
            if implications[node]['neg']:
                new_assignments.append(-node)
                if settings.DEBUG:
                    print(f'    - `node {node} <= 0`')
                
                continue

            if implications[node]['pos']:
                new_assignments.append(node)
                if settings.DEBUG:
                    print(f'    - `node {node} > 0`')


        if settings.DEBUG:
            print(f'\n- New assignment: `{new_assignments}`')
            print('--------------------\n')
        return conflict_clause, new_assignments


    def get_assignment(self) -> dict:
        pass

    def get_solution(self):
        return self.dnn_theorem_prover.solution

