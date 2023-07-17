from pprint import pprint
import numpy as np
import torch
import time
import copy

from dnn_solver.dnn_theorem_prover_cdcl import DNNTheoremProverCDCL
from heuristic.decision.decider import Decider
from sat_solver.cdcl_solver import CDCLSolver
from sat_solver.sat_solver import Solver
from utils.timer import Timers
import settings


class TheorySolver(Solver):

    def __init__(self, variables, layers_mapping, decider=None):
        super().__init__()

        self._solver = CDCLSolver(
            formula=None,
            variables=variables,
            layers_mapping=layers_mapping,
            decider=decider,
            theory_solver=self,
        )

    def get_assignment(self) -> dict:
        pass

    def solve(self, timeout=None) -> bool:
        return self._solver.solve(timeout)

    def set_early_stop(self, status):
        self._solver.set_early_stop(status)


class DNNSolverCDCL(TheorySolver):

    def __init__(self, net, spec, dataset):

        self.net = net

        layers_mapping = net.layers_mapping
        variables = [v for d in layers_mapping.values() for v in d]
        self.dataset = dataset

        self.decider = Decider(net, dataset)
        self.dnn_theorem_prover = DNNTheoremProverCDCL(net, spec=spec, decider=self.decider)

        super().__init__(variables=variables, layers_mapping=layers_mapping, decider=self.decider)
        self.start_time = time.time()


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
        theory_sat, implications, is_full_assignment = self.dnn_theorem_prover(assignment, full_assignment=full_assignment)
        Timers.toc('Theorem deduction')

        print(f'[{time.time()-self.start_time:.02f}]', self.dnn_theorem_prover.count, 'dnn_theorem_prover:', len([v for v, _, is_implied in self._solver.iterable_assignment() if not is_implied]), time.time() - tic, 'SAT' if theory_sat else 'UNSAT')

        if self.get_solution() is not None:
            self.set_early_stop('SAT')
            return conflict_clause, new_assignments

        if not theory_sat:
            self.dnn_theorem_prover.restore_input_bounds()

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
            return conflict_clause, new_assignments



        if is_full_assignment:
            return conflict_clause, new_assignments

        # deduce next layers
        for node in implications:
            if settings.DEBUG:
                print(f'    - `node {node} <= 0`:', implications[node]['neg'])
            
            if implications[node]['neg']:
                new_assignments.append(-node)
                continue

            if settings.DEBUG:
                print(f'    - `node {node} > 0`:', implications[node]['pos'])

            if implications[node]['pos']:
                new_assignments.append(node)

        if settings.DEBUG:
            print(f'\n- New assignment: `{new_assignments}`')
            print()
        return conflict_clause, new_assignments


    def get_assignment(self) -> dict:
        pass

    def get_solution(self):
        return self.dnn_theorem_prover.solution

