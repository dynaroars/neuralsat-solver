'''
    clone from: https://github.com/AvivYaish/SMTsolver/blob/master/smt_solver/sat_solver/sat_solver.py
'''

from collections import deque, Counter
from pprint import pprint
import sortedcontainers
import random
import copy
import time


class CDCLSolver:

    def __init__(self, formula, variables=None, max_new_clauses=float('inf'), halving_period=float('inf'),
                 layers_mapping=None, decider=None, theory_solver=None
            ):
        if formula is None:
            self._formula = set()
        else:
            self._formula = formula

        # theory
        self._theory_solver = theory_solver
        self._decider = decider
        
        self._max_new_clauses = max_new_clauses
        self._halving_period = halving_period

        self._assignment = dict()
        self._assignment_by_level = []
        self._satisfaction_by_level = []
        self._literal_to_clause = {}
        self._satisfied_clauses = set()
        self._last_assigned_literals = deque()
        self._variable_to_watched_clause = {} 

        self._all_vars = sortedcontainers.SortedList(variables)

        for var in self._all_vars:
            self._formula.add(frozenset({var, -var}))

        for clause in self._formula:
            self._add_clause(clause)
        
        # print(variables)
        # exit()
        self.initialized = False
        
        self._learned_conflict_clauses = set()

        
    def _add_clause(self, clause):
        """
        Initialize all clause data structures for the given clause.
        """
        for idx, literal in enumerate(clause):
            variable = abs(literal)
            if literal not in self._literal_to_clause:
                self._literal_to_clause[literal] = set()
            if variable not in self._variable_to_watched_clause:
                self._variable_to_watched_clause[variable] = set()

            self._literal_to_clause[literal].add(clause)
            if idx <= 1:
                self._variable_to_watched_clause[variable].add(clause)

    def _assign(self, clause, literal: int, is_implied=False):
        """
        Assigns a satisfying value to the given literal.
        """
        variable = abs(literal)
        self._assignment[variable] = {
            "value": literal > 0,                           # Satisfy the literal
            "clause": clause,                               # The clause which caused the assignment
            "level": len(self._assignment_by_level) - 1,    # The decision level of the assignment
            "idx": len(self._assignment_by_level[-1]),      # Defines an assignment order in the same level
            "is_implied": is_implied
        }

        self._all_vars.discard(variable)

        # Keep data structures related to satisfied clauses up to date
        newly_satisfied_clauses = self._literal_to_clause[literal] - self._satisfied_clauses
        self._satisfaction_by_level[-1].extend(newly_satisfied_clauses)
        self._satisfied_clauses |= newly_satisfied_clauses

        # Keep data structures related to variable assignment up to date
        self._assignment_by_level[-1].append(variable)
        self._last_assigned_literals.append(literal)



    def _unassign(self, variable: int):
        """
        Unassigns the given variable.
        """
        del self._assignment[variable]
        self._all_vars.add(variable)

    def get_assignment(self) -> dict:
        return {var: val for var, val, _ in self.iterable_assignment()}

    def get_variable_assignment(self, variable) -> bool:
        return self._assignment.get(variable, {"value": None})["value"]

    def iterable_assignment(self):
        """
        :return: a (variable: int, value: bool) tuple for every assigned variable.
        """
        for var in self._assignment:
            yield var, self._assignment[var]["value"], self._assignment[var]["is_implied"]

    def _find_last_literal(self, clause, removed_vars=[]):
        """
        :return: the last assigned literal in the clause, the second highest assignment level of literals in the clause,
        and the number of literals from the highest assignment level.
        """
        last_literal, prev_max_level, max_level, max_idx, max_level_count = None, -1, -1, -1, 0
        for literal in clause:
            variable = abs(literal)

            # FIXME: looks not right
            if (variable in removed_vars) or (variable not in self._assignment):
                continue

            level, idx = self._assignment[variable]["level"], self._assignment[variable]["idx"]
            if level > max_level:
                prev_max_level = max_level
                last_literal, max_level, max_idx, max_level_count = literal, level, idx, 1
            elif level == max_level:
                max_level_count += 1
                if idx > max_idx:
                    last_literal, max_idx = literal, idx
            elif level > prev_max_level:
                prev_max_level = level

        if (prev_max_level == -1) and (max_level != -1):
            prev_max_level = max_level - 1
        # print('- [Find]:', last_literal, prev_max_level, max_level, max_level_count)
        return last_literal, prev_max_level, max_level, max_level_count

    def _conflict_resolution(self, conflict_clause):
        """
        Learns conflict clauses using implication graphs, with the Unique Implication Point heuristic.
        """
        # print('--------------_conflict_resolution--------------')
        conflict_clause = set(conflict_clause)
        # print('conflict_clause:', conflict_clause)
        # if frozenset(conflict_clause) not in self._generated_conflict_clauses:
        #     self._add_conflict_clause(frozenset(conflict_clause))
        
        removed_vars = []
        while True:
            last_literal, prev_max_level, max_level, max_level_count = self._find_last_literal(conflict_clause, removed_vars)
            if last_literal is None:
                return None, None, -1
            clause_on_incoming_edge = self._assignment[abs(last_literal)]["clause"]
            # print(last_literal, prev_max_level, max_level, max_level_count, clause_on_incoming_edge)
            if (max_level_count == 1) or (clause_on_incoming_edge is None):
                if max_level_count != 1:
                    # If the last literal was assigned because of the theory, there is no incoming edge
                    # The literal to reassign should be the decision literal of the same level
                    # print('    - last_literal before:', last_literal)
                    last_literal = self._assignment_by_level[max_level][0]
                    # print(self._assignment_by_level[max_level], conflict_clause)
                    # print('    - last_literal after:', last_literal)
                    if self._assignment[last_literal]["value"]:
                        last_literal = -last_literal
                    conflict_clause.add(last_literal)
                # If the last assigned literal is the only one from the last decision level:
                # return the conflict clause, the next literal to assign (which should be the
                # watch literal of the conflict clause), and the decision level to jump to
                return frozenset(conflict_clause), last_literal, prev_max_level

            # Resolve the conflict clause with the clause on the incoming edge
            # Might be the case that the last literal was assigned because of the
            # theory, and in that case it is impossible to do resolution
            # print('conflict_clause:', conflict_clause, 'last_literal:', last_literal)
            conflict_clause |= clause_on_incoming_edge
            # print()
            conflict_clause.remove(last_literal)
            conflict_clause.remove(-last_literal)
            removed_vars.append(abs(last_literal))

    def _bcp(self):
        """
        Performs BCP, as triggered by the last assigned literals. If new literals are assigned as part of the BCP,
        the BCP continues using them. The BCP uses watch literals.
        :return: None, if there is no conflict. If there is one, the conflict clause is returned.
        """
        # print('[bcp]', self._last_assigned_literals)
        
        # print('\t[variable to watched clause]')
        # for k, v in self._variable_to_watched_clause.items():
        #     print(k, '--->', v)
        # print()

        # print('\t[_satisfied_clauses]')
        # for k, v in enumerate(self._satisfied_clauses):
        #     print(k, '--->', v)
        # print()

        while self._last_assigned_literals:
            watch_literal = self._last_assigned_literals.popleft()
            # print('watch_literal:', watch_literal)
            for clause in self._variable_to_watched_clause[abs(watch_literal)].copy():
                if clause not in self._satisfied_clauses:
                    conflict_clause = self._replace_watch_literal(clause, watch_literal)
                    if conflict_clause is not None:
                        return conflict_clause
        return None  # No conflict-clause


    def _tcp(self):
        """
        Theory constraint propagation.
        """
        conflict_clause, new_assignments = self._theory_solver.propagate()
        if conflict_clause is not None:
            return conflict_clause
        for literal in new_assignments:
            self._assign(None, literal, is_implied=True)
            # print(f'- [Imply] {abs(literal)}={literal}')
        return None
    
    
    def _replace_watch_literal(self, clause, watch_literal: int):
        """
        - If the clause is satisfied, nothing to do.
        - Else, it is not satisfied yet:
          - If it has 0 unassigned literals, it is UNSAT.
          - If it has 1 unassigned literals, assign the correct value to the last literal.
          - If it has > 2 unassigned literals, pick one to become the new watch literal.
        """
        watch_variable, replaced_watcher, unassigned_literals = abs(watch_literal), False, []
        for unassigned_literal in clause:
            unassigned_variable = abs(unassigned_literal)
            if unassigned_variable in self._assignment:
                # If the current literal is assigned, it cannot replace the current watch literal
                continue
            unassigned_literals.append(unassigned_literal)

            if replaced_watcher:
                # If we already replaced the watch_literal
                if len(unassigned_literals) > 1:
                    break
            elif clause not in self._variable_to_watched_clause[unassigned_variable]:
                # If the current literal is not already watching the clause, it can replace the watch literal
                self._variable_to_watched_clause[watch_variable].remove(clause)
                self._variable_to_watched_clause[unassigned_variable].add(clause)
                replaced_watcher = True

        if len(unassigned_literals) == 0:
            # Clause is UNSAT, return it as the conflict-clause
            return clause
        if len(unassigned_literals) == 1:
            # The clause is still not satisfied, and has only one unassigned literal.
            # Assign the correct value to it. Because it is now watching the clause,
            # and was also added to self._last_assigned_literals, we will later on
            # check if the assignment causes a conflict
            literal = unassigned_literals.pop()
            self._assign(clause, literal)
            # print(f'\t- [c] {clause}, {literal}')
        return None

    def _add_conflict_clause(self, conflict_clause):
        """
        Adds a conflict clause to the formula.
        """
        # save conflict clauses
        print('conflict_clause:', conflict_clause)
        self._learned_conflict_clauses.add(conflict_clause)
        
        # bcp
        self._add_clause(conflict_clause)


    def _constraint_propagation_to_exhaustion(self, propagation_func):
        """
        Performs constraint propagation using the given function
        until exhaustion, returns False iff formula is UNSAT.
        """
        # print('- [BCP + TCP]', propagation_func)
        conflict_clause = propagation_func()
        while conflict_clause is not None:
            conflict_clause, watch_literal, level_to_jump_to = self._conflict_resolution(conflict_clause)
            # print(self.get_assignment(), conflict_clause)
            # print('watch_literal:', watch_literal, '\nlen:', len(list(conflict_clause)), '\nconflict_clause:', conflict_clause)
            # print(len(list(conflict_clause)), conflict_clause)
            # print()
            if level_to_jump_to == -1:
                # An assignment that satisfies the formula's unit clauses causes a conflict, so the formula is UNSAT
                return False
            
            self.backtrack(level_to_jump_to)
            self._add_conflict_clause(conflict_clause)
            self._assign(conflict_clause, watch_literal)
            # print(f'- [Conflict] {abs(watch_literal)}={watch_literal}, dl={level_to_jump_to}')
            # print(f'\t- [cc] {conflict_clause}, {watch_literal}')

            conflict_clause = propagation_func()
        return True


    def propagate(self) -> bool:
        if (self._theory_solver is not None) and (not self.initialized):
            self.initialized = True
            if not self._constraint_propagation_to_exhaustion(self._tcp):
                return False
            
        while self._last_assigned_literals:
            if (not self._constraint_propagation_to_exhaustion(self._bcp)) or \
                ((self._theory_solver is not None) and (not self._constraint_propagation_to_exhaustion(self._tcp))):
                return False
        return True

    def backtrack(self, level: int):
        """
        Non-chronological backtracking.
        """
        self._last_assigned_literals = deque()
        while len(self._assignment_by_level) > level + 1:
            for variable in self._assignment_by_level.pop():
                self._unassign(variable)
            for clause in self._satisfaction_by_level.pop():
                self._satisfied_clauses.remove(clause)

    def _decide(self):
        unassigned_variables = list(self._all_vars)
        if self._decider is None:
            variable, value = unassigned_variables[0], True
        else:
            variable, value = self._decider._get(unassigned_variables)
        assert variable not in self._assignment, print(variable, unassigned_variables)
        self.create_new_decision_level()
        self._assign(None, variable if value else -variable)


    def create_new_decision_level(self):
        self._assignment_by_level.append(list())
        self._satisfaction_by_level.append(list())


    def _satisfy_unit_clauses(self):
        # print('unit bcp')
        self.create_new_decision_level()
        for clause in self._formula:
            if len(clause) == 1:
                for literal in clause:
                    if abs(literal) not in self._assignment:
                        self._assign(clause, literal)

    def _is_sat(self) -> bool:
        return self._formula.issubset(self._satisfied_clauses)


    def solve(self, timeout=None) -> bool:
        self._satisfy_unit_clauses()
        start = time.perf_counter()
        while True:
            if timeout is not None:
                if time.perf_counter() - start >= timeout:
                    return 'TIMEOUT'
            if not self.propagate():
                return 'UNSAT'
            if self._is_sat():
                print(self.get_assignment())
                return 'SAT'
            self._decide()


if __name__ == "__main__":
    formula = set([
        frozenset([-2, -3, -4, 5]),
        frozenset([-1, -5, 6]),
        frozenset([-5, 7]),
        frozenset([-1, -6, -7]),
        frozenset([-1, -2, 5]),
        frozenset([-1, -3, 5]),
        frozenset([-1, -4, 5]),
        frozenset([-1, 2, 3, 4, 5, -6]),
    ])
    
    variables = list(sorted(set([abs(_) for i in formula for _ in list(i)])))
    
    solver = CDCLSolver(formula=formula, variables=variables)
    print(solver.solve())
    print(solver.get_assignment())
    pprint(solver._assignment)