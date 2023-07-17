import os

class DataCollector:
    
    def __init__(self):
        
        self.learned_clauses = []
        
        self.theory_implication_time = 0.0
        
        self.count_unsat_by_lp = 0
        
        self.count_unsat_by_abs = 0
        
        self.count_implication = 0
        
        
    def add(self, learned_clauses=None, theory_implication_time=None, unsat_by_lp=None, unsat_by_abs=None, implication=None, runtime=None):
        if learned_clauses is not None:
            # print('added:', list(learned_clauses))
            self.learned_clauses.append(learned_clauses)
            
        if theory_implication_time is not None:
            self.theory_implication_time += theory_implication_time
            
        if unsat_by_lp is not None:
            self.count_unsat_by_lp += 1
            
        if unsat_by_abs is not None:
            self.count_unsat_by_abs += 1
            
        if implication is not None:
            self.count_implication += implication
            
        if runtime is not None:
            self.runtime = runtime
            
            
    def dump(self, file):
        name = os.path.splitext(file)[0]
        print(name)
        with open(name + '.stat', 'w') as fp:
            print(self, file=fp)
            
            
        with open(name + '.clause', 'w') as fp:
            for c in self.learned_clauses:
                print(list(c), file=fp)
            
            
    def __repr__(self):
        return (
            'Data Collector:\n'
            f'\t- learned clauses: {len(self.learned_clauses)}\n'
            # f'\t- learned clauses: {self.learned_clauses}\n'
            # f'\t- learned clauses: {[len(_) for _ in self.learned_clauses]}\n'
            f'\t- unsat by lp: {self.count_unsat_by_lp}\n'
            f'\t- unsat by abs: {self.count_unsat_by_abs}\n'
            f'\t- implication: {self.count_implication}\n'
            f'\t- theory implication time: {self.theory_implication_time}\n'
            f'\t- runtime: {self.runtime}\n'
        )
            
            
            
collector = DataCollector()