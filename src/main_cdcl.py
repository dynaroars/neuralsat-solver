import argparse
import torch
import time
import os

from heuristic.falsification import randomized_falsification
from dnn_solver.dnn_solver_cdcl import DNNSolverCDCL
from utils.read_vnnlib import read_vnnlib_simple
from dnn_solver.spec import SpecificationVNNLIB
from utils.dnn_parser import DNNParser
from utils.timer import Timers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--spec', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acasxu', 'mnist', 'cifar', 'test'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--attack', action='store_true')
    parser.add_argument('--timer', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--file', type=str, default='res.txt')
    args = parser.parse_args()

    args.device = torch.device(args.device) # if args.dataset in ['mnist', 'cifar'] else torch.device('cpu')
    args.attack = True
    
    net = DNNParser.parse(args.net, args.dataset, args.device)
    spec_list = read_vnnlib_simple(args.spec, net.n_input, net.n_output)
    tic = time.time()

    if args.timer:
        Timers.reset()
        Timers.tic('Main')
        
    attacked = False
    status = 'UNKNOWN'

    Timers.tic('Random attack')
    for i, s in enumerate(spec_list):
        spec = SpecificationVNNLIB(s)
        rf = randomized_falsification.RandomizedFalsification(net, spec)
        stat, adv = rf.eval(timeout=0.5)
        if stat == 'violated':
            attacked = True
            status = 'SAT'
            # print(args.net, args.spec, status, time.time() - tic)
            break
    Timers.toc('Random attack')

    if not attacked:
        for i, s in enumerate(spec_list):
            spec = SpecificationVNNLIB(s)
            solver = DNNSolverCDCL(net, spec, args.dataset)
            status = solver.solve(timeout=args.timeout)
            if status in ['SAT', 'UNKNOWN']:
                break


    print(f'\n\n[{args.dataset}]', args.net, args.spec, status, f'{time.time()-tic:.02f}')
    # print(f'\tExport to: results/{args.dataset}/{args.file}\n')
    os.makedirs(f'results/{args.dataset}', exist_ok=True)
    with open(f'results/{args.dataset}/{args.file}', 'w') as fp:
        print(f'{status},{time.time()-tic:.02f}', file=fp)

    if args.timer:
        Timers.toc('Main')
        Timers.print_stats()
