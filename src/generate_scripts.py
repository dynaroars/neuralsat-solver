import os


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
            
            
if __name__ == "__main__":
    BENCHMARK = 'acasxu'
    TIMEOUT = 1000
    CMD = 'python3 main_cdcl.py --net {} --spec {} --dataset {} --device cuda --file {} --timeout {}'
    
    benchmark_dir = f'benchmark/{BENCHMARK}'
    csv = f'{benchmark_dir}/instances.csv'
    
    with open(f'run_{BENCHMARK}.sh', 'w') as fp:
        for line in open(csv).read().strip().split('\n'):
            net_name, spec_name, timeout = line.split(',')
            net = os.path.join(benchmark_dir, net_name)
            spec = os.path.join(benchmark_dir, spec_name)
                
            if not (os.path.exists(net) and os.path.exists(spec)):
                print('skip', BENCHMARK, net_name, spec_name)
                continue
            
            file = f'{os.path.basename(net_name)[:-5]}_{os.path.basename(spec_name)[:-7]}.txt'
        
            print(CMD.format(net, spec, BENCHMARK, file, TIMEOUT), file=fp)