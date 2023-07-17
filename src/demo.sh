# UNSAT
python3 main.py --net benchmark/mnistfc/nnet/mnist-net_256x2.onnx --spec benchmark/mnistfc/spec/prop_0_0.03.vnnlib --dataset mnist --file result_mnist-net_256x2_prop_0_0.03.txt --device cuda --attack --verbose --timer
python3 main.py --net benchmark/mnistfc/nnet/mnist-net_256x2.onnx --spec benchmark/mnistfc/spec/prop_0_0.03.vnnlib --dataset mnist --file result_mnist-net_256x2_prop_0_0.03.txt --device cuda --attack --verbose --timer

# SAT
python3 main.py --net benchmark/mnistfc/nnet/mnist-net_256x2.onnx --spec benchmark/mnistfc/spec/prop_8_0.05.vnnlib --dataset mnist --file result_mnist-net_256x2_prop_8_0.05.txt --attack --timer --verbose

# SAT
python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_2_4_batch_2000.onnx --spec benchmark/acasxu/spec/prop_2.vnnlib --dataset acasxu --file result_ACASXU_run2a_2_4_batch_2000_prop_2.txt --attack
python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_2_4_batch_2000.onnx --spec benchmark/acasxu/spec/prop_2.vnnlib --dataset acasxu --file result_ACASXU_run2a_2_4_batch_2000_prop_2.txt --verbose --timer

# UNSAT
python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.onnx --spec benchmark/acasxu/spec/prop_1.vnnlib --dataset acasxu --file result_ACASXU_run2a_1_1_batch_2000_prop_1.txt --verbose