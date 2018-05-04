FLAGS= -O3 -std=c++11

OUTDIR=./lib

CSRC=./src/cc/model/model.cc ./src/cc/matrix/matrix.cc ./src/cc/layers/*.cc
NCSRC=./src/cc/model/model.cu ./src/cc/matrix/matrix.cu ./src/cc/layers/*.cu

.PHONY: all mlp_cpu mlp_gpu mlp_test_cpu mlp_test_gpu clean

all: mlp_gpu

mlp_cpu:
	g++ $(FLAGS) -fPIC -shared -o $(OUTDIR)/mlp.so $(CSRC) ./src/cc/nn.cc

mlp_gpu:
	nvcc $(FLAGS) -Xcompiler -fPIC -shared -o $(OUTDIR)/mlp.so $(NCSRC) ./src/cc/nn.cc

mlp_test_cpu:
	g++ $(FLAGS) -fPIC -shared -o $(OUTDIR)/mlp_test.so $(CSRC) ./tests/model_tests/MNIST_test/nn.cc

mlp_test_gpu:
	nvcc $(FLAGS) -Xcompiler -fPIC -shared -o $(OUTDIR)/mlp_test.so $(NCSRC) ./tests/model_tests/MNIST_test/nn.cc

clean:
	rm -rf ./lib

$(shell mkdir -p lib)