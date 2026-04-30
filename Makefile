# Stochastic Block-Jacobi PageRank -- build system
# ------------------------------------------------
# Targets:
#   make            -> CPU + OpenMP build (no CUDA)
#   make cuda       -> CPU + OpenMP + CUDA  (needs nvcc)
#   make run-sample -> quick smoke test on data/sample_graph.txt
#   make clean

CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp -Iinclude
NVFLAGS  ?= -O3 -std=c++17 -Xcompiler -fopenmp -Iinclude
LDFLAGS  ?= -fopenmp
LDLIBS   ?=

SRC_CPU = src/main.cpp \
          src/graph_loader.cpp \
          src/utils.cpp \
          src/pagerank_sequential.cpp \
          src/pagerank_openmp.cpp \
          src/pagerank_hybrid.cpp

OBJ_CPU = $(SRC_CPU:.cpp=.o)

BIN     = build/pagerank

.PHONY: all cuda clean run-sample dirs

all: dirs $(BIN)

dirs:
	@mkdir -p results build

$(BIN): $(OBJ_CPU)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- CUDA build -------------------------------------------------------------
# Re-builds everything with -DSBJ_WITH_CUDA so the hybrid path uses real GPU.
cuda: dirs
	$(NVCC) $(NVFLAGS) -DSBJ_WITH_CUDA \
	    src/main.cpp \
	    src/graph_loader.cpp \
	    src/utils.cpp \
	    src/pagerank_sequential.cpp \
	    src/pagerank_openmp.cpp \
	    src/pagerank_hybrid.cpp \
	    src/pagerank_cuda.cu \
	    src/pagerank_hybrid_cuda.cu \
	    -o $(BIN)

run-sample: all
	./$(BIN) --mode seq    --input data/sample_graph.txt --iters 50
	./$(BIN) --mode omp    --input data/sample_graph.txt --iters 50
	./$(BIN) --mode hybrid --input data/sample_graph.txt --iters 50 --hd 5

clean:
	rm -f $(OBJ_CPU) $(BIN)
