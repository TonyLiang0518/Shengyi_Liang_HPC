CXX = nvcc
FLAG1 = -arch=sm_61
FLAG2 = -Xcompiler -fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(FLAG1) $< $(LIBS) -o $@ $(FLAG2)

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean