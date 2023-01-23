all:
	nvcc main.c get_arg.c array.c sort.c cuda_sort.cu

clean:
	rm main