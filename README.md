# cuda-mpi-openmp

### update

```bash
git pull
```

### compile & run
```bash
nvcc -o cuda_exe --std=c++11 -Werror cross-execution-space-call -lm main.cu
./cuda_exe
```

### Sign the src
```bash
gpg --pinentry-mode loopback -u "D873F5BD" -ab main.cu
git add main.cu.asc -f && git commit -m "added sign" && git push
```


### Testing
```bash
python run_test.py --binary_path_cuda <binary_path_cuda> [--binary_path_cpu <binary_path_cpu>] [--kernel_sizes <kernel_sizes> ("[[1, 32], [512, 512], [1024, 1024]]")] **kwargs
```

#### lab1
```bash
python run_test.py --binary_path_cuda ./lab1/src/cuda_exe_to_plot --binary_path_cpu ./lab1/src/cpu_exe_to_plot "[[1, 32], [512, 512], [1024, 1024]]"
```