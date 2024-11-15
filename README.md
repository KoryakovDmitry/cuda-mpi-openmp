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

### fast access
```bash
ssh -L 5322:0.0.0.0:5322 sedan
python3 -m http.server 5322
```

### Testing
```bash
python run_test.py --binary_path_cuda <binary_path_cuda> [--k_times <k_times>] [--binary_path_cpu <binary_path_cpu>] [--kernel_sizes <kernel_sizes> ("[[1, 32], [512, 512], [1024, 1024]]")] **kwargs
```

#### lab1
```bash
python run_test.py --binary_path_cuda ./lab1/src/cuda_exe_to_plot

python run_test.py --binary_path_cuda ./lab1/src/cuda_exe_to_plot --binary_path_cpu ./lab1/src/cpu_exe_to_plot --k_times 20 --kernel_sizes "[[1, 32], [512, 512], [1024, 1024]]"  --metadata_columns2plot '["vector_size"]'
```
#### lab2
```bash
python --binary_path_cuda ./lab2/src/to_plot_cuda_exe --k_times 20 --kernel_sizes "[[[512, 512], [1024, 1024]]]" --metadata_columns2plot '["filename"]' 
```