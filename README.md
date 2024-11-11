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
python test.py <binary_path> <k_times> <max_vector_size> [<kernel_sizes> ([[1, 32], [512, 512], [1024, 1024]])]
```