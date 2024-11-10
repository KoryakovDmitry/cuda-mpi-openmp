# cuda-mpi-openmp

### compile & run
```bash
nvcc -o cuda_exe main.cu
./cuda_exe
```

### Sign the src
```bash
gpg --pinentry-mode loopback -u "D873F5BD" -ab main.c
git add main.c.asc -f && git commit -m "added sign" && git push
```