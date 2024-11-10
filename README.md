# cuda-mpi-openmp

### update

```bash
git pull
```

### compile & run
```bash
nvcc -o cuda_exe main.cu
./cuda_exe
```

### Sign the src
```bash
gpg --pinentry-mode loopback -u "D873F5BD" -ab main.cu
git add main.cu.asc -f && git commit -m "added sign" && git push
```