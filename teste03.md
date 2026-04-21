# Compilação LAMMPS - AMD MI300X (ROCm/Kokkos)

### 1. Variáveis de Ambiente
```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### 2. Dependências do Sistema
```bash
apt-get update
apt-get install -y mpich libmpich-dev libfftw3-dev build-essential cmake wget
```

### 3. Download e Extração do Código Fonte (Versão 22Jul2025)
```bash
wget https://download.lammps.org/tars/lammps-22Jul2025.tar.gz
tar -xzvf lammps-22Jul2025.tar.gz
cd lammps-22Jul2025
mkdir build && cd build
```

### 4. Configuração CMake (Otimizada para MI300X/gfx942)
```bash
cmake ../cmake \
    -D BUILD_MPI=yes \
    -D PKG_KOKKOS=yes \
    -D Kokkos_ENABLE_HIP=yes \
    -D Kokkos_ARCH_VEGA942=yes \
    -D DOWNLOAD_KOKKOS=yes \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -D PKG_KSPACE=yes \
    -D FFT=FFTW3 \
    -D FFT_KOKKOS=HIPFFT \
    -D PKG_MANYBODY=yes \
    -D PKG_MOLECULE=yes \
    -D CMAKE_BUILD_TYPE=Release
```

### 5. Compilação
```bash
make -j $(nproc)
```

### 6. Script de Teste de Estresse (4 milhões de átomos)
Criar arquivo `in.stress`:
```lammps
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 100 0 100 0 100
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.44 87287 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no
fix             1 all nve
thermo          100
run             2000
```

### 7. Execução do Benchmark
```bash
./lmp -k on g 1 -sf kk -in in.stress
```

### 8. Monitoramento de Hardware (Opcional - outro terminal)
```bash
watch -n 1 rocm-smi
```
