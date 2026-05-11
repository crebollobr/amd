Software Jobs GPU GPU-horas % GPU CPU-horas (desses jobs) Usuários  
NAMD 56 4,288 52.1% 51,320 4  
PyTorch 27 1,448 17.6% 23,169 1  
Python (genérico) 108 1,100 13.4% 15,421 6  
GROMACS 22 814 9.9% 13,031 3  
LAMMPS 13 297 3.6% 2,388 2  
AMBER 13 195 2.4% 3,123 2  
Quantum ESPRESSO 1 69 0.8% 1,108 1  
Código próprio 3 16 0.2% 247 2  
Script não encontrado 1 1 0.0% 4 1  
TOTAL 244 8,228 100% 109,812

TensorFlow

12/03/2026 NAMD e LAMMPS

# 

PyTorch OK  
GROMACS OK  
NAMD OK  
LAMMPS OK  
AMBER e Ambertools Problema OK  
ESPRESSO NOK

OpenMPI e MPICH  
TensorFlow  
BerkeleyGW  
CP2K  
QMCPACK

# 

# teste AMD

# PyTorch

# Instalação

docker run \-dit \\  
  \--name pytorch \\  
  \--device=/dev/kfd \\  
  \--device=/dev/dri \\  
  \--group-add video \\  
  \--ipc=host \\  
  \--shm-size 8G \\  
  rocm/dev-ubuntu-22.04:6.2

erro  pip3 install torch torchvision \--index-url [https://download.pytorch.org/whl/rocm7.1](https://download.pytorch.org/whl/rocm7.1)

pip3 install \--no-cache-dir torch torchvision \--index-url [https://download.pytorch.org/whl/rocm7.1](https://download.pytorch.org/whl/rocm7.1)

apt install \-y python3-pip ocl-icd-opencl-dev opencl-headers build-essential

pip3 install pyopencl

# Teste simples

Arquivo [teste01.py](http://teste01.py)

import pyopencl as cl

platforms \= cl.get\_platforms()

print("Plataformas OpenCL encontradas:\\n")

for platform in platforms:  
    print("Plataforma:", platform.name)

    devices \= platform.get\_devices()

    for dev in devices:  
        print("  Dispositivo:", dev.name)  
        print("  Tipo:", cl.device\_type.to\_string(dev.type))  
        print("  Memória Global:", dev.global\_mem\_size // (1024\*\*2), "MB")  
        print("  Compute Units:", dev.max\_compute\_units)  
        print()

**Resultado**

Plataforma: AMD Accelerated Parallel Processing  
  Dispositivo: gfx942:sramecc+:xnack-  
  Tipo: ALL | GPU  
  Memória Global: 196288 MB  
  Compute Units: 304

# Teste mais complexo

Aruivo [teste02.py](http://teste02.py)

import torch  
import time

print("CUDA disponível?", torch.cuda.is\_available())

device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")

print("Usando dispositivo:", device)

size \= 12000

print(f"Criando matrizes {size}x{size}...")

a \= torch.randn(size, size, device=device)  
b \= torch.randn(size, size, device=device)

print("Iniciando multiplicação pesada...")

start \= time.time()

c \= torch.matmul(a, b)

torch.cuda.synchronize()

end \= time.time()

print("Tempo gasto:", end \- start, "segundos")  
print("Resultado final shape:", c.shape)

python3 [teste02.py](http://teste02.py)

**Resultado**

CUDA disponível? True  
Usando dispositivo: cuda  
Criando matrizes 12000x12000...  
Iniciando multiplicação pesada...  
Tempo gasto: 2.3838930130004883 segundos  
Resultado final shape: torch.Size(\[12000, 12000\])

# GROMACS

docker run \-dit   \--name gromacs   \--device=/dev/kfd   \--device=/dev/dri   \--group-add video   \--ipc=host   \--shm-size 8G   rocm/dev-ubuntu-22.04:6.2

docker exec \-ti gromacs bash

apt install \-y git  
apt update  
apt install \-y build-essential gfortran  
apt install cmake

apt install rocm-dev rocm-hip-sdk

wget [https://ftp.gromacs.org/gromacs/gromacs-2025.3.tar.gz](https://ftp.gromacs.org/gromacs/gromacs-2025.3.tar.gz)

tar \-xvf gromacs-2025.3.tar.gz  
cd gromacs-2025.3

mkdir build  
cd build

cmake .. \\  
 \-DGMX\_BUILD\_OWN\_FFTW=ON \\  
 \-DGMX\_GPU=HIP \\  
 \-DGMX\_MPI=ON \\  
 \-DGMX\_OPENMP=ON \\  
 \-DGMX\_HIP\_ARCH=gfx942 \\  
 \-DGMX\_PME\_GPU=ON \\  
 \-DGMX\_BUILD\_SHARED\_EXE=OFF \\  
 \-DBUILD\_SHARED\_LIBS=OFF \\  
 \-DHIP\_ROOT\_DIR=/opt/rocm-6.2.0 \\  
 \-DCMAKE\_PREFIX\_PATH=/opt/rocm-6.2.0 \\  
 \-DCMAKE\_INTERPROCEDURAL\_OPTIMIZATION=OFF \\  
 \-DCMAKE\_C\_COMPILER=gcc \\  
 \-DCMAKE\_CXX\_COMPILER=g++ \\  
 \-DCMAKE\_INSTALL\_PREFIX=/opt/gromacs-mi300x \\  
 \-DCMAKE\_BUILD\_TYPE=Release

Perfeito. Vou explicar **linha por linha**, tecnicamente, o último `cmake` que resolve o problema de link com HIP \+ MPI na sua **AMD Instinct MI300X** ao compilar o **GROMACS**.

---

# **📌 O comando completo**

```shell
cmake .. \
 -DGMX_BUILD_OWN_FFTW=ON \
 -DGMX_GPU=HIP \
 -DGMX_MPI=ON \
 -DGMX_OPENMP=ON \
 -DGMX_HIP_ARCH=gfx942 \
 -DGMX_PME_GPU=ON \
 -DGMX_BUILD_SHARED_EXE=OFF \
 -DBUILD_SHARED_LIBS=OFF \
 -DHIP_ROOT_DIR=/opt/rocm-6.2.0 \
 -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 \
 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
 -DCMAKE_C_COMPILER=gcc \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=/opt/gromacs-mi300x \
 -DCMAKE_BUILD_TYPE=Release
```

---

# **🔎 Explicação detalhada**

---

## **`-DGMX_BUILD_OWN_FFTW=ON`**

Força o GROMACS a compilar a própria FFTW (CPU).

Isso evita:

* conflitos com FFTW do sistema  
* problemas de ABI  
* incompatibilidades MPI

Mesmo usando GPU, a FFTW CPU ainda é usada em partes auxiliares.

---

## **`-DGMX_GPU=HIP`**

Ativa backend GPU via HIP (ROCm).

Sem isso:

* GROMACS compila apenas CPU.

Com isso:

* kernels nonbonded são compilados para AMD GPU  
* backend HIP é ativado

---

## **`-DGMX_MPI=ON`**

Compila versão MPI (`gmx_mpi`).

Necessário para:

* multi-node  
* múltiplas GPUs  
* domínio paralelo

---

## **`-DGMX_OPENMP=ON`**

Ativa paralelismo OpenMP na CPU.

Importante porque:

* Thread-MPI não é ideal para HPC real  
* OpenMP melhora performance CPU+GPU híbrida

---

## **`-DGMX_HIP_ARCH=gfx942`**

Define arquitetura da GPU.

MI300X \= `gfx942`.

Isso faz o HIP gerar código otimizado especificamente para CDNA3.

Sem isso:

* compilação pode gerar código genérico  
* performance cai  
* pode até falhar runtime

---

## **`-DGMX_PME_GPU=ON`**

Move PME (Particle Mesh Ewald) para GPU.

Sem isso:

* PME roda na CPU  
* perde performance significativa

Com isso:

* Nonbonded \+ PME 100% na GPU  
* máximo aproveitamento da MI300X

---

# **🔥 Parte crítica que resolve seu erro**

---

## **`-DGMX_BUILD_SHARED_EXE=OFF`**

Evita gerar executáveis que dependem de bibliotecas compartilhadas externas.

Força linkagem mais direta.

---

## **`-DBUILD_SHARED_LIBS=OFF`**

🚨 Esse é o que resolve o erro de `vtable`.

Por padrão:

```
BUILD_SHARED_LIBS=ON
```

Isso cria:

```
libgromacs_mpi.so
```

Mas no backend HIP \+ MPI, alguns objetos GPU (como `Gpu3dFft::Impl`) não entram corretamente na shared lib.

Resultado:

```
undefined reference to vtable
```

Ao colocar:

```
BUILD_SHARED_LIBS=OFF
```

Tudo é linkado estaticamente dentro do executável final.

Isso elimina o problema de símbolos não resolvidos.

---

# **🔧 ROCm Paths**

---

## **`-DHIP_ROOT_DIR=/opt/rocm-6.2.0`**

Define onde está o HIP.

Importante porque ROCm 6.x instala em:

```
/opt/rocm-6.2.0
```

Não usar `/opt/rocm` evita problemas com symlink inexistente.

---

## **`-DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0`**

Diz ao CMake para procurar pacotes (rocfft, rocprim, etc) nesse prefixo.

Sem isso:

* CMake não encontra rocFFT  
* backend FFT fica inconsistente

---

# **🧠 Otimização / estabilidade**

---

## **`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`**

Desativa LTO (Link Time Optimization).

ROCm 6.2 \+ GCC \+ HIP frequentemente falha com LTO ativado.

Você viu no log:

```
lto-wrapper: warning
```

Desativar LTO evita link quebrado.

---

# **🧰 Compiladores**

---

## **`-DCMAKE_C_COMPILER=gcc`**

## **`-DCMAKE_CXX_COMPILER=g++`**

Força uso do GCC.

Não use `hipcc` aqui — o GROMACS gerencia HIP internamente.

---

# **📂 Instalação**

---

## **`-DCMAKE_INSTALL_PREFIX=/opt/gromacs-mi300x`**

Define onde o GROMACS será instalado.

Depois:

```shell
make install
source /opt/gromacs-mi300x/bin/GMXRC
```

---

# **🚀 Tipo de build**

---

## **`-DCMAKE_BUILD_TYPE=Release`**

Ativa:

* otimizações O3  
* desativa asserts  
* build ideal para produção

---

# **🎯 O que esse CMake garante**

* HIP totalmente ativado  
* PME na GPU  
* FFT consistente  
* Link estável  
* Sem erro de vtable  
* Estável para MI300X \+ ROCm 6.2

---

# **🧪 Depois de compilar**

Confirme:

```shell
gmx_mpi --version
```

Deve mostrar algo como:

```
GPU support: HIP
HIP architecture: gfx942
```

---

---

# **✅ TESTE LIMPO – DO ZERO ATÉ USAR GPU**

## **1️⃣ Criar diretório**

```shell
cd ~
rm -rf gpu-test
mkdir gpu-test
cd gpu-test
source /opt/gromacs-mi300x/bin/GMXRC
```

---

## **2️⃣ Criar caixa de água grande**

```shell
gmx_mpi solvate -cs spc216.gro -o water.gro -box 10 10 10
```

No final do output, anote o número de resíduos SOL.

---

## **3️⃣ Criar topologia**

Substitua XXXX pelo número de SOL mostrado no passo anterior:

```shell
cat > topol.top <<EOF
#include "amber99sb-ildn.ff/forcefield.itp"
#include "amber99sb-ildn.ff/spc.itp"

[ system ]
Water box

[ molecules ]
SOL XXXX
EOF
```

---

## **4️⃣ Criar arquivo de parâmetros**

```shell
cat > run.mdp <<EOF
integrator      = md
nsteps          = 500000
dt              = 0.002
cutoff-scheme   = Verlet
nstlist         = 20
rlist           = 1.2
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
tcoupl          = no
pcoupl          = no
constraints     = none
EOF
```

---

## **5️⃣ Gerar arquivo binário**

```shell
gmx_mpi grompp -f run.mdp -c water.gro -p topol.top -o run.tpr
```

---

## **6️⃣ Rodar usando GPU (modo compatível com HIP atual)**

⚠️ PME e bonded ainda não suportam GPU via HIP.

```shell
export HIP_VISIBLE_DEVICES=0

mpirun -np 1 gmx_mpi mdrun \
    -deffnm run \
    -nb gpu \
    -pme cpu
```

---

# **🔎 Confirmar uso da GPU**

Em outro terminal:

```shell
watch -n 0.5 rocm-smi
```

Se aparecer algo como:

* GPU% \> 20%  
* SCLK \> 1500 MHz  
* Power \> 300W

✔️ Está usando GPU corretamente.

---

# **📌 Resumo importante**

Para GROMACS 2025.3 \+ HIP:

| Tipo de cálculo | GPU |
| ----- | ----- |
| Nonbonded | ✅ |
| PME | ❌ |
| Bonded | ❌ |

Portanto, o comando correto é sempre:

```
-nb gpu -pme cpu
```

---

