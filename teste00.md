Aqui está o conteúdo do documento convertido para o formato Markdown compatível com o GitHub, organizado de forma clara e estruturada:

---

# Relatório de Teste e Compilação: AMD Instinct MI300X

## 1. Estatísticas de Uso de Software

Resumo da utilização de recursos por diferentes softwares e usuários.

| Software | Jobs | GPU-horas | % GPU | CPU-horas | Usuários |
| --- | --- | --- | --- | --- | --- |
| **NAMD** | 56 | 4,288 | 52.1% | 51,320 | 4 |
| **PyTorch** | 27 | 1,448 | 17.6% | 23,169 | 1 |
| **Python (genérico)** | 108 | 1,100 | 13.4% | 15,421 | 6 |
| **GROMACS** | 22 | 814 | 9.9% | 13,031 | 3 |
| **LAMMPS** | 13 | 297 | 3.6% | 2,388 | 2 |
| **AMBER** | 13 | 195 | 2.4% | 3,123 | 2 |
| **Quantum ESPRESSO** | 1 | 69 | 0.8% | 1,108 | 1 |
| **Código próprio** | 3 | 16 | 0.2% | 247 | 2 |
| **Script não encontrado** | 1 | 1 | 0.0% | 4 | 1 |
| **TOTAL** | **244** | **8,228** | **100%** | **109,812** | - |

### Status de Compatibilidade (12/03/2026)

* **OK:** PyTorch, GROMACS, NAMD, LAMMPS.
* **Problema/OK:** AMBER e Ambertools.
* **NOK:** Quantum ESPRESSO.

---

## 2. Configuração do Ambiente PyTorch (ROCm)

### Instalação via Docker

```bash
docker run -dit \
  --name pytorch \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  rocm/dev-ubuntu-22.04:6.2

```

### Instalação de Dependências

```bash
# Correção do erro de instalação do torch
pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# Bibliotecas auxiliares
apt install -y python3-pip ocl-icd-opencl-dev opencl-headers build-essential
pip3 install pyopencl

```

### Teste de Verificação (OpenCL)

**Arquivo:** `teste01.py`

```python
import pyopencl as cl
platforms = cl.get_platforms()
print("Plataformas OpenCL encontradas:\n")
for platform in platforms:
    print("Plataforma:", platform.name)
    devices = platform.get_devices()
    for dev in devices:
        print("  Dispositivo:", dev.name)
        print("  Tipo:", cl.device_type.to_string(dev.type))
        print("  Memória Global:", dev.global_mem_size // (1024**2), "MB")
        print("  Compute Units:", dev.max_compute_units)

```

---

## 3. Benchmarking de Performance

### Teste de Multiplicação de Matrizes

**Arquivo:** `teste02.py`

```python
import torch
import time

print("CUDA disponível?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

size = 12000
print(f"Criando matrizes {size}x{size}...")
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

print("Iniciando multiplicação pesada...")
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
end = time.time()

print("Tempo gasto:", end - start, "segundos")
print("Resultado final shape:", c.shape)

```

**Resultado Obtido:**

* **Tempo:** ~2.38 segundos.
* **Dispositivo:** Detectado como `cuda` (via camada de compatibilidade ROCm).

---

## 4. Compilação do GROMACS 2025.3 para MI300X

### Preparação do Container

```bash
docker run -dit --name gromacs --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/dev-ubuntu-22.04:6.2
docker exec -ti gromacs bash
apt update && apt install -y git build-essential gfortran cmake rocm-dev rocm-hip-sdk wget

```

### Script de Configuração CMake (Otimizado)

Este comando resolve problemas de linkagem com **HIP + MPI** e evita o erro de **vtable**.

```bash
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

### Principais Flags Explicadas

* `-DGMX_HIP_ARCH=gfx942`: Gera código específico para a arquitetura **CDNA3** (MI300X).
* `-DBUILD_SHARED_LIBS=OFF`: Resolve o erro de `undefined reference to vtable` ao forçar linkagem estática.
* `-DGMX_PME_GPU=ON`: Move o cálculo de *Particle Mesh Ewald* para a GPU.
* `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`: Desativa o LTO para evitar falhas de compilação conhecidas no ROCm 6.2.

---

## 5. Guia de Teste Pós-Instalação

### Verificação da GPU

Após compilar, verifique se o suporte HIP foi ativado:

```bash
gmx_mpi --version | grep -i "HIP"

```

### Execução de Simulação de Teste

```bash
# 1. Configurar ambiente
source /opt/gromacs-mi300x/bin/GMXRC

# 2. Rodar simulação (Nota: GROMACS 2025.3 via HIP requer PME na CPU em alguns casos)
export HIP_VISIBLE_DEVICES=0
mpirun -np 1 gmx_mpi mdrun -deffnm run -nb gpu -pme cpu

```

### Resumo de Suporte HIP no GROMACS

| Tipo de Cálculo | Suporte GPU (HIP) |
| --- | --- |
| **Nonbonded** | ✅ Sim |
| **PME** | ❌ Não (nesta versão/config) |
| **Bonded** | ❌ Não |

> **Dica:** Monitore o uso em tempo real com `watch -n 0.5 rocm-smi`. Valores de **Power > 300W** indicam processamento intenso na MI300X.
