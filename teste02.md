# NAMD 3.0.2 GPU-Resident Build para AMD MI300X (ROCm 7.x)

Este guia documenta o processo testado e validado para compilar o NAMD 3.0.2 com suporte nativo a GPU-Resident em aceleradores AMD (arquitetura `gfx942` / MI300X) usando o ROCm 7.x, contornando a remoção de APIs legadas de textura (`tex1Dfetch`).

## 1. Dependências do Sistema

Instale os pacotes básicos necessários para o NAMD:

```bash
apt-get update
apt-get install tcl-dev libfftw3-dev
```

## 2. Compilação do Charm++ (Modo Multicore)

Para simulações em nó único com GPU, a arquitetura `multicore` é obrigatória (não utilize `netlrts` ou `smp`).

```bash
tar xvf charm-8.0.0.tar
cd charm-8.0.0/

./build charm++ multicore-linux-x86_64 --with-production -j $(nproc)

cd ..
```

## 3. Configuração do NAMD

A flag `--with-single-node-hip` é crucial para habilitar o modo GPU-Resident em compilações *multicore*.

```bash
./config build-multicore-gpu/Linux-x86_64-g++ \
  --charm-arch multicore-linux-x86_64 \
  --with-hip \
  --with-single-node-hip \
  --rocm-prefix /opt/rocm \
  --hipcub-prefix /opt/rocm \
  --rocprim-prefix /opt/rocm \
  --with-tcl --tcl-prefix /usr \
  --with-fftw3 --fftw-prefix /usr
```

## 4. Correção para ROCm 7.x e Compilação Final

O ROCm 7.x removeu o suporte a certas APIs de textura de imagem. Forçamos o uso de arrays de tabela (`USE_TABLE_ARRAYS`) para corrigir o erro de compilação `tex1Dfetch`.

```bash
cd build-multicore-gpu/Linux-x86_64-g++/

# Inserir correções no Make.config
echo "EXTRADEFINES+=-DUSE_TABLE_ARRAYS" >> Make.config
echo "CUDAFLAGS+=-DUSE_TABLE_ARRAYS" >> Make.config
echo "TCLINCL = -I/usr/include/tcl" >> Make.config

# Compilar
make -j $(nproc)
```

## 5. Benchmark de Validação: STMV

Para validar a performance máxima da GPU, utilizamos o benchmark do **Satellite Tobacco Mosaic Virus (STMV)**, contendo mais de 1 milhão de átomos. Ele é o padrão-ouro para estressar a memória HBM3 e os Compute Units da MI300X.

* **Download:** [https://www.ks.uiuc.edu/Research/namd/benchmarks/](https://www.ks.uiuc.edu/Research/namd/benchmarks/) (`stmv_gpu.tar.gz`)
* **Motivo:** Garante a saturação completa da GPU e certifica que o modo *GPU-Resident* está operando de forma otimizada.

## 6. Execução Otimizada (Sem Charmrun)

Com a build `multicore`, ignoramos o uso de redes virtuais (`charmrun`). Chame o binário diretamente mapeando as threads da CPU e vinculando a GPU.

**Script de execução (`run.sh`):**

```bash
#!/bin/bash
NAMD_BIN=/root/NAMD_3.0.2_Source/build-multicore-gpu/Linux-x86_64-g++/namd3

# +p8: 8 threads de CPU
# +setcpuaffinity: Evita troca de contexto
# +devices 0: Vincula à primeira GPU MI300X
$NAMD_BIN +p8 +setcpuaffinity +devices 0 stmv_gpures_npt.namd | tee output_stmv_mi300x.log

Monitore a execução com `rocm-smi` em outro terminal. O uso da VRAM deve ficar em torno de 5~6GB e o uso da GPU em 100%. A métrica de performance final (ns/day) aparecerá no log.
```
