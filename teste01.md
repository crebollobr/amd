# Problema de Compilação: NAMD 3.0.2 e ROCm 7.2.0

Olá,

Estou tentando compilar o NAMD 3.0.2 para GPUs AMD (MI300X / arquitetura `gfx942`), mas o código-fonte atual parece não ser compatível com a versão 7.2.0 do ROCm. 

Abaixo estão os passos exatos que executei e o resumo do erro gerado.

## Comandos Executados

```bash
# 1. Instalação de dependências
apt install tcl-dev libfftw3-dev

# 2. Preparação e compilação do Charm++
rm -rf charm-8.0.0 Linux-x86_64-g++/
tar xvf charm-*.tar
cd charm-8.0.0/
./build charm++ multicore-linux-x86_64 --with-production -O -DNDEBUG -j $(nproc)
cd ..

# 3. Configuração do NAMD
./config Linux-x86_64-g++ \
  --charm-arch multicore-linux-x86_64 \
  --with-hip \
  --rocm-prefix /opt/rocm \
  --hipcub-prefix /opt/rocm \
  --rocprim-prefix /opt/rocm \
  --with-tcl --tcl-prefix /usr \
  --with-fftw3 --fftw-prefix /usr

# 4. Ajuste de caminhos e compilação final
cd Linux-x86_64-g++/
echo "TCLINCL = -I/usr/include/tcl" >> Make.config
make -j


src/ComputeBondedCUDAKernel.cu:45:21: error: 'tex1Dfetch<HIP_vector_type<float, 4>, nullptr>' is unavailable: The image/texture API not supported on the device
   45 |   const float4 t0 = tex1Dfetch<float4>(tex, i0);
      |                     ^
/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/texture_indirect_functions.h:42:37: note: 'tex1Dfetch<HIP_vector_type<float, 4>, nullptr>' has been explicitly marked unavailable here

(...)

fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated when compiling for gfx942.
make: *** [Make.depends:9504: obj/ComputeBondedCUDAKernel.o] Error 1


```bash
