# POC: Quantum ESPRESSO em AMD Instinct MI300X

**Data**: 2026-05-04
**Plataforma**: DigitalOcean GPU Droplet (`134.199.200.60`, 1× MI300X 192 GB)
**Status**: ❌ **Não viável com toolchain livre em maio/2026**

---

## TL;DR

Tentamos rodar Quantum ESPRESSO (QE) com aceleração em uma MI300X usando ROCm 7.2 e, em seguida, ROCm 6.0.2. **Em nenhum caminho conseguimos uma execução estável**. O único stack que upstream considera validado é **CCE Fortran (HPE Cray) + ROCm 6.0/6.1**, e o CCE não é distribuído publicamente.

Para quem quer só o veredito:

- ✅ **Hardware OK**: MI300X funciona, ROCm 7.2 expõe `gfx942` corretamente.
- ✅ **Buildou um `pw.x`**: depois de 5 patches em `develop_omp5` + flag de linker.
- ❌ **Runtime quebrado**: `pw.x` aborta em `rocBLAS: invalid pointer` no primeiro passo do SCF.
- ❌ **Sem caminho com flang livre**: `amdflang-22` é strict demais; `flang-classic 17` (ROCm 6.0.2) não suporta as diretivas OpenMP 5 que o código usa.
- 🔒 **CCE Cray não tem trial**: requer contrato HPE ativo.

---

## Hardware e ambiente base

```text
Host          : DigitalOcean Droplet rocm-7-2-software-gpu-mi300x1-192gb-devcloud-atl1
SO            : Ubuntu 24.04.4 LTS, kernel 6.8
GPU           : 1× AMD Instinct MI300X VF, 192 GB HBM, gfx942
CPU/RAM       : 20 vCPU, 235 GB RAM
Disco         : 697 GB (611 GB livres)
ROCm          : 7.2.0 em /opt/rocm-7.2.0
Compiladores  : amdclang 22.0, amdflang 22.0 (LLVM/MLIR), gfortran 13.3
```

Imagem secundária usada no caminho 3:

```text
Imagem        : rocm/dev-ubuntu-22.04:6.0.2-complete
ROCm          : 6.0.2
Compilador    : AMD flang-classic 17.0.0 (PGI/legado, equivalente livre do AOMP)
```

---

## O que é o `develop_omp5`

QE upstream (`gitlab.com/QEF/q-e`) tem três níveis de suporte a GPU:

| Caminho | Compilador | Hardware | Status upstream (2026-05) |
|---|---|---|---|
| **CUDA / OpenACC** | NVHPC | NVIDIA | mantido, mainline (`develop`) |
| **OpenMP target offload** | CCE Fortran | AMD/Intel | **branch experimental** `develop_omp5` |
| **HIP nativo** | — | AMD | não existe |

A branch `develop_omp5`:

- Última atividade: **2025-06-21** (commit `50d1c488`).
- Mantida por `icarnimeo` (Ivan Carnimeo) na CNR-IOM.
- Tem MR aberto **!2679 — Eigensolver for AMD GPU offload**, com nota explícita: *"Developed and tested on LUMI with rocm/6.0.3 and **cce19** (container)"*.
- **Não é mirrorada para o GitHub** — só `develop`, `master`, `qe-6.3-backports` e `alternative_fftxlib` estão lá.

Em **2026-04-29**, mainline (`develop`) trouxe um commit relevante:

```
617c5dca  offload_type cannot be offload_kind_omp until actual omp implementation lands.
```

Tradução: o upstream **removeu** o caminho OMP do `develop` admitindo que a implementação não está pronta. Isso é confirmação oficial de que QE+AMD via OpenMP-offload não está viável fora de quem tem CCE.

---

## Tentativa 1 — ROCm 7.2 + amdflang-22 + QE 7.5 (mainline)

**Objetivo**: caminho mais simples, usar a release stable e habilitar `QE_ENABLE_OFFLOAD=ON`.

**Build**: morre cedo com:

```fortran
PW/src/orthoatwfc.f90:63:6: error:
    No specific subroutine of generic 'calbec' matches the actual arguments
    CALL calbec(offload_type, npw, vkb, wfcatom, becp)
```

**Causa raiz**: em `Modules/becmod.f90`, a `INTERFACE calbec` declara apenas as variantes `_acc` (CUDA/OpenACC) e `_cpu`. Quando `__OPENMP_GPU` está definido, `offload_type` vira `TYPE(offload_kind_omp)` — e **não existe nenhum `calbec_*_omp`**. O dispatch falha.

**Implicação**: o caminho OMP-offload em **QE 7.5 vanilla simplesmente não existe** — só o `develop_omp5` tem essas variantes.

---

## Tentativa 2 — ROCm 7.2 + amdflang-22 + `develop_omp5`

**Objetivo**: usar a branch oficial AMD de QE.

### Patches necessários para o build passar

| # | Arquivo | Problema | Correção |
|---|---|---|---|
| 1 | `UtilXlib/rocblas.f90` | `c_loc(x)` em arg sem `TARGET` (subrotinas `rocblas_dzger`, `rocblas_a2a_zaxpy`) | adicionar `TARGET` em 5 declarações |
| 2 | `FFTXlib/src/fft_scalar.hipFFT.f90` | `stream_ = 0` em `TYPE(C_PTR)`; `IF(stream_==0)`; arrays sem `TARGET` para `c_loc()` | trocar por `c_null_ptr` / `C_ASSOCIATED`; adicionar `TARGET` em `c, cout, f_d, FFT_DIM, DATA_DIM` |
| 3 | `FFTXlib/src/tg_gather.f90:245` | `is_device_ptr(tg_v_d)` em array Fortran (não C_PTR) | trocar por `has_device_addr(tg_v_d)` (OMP 5.1+) |
| 4 | `FFTXlib/src/fft_buffers.f90:69,73` | `!$omp allocate(aux) allocator(pinned_alloc)` em var SAVE module-scope (proibido por spec OMP) | comentar diretiva |
| 5 | `KS_Solvers/Davidson/divdc3_dev.f90` (novo) | `__divdc3` ausente no device-rt do ROCm 7.2 (cegterg usa divisão complexa em `!$omp target`) | criar módulo Fortran `bind(C, name="__divdc3")` com `!$omp declare target` |
| 6 | cmake (re-configure) | `ld.lld` rejeita módulos device-side duplicados quando linkam várias static libs | `-DCMAKE_EXE_LINKER_FLAGS="-Xoffload-linker --allow-multiple-definition -Wl,--allow-multiple-definition"` |

### Comando cmake final (que passa)

```bash
cmake .. \
  -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_Fortran_COMPILER=amdflang \
  -DQE_ENABLE_OPENMP=ON \
  -DQE_ENABLE_OFFLOAD=ON \
  -DQE_ENABLE_ROCM=ON \
  -DQE_ENABLE_MPI=OFF -DQE_ENABLE_TEST=OFF \
  -DQE_ENABLE_SCALAPACK=OFF -DQE_ENABLE_HDF5=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
  -DCMAKE_Fortran_FLAGS="-O2 -fopenmp --offload-arch=gfx942" \
  -DCMAKE_C_FLAGS="-O2 -fopenmp --offload-arch=gfx942" \
  -DCMAKE_EXE_LINKER_FLAGS="-Xoffload-linker --allow-multiple-definition -Wl,--allow-multiple-definition"
```

Resultado: **build chega a 100%**, gera `pw.x` (18 MB) que abre com banner correto:

```text
Program PWSCF v.7.4.1 starts on  4May2026 at 15: 7: 5
   Git branch: develop_omp5
   Last git commit: 50d1c488aa070cfcf3a3b27b74853626837c0adf-dirty
```

### Smoke test (Si bulk SCF) — falha em runtime

Input: célula primitiva de Si, `ecutwfc=20`, k-mesh 4×4×4.

```text
$ pw.x -i si.in
...
Starting wfcs are 8 randomized atomic wfcs
rocBLAS: invalid pointer
rocBLAS: invalid pointer
rocBLAS: invalid pointer
rocBLAS: invalid pointer
rocBLAS: invalid pointer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   Error in routine cdiaghg (9):
   S matrix not positive definite
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

GPU saiu de 0% → 7% e morreu. As cinco mensagens `rocBLAS: invalid pointer` confirmam que o problema é no marshalling do device pointer, não na BLAS em si.

### Causa raiz

O build emitiu, em vários sítios, warnings da forma:

```
warning: Use of non-C_PTR type 'a' in USE_DEVICE_PTR is deprecated,
         use USE_DEVICE_ADDR instead [-Wopen-mp-usage]
```

`amdflang-22` aceita o código antigo com warnings, mas **emite endereços incorretos** quando passa o array Fortran para a runtime OpenMP. O resultado: `rocblas_*` recebe um ponteiro do host onde esperava um device pointer. Isso é estrutural — **não é fixável com 5 patches pontuais**, requer migração de todas as ocorrências de `use_device_ptr` (em arrays não-`C_PTR`) para `use_device_addr` em todo o código.

---

## Tentativa 3 — Container ROCm 6.0.2 + flang-classic 17

**Hipótese**: o `develop_omp5` foi escrito para flang-classic permissivo (AOMP/CCE-style). Talvez um compilador legado aceite o código original sem precisar dos patches strict.

**Stack**:

```dockerfile
FROM rocm/dev-ubuntu-22.04:6.0.2-complete
# vem com AMD flang-classic 17.0.0 (linhagem PGI, mesma família do CCE)
RUN apt-get install -y cmake gfortran build-essential \
                       libopenblas-dev liblapack-dev libfftw3-dev
```

GPU passada via `--device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render`.

### Resultado

Sem nenhum patch: build morre em ~30%:

```
F90-S-0091-Constant expression of wrong data type
    (FFTXlib/src/fft_scalar.hipFFT.f90: 468)
flang: error: unable to execute command: Trace/breakpoint trap (core dumped)
```

Com os patches da Tentativa 2 aplicados: build morre em ~29%, **outro erro**:

```
F90-S-0034-Syntax error at or near :
    (FFTXlib/src/fft_types.f90: 335..344, 1039)
```

Olhando o código:

```fortran
#if defined (__OPENMP_GPU)
    !$omp target enter data map(always,alloc:desc%nsp)
    !$omp target enter data map(always,alloc:desc%nsw)
    ...
#endif
```

**flang-classic 17 não implementa o map-type-modifier `always` do OpenMP 5.0** — sintaxe que aparece em 26 sítios de `fft_types.f90` apenas. O `always` é central na lógica de transferência host↔device do `develop_omp5`; não dá pra removê-lo sem reescrever a estratégia de mapeamento.

---

## Diagnóstico final

A matriz que importa:

| Compilador Fortran | Free | OMP 5 `map(always,...)` | `use_device_ptr` em array Fortran (legado) | Build do `pw.x` | Runtime |
|---|---|---|---|---|---|
| **amdflang 22** (ROCm 7.2) | ✅ | ✅ aceita | ⚠️ aceita com warning, **gera código errado** | ✅ com 5 patches + flag linker | ❌ `rocBLAS invalid pointer` |
| **flang-classic 17** (ROCm 6.0.2) | ✅ | ❌ syntax error | ✅ aceita corretamente | ❌ não passa de 30 % | n/a |
| **CCE Fortran 19** (HPE Cray) | ❌ proprietário | ✅ | ✅ | ✅ | ✅ (validado pelo upstream) |

**O `develop_omp5` foi escrito para um ponto-doce que só o CCE ocupa**: aceita as features OMP 5 modernas (como amdflang novo), *e* mantém a interpretação permissiva de `use_device_ptr` em arrays Fortran (como flang-classic legado). Nenhum compilador livre em 2026 cobre os dois.

---

## Tem container pronto?

| Fonte | QE em AMD? |
|---|---|
| Docker Hub `rocm/*` (oficial AMD) | ❌ tem GROMACS e LAMMPS, **sem QE** |
| Docker Hub `amd/*` | ❌ não existe |
| AMD `HPCTrainingDock` | ❌ tem CP2K, GROMACS, LAMMPS, PyTorch, JAX, **sem QE** |
| NGC NVIDIA `hpc/quantum_espresso` | só NVIDIA, inútil para MI300X |
| `matteosecli/qe-singularity`, `anj1/qe-container` etc. | CPU-only ou NVIDIA |

A **AMD não publica imagem de QE pra MI300X** porque com a toolchain oficial deles (amdflang) o build não é estável — exatamente o que reproduzimos aqui.

---

## Tem trial do Cray CCE?

Não.

| Item | Disponibilidade |
|---|---|
| CCE como download standalone | ❌ não existe |
| Trial / evaluation gratuito | ❌ não existe |
| Container Docker oficial | ⚠️ Dockerfiles publicados, mas exigem **USS ISO** gated |
| Acesso ao HPE Support Center | 🔒 contrato HPE ativo + HPE Passport |
| Hardware necessário | ✅ commodity x86_64 (não precisa de Cray EX) — mas *"not officially supported"* |

Caminhos legítimos para chegar ao CCE:
1. **Alocação em Frontier** (DOE/ORNL) ou **LUMI** (EuroHPC, CSC Finlândia) — chamadas de propostas científicas com peer-review.
2. **Cliente HPE Cray** com contrato ativo.
3. Não há caminho razoável para um POC em droplet isolado.

---

## Releases / canais (GitHub mirror)

QE não usa "alpha" / "beta". Usa `-rc.X` quando faz release candidate.

| Tag | O que é |
|---|---|
| `qe-7.5` (e variante `qe-7.5CN`) | última stable |
| `qe-7.5-rc.1` | RC antes do 7.5 |
| `qe-7.4.1`, `qe-7.4`, `qe-7.3.1`, ... | releases anteriores |
| `qe-7.2-omp5-1.1-before_merge` | snapshot do trabalho OMP5 (defasado, 2022) |
| `qe-before-merge-qegpu` | snapshot histórico antes de merge antigo de QE-GPU |

Branches em `github.com/QEF/q-e`: apenas `develop`, `master`, `qe-6.3-backports`, `alternative_fftxlib`. **`develop_omp5` não é mirrorada** — só está no GitLab.

---

## Caminhos práticos a partir daqui

1. **POC com QE CPU-only**: build de `qe-7.5` vanilla com 20 cores OpenMP. Funciona, MI300X fica ociosa, mas serve como prova de conceito do *workload* e do *input*.
2. **Pivotar para CP2K**: container `rocm/cp2k` mantido pela AMD, suporte AMD GPU validado, cobre boa parte do que QE faz em DFT plane-wave/Gaussian. Mudança não-trivial mas factível.
3. **QMCPACK em MI300X**: validado em Frontier, OpenMP-offload funcional via amdclang. Outro caminho científico próximo.
4. **Pedir alocação em LUMI** se o objetivo é especificamente QE em produção AMD.
5. **Aguardar** — o commit `617c5dca` do upstream sinaliza que QE pretende reintroduzir OMP-offload "quando a implementação chegar". Sem timeline público.

---

## Sumário de uma linha

> **Quantum ESPRESSO em AMD Instinct MI300X com toolchain livre não está viável em maio de 2026.** O caminho oficial AMD do upstream (`develop_omp5`) compila com hacks mas roda só com Cray CCE; com `amdflang` ou `flang-classic` da AMD, ou o build não fecha ou o runtime entrega ponteiros inválidos para rocBLAS.

---

## Anexo: paths e estado do droplet

```
VM           : 134.199.200.60:2222 (root, chave /tmp/id_rsa_amber)
Workdir host : /root/qe-poc/
  q-e-omp5/      → checkout patcheado (Tentativa 2), build/bin/pw.x existe mas crasha
  qe-c-build/    → checkout container (Tentativa 3), não passa do build
Container    : qe-omp5-build (rodando, base rocm/dev-ubuntu-22.04:6.0.2-complete)
Logs build   : ~/qe-poc/build-omp5.log (host), /work/build.log (container)
Patches      : todos com .bak no diretório original
```

## Referências

- [QEF/q-e (GitLab)](https://gitlab.com/QEF/q-e) — repo upstream
- [QEF/q-e (GitHub mirror)](https://github.com/QEF/q-e) — read-only
- [MR !2679 — Eigensolver for AMD GPU offload](https://gitlab.com/QEF/q-e/-/merge_requests/2679) — em draft, testado só com CCE 19
- [Issue !716 — Compiler Support Status for AMD GPUs](https://gitlab.com/QEF/q-e/-/issues/716) — histórico do problema com flang
- [Quantum ESPRESSO towards performance portability — paper](https://www.sciencedirect.com/science/article/pii/S187705092401696X)
- [HPE CPE Container Guidance](https://cpe.ext.hpe.com/docs/latest/install/installation-guidance-container.html)
- [HPE CPE Token-Authed Repo (gated)](https://cpe.ext.hpe.com/docs/25.03/install/token-authed-repo.html)
- [LUMI access](https://www.lumi-supercomputer.eu/get-started/users/)
- [AMD HPCTrainingDock](https://github.com/amd/HPCTrainingDock) — sem QE
