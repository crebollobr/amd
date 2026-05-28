# OpenFOAM em GPU AMD MI300 — Estado da Arte e Plano de PoC

> Documento de referência para PoC de OpenFOAM em GPU AMD Instinct MI300.
> Data: 2026-05-28

---

## TL;DR

**"100% OpenFOAM rodando em GPU" não existe em produção hoje.**
O OpenFOAM mainline (ESI/Foundation) é fundamentalmente CPU + MPI. Existem três caminhos para envolver a MI300, com graus muito diferentes de maturidade.

| Caminho | O que vai pra GPU | Maturidade | Recomendado para PoC? |
|---|---|---|---|
| **petsc4Foam + PETSc/ROCm** | Solver linear (~60–80% do runtime) | Produção | **Sim** — primeiro passo |
| **neoFOAM (Kokkos/HIP)** | Tudo (GPU-native) | Experimental | Como segundo experimento |
| **RapidCFD / forks HIP** | Solver + alguns kernels | Abandonado | Não |

---

## 1. Por que não dá pra rodar OpenFOAM "100% GPU" hoje

O OpenFOAM foi escrito ao longo de ~20 anos como uma biblioteca C++ orientada a objetos para CPU, com paralelismo via MPI (domain decomposition). As estruturas centrais (`fvMesh`, `volField`, `surfaceField`, operadores `fvm::`/`fvc::`) assumem memória host e laços sequenciais sobre células/faces.

Portar isso para GPU exige reescrever:

- **Armazenamento de campos** (precisa viver na memória GPU)
- **Montagem da matriz** (loop sobre faces → kernel paralelo)
- **Discretização** (esquemas div/grad/laplaciano)
- **Boundary conditions** (cada BC é uma classe própria)
- **Solver linear** (essa parte sim já tem soluções maduras em GPU)
- **I/O, parallel comms, decomposição**

Por isso a indústria foca no **solver linear**, que é o pedaço mais pesado e mais fácil de isolar.

---

## 2. Os três caminhos em detalhe

### 2.1. petsc4Foam + PETSc/ROCm  ← **recomendado para PoC**

**O que é:** plugin oficial do OpenFOAM (mantido pela ESI + parceiros) que substitui os solvers lineares nativos por chamadas ao PETSc. O PETSc, por sua vez, pode ser compilado com backend ROCm/HIP (via hipSPARSE, hipBLAS, AmgX-equivalentes) e executar GMRES/CG/BiCGStab + precondicionadores (ILU, BoomerAMG via hypre/ROCm) na GPU.

**Quanto da app vai pra GPU:** o solver linear, que tipicamente é 60–80% do tempo de simulação em casos grandes (especialmente pressão em simpleFoam/pimpleFoam).

**Por que é o caminho certo agora:**
- É produção, não pesquisa.
- AMD publica benchmarks oficiais usando exatamente essa stack na linha MI.
- Toolchain bem definida: ROCm + PETSc-ROCm + petsc4Foam + OpenFOAM padrão.
- O case roda igual ao CPU — você muda só o `system/fvSolution` para apontar pro PETSc.
- Gera número comparável (speedup vs. CPU puro, escalabilidade).

**Limitações:**
- Não é "100% GPU" — montagem da matriz e operadores ainda na CPU, com cópia host↔device a cada iteração externa (overhead real).
- Speedup depende fortemente do case: malhas pequenas perdem para CPU pelo overhead de transferência.
- Precondicionador AMG em GPU AMD ainda é menos otimizado que em NVIDIA (gap está fechando, mas existe).

### 2.2. neoFOAM (Kokkos + HIP)

**O que é:** iniciativa nova (ESI + comunidade acadêmica) de reescrever o OpenFOAM com design GPU-native usando Kokkos como camada de portabilidade. Roda em CPU, CUDA, HIP (ROCm), SYCL.

**Quanto vai pra GPU:** o objetivo é tudo — campos, operadores, BCs, solver. Esse é o único caminho que pode honestamente ser chamado de "OpenFOAM 100% GPU".

**Status real (2026):**
- Cobertura de solvers muito limitada (basicamente alguns casos demo de Laplace/advection-diffusion).
- Não roda casos industriais ainda — falta a maior parte dos schemes, BCs, modelos de turbulência, etc.
- API ainda mudando.
- Ótimo como **prova de conceito de pesquisa** e para acompanhar a direção futura do projeto. Ruim para validar workload real hoje.

**Quando faria sentido:** segundo experimento, depois do petsc4Foam, com expectativa explícita de "estou testando o futuro, não o presente".

### 2.3. RapidCFD e forks HIP

**O que é:** RapidCFD foi um fork (~2014, da empresa Symscape) que portou parte do simpleFoam para CUDA. Houve esforços comunitários de HIP-ificar para AMD.

**Status:** abandonado/estagnado. Cobertura mínima (basicamente steady-state incompressível), código defasado em relação ao OpenFOAM atual, sem manutenção.

**Recomendação:** ignorar para MI300.

---

## 3. Plano de PoC recomendado

### Fase 1 — Validar toolchain (1–2 dias)

Objetivo: confirmar que ROCm reconhece a MI300 e que o stack base funciona.

```bash
# Na VM com MI300
rocminfo | grep -i mi300        # deve listar gfx942
rocm-smi                         # status da GPU
hipcc --version                  # ROCm 6.x+
```

Pré-requisitos:
- ROCm 6.x ou superior (MI300 = gfx942 precisa ROCm recente)
- Driver AMDGPU compatível
- ~50 GB de disco para build

### Fase 2 — Build OpenFOAM + PETSc-ROCm + petsc4Foam (1 dia)

Caminho recomendado: **container ROCm oficial** para isolar dependências.

```
rocm/dev-ubuntu-22.04:6.x
  └── OpenFOAM v2312 ou v2406 (ESI)
  └── PETSc compilado com --with-hip=1 --with-hipc=hipcc
  └── petsc4Foam compilado contra esse PETSc
```

### Fase 3 — Rodar case de referência (1 dia)

**Case sugerido:** `motorBike` do tutorial, refinado para malha de ~10–50M células (o default de ~300k é pequeno demais pra GPU brilhar).

Rodar três vezes:
1. **Baseline CPU**: solver nativo do OpenFOAM, N cores
2. **PETSc CPU**: mesma config, mas solver via PETSc em CPU (mede overhead do PETSc)
3. **PETSc GPU**: mesmo PETSc, agora com `-mat_type aijhipsparse -vec_type hip`

Métricas:
- Tempo total de simulação
- Tempo por iteração do solver de pressão (o pesado)
- Speedup PETSc-GPU vs. baseline CPU
- Ocupação da GPU (`rocm-smi` durante o run)

### Fase 4 (opcional) — neoFOAM smoke test

Compilar neoFOAM com backend HIP, rodar os exemplos demo, ver que GPU é de fato 100% usada — entender o roadmap, não comparar performance.

---

## 4. Decisões abertas

- [ ] Confirmar versão de ROCm instalada na VM (`rocminfo`)
- [ ] Container vs. build nativo (recomendo container)
- [ ] Versão de OpenFOAM (ESI v2406 é a mais alinhada com petsc4Foam atual)
- [ ] Tamanho do case (malha precisa ser grande o suficiente pra GPU compensar overhead)
- [ ] Critério de sucesso da PoC: qual speedup mínimo justifica seguir?

---

## 5. Referências para consultar durante a execução

- **petsc4Foam:** repositório oficial no GitLab da ESI (`develop.openfoam.com/modules/external-solver`)
- **PETSc on AMD GPUs:** documentação PETSc seção "GPU support" / hipSPARSE backend
- **ROCm + MI300:** docs.amd.com/rocm — guia de instalação para Instinct MI300
- **AMD OpenFOAM benchmarks:** white papers da AMD para a linha Instinct (procurar "OpenFOAM HPC MI200/MI300")
- **neoFOAM:** `github.com/exasim-project/NeoFOAM`

---

## 6. Resumo da recomendação

> **Fazer Fase 1 → 3 com petsc4Foam + PETSc-ROCm como PoC principal.** Gera número de speedup real, valida toolchain MI300, e é comparável com benchmarks publicados pela AMD. Se houver interesse em explorar GPU-native depois, neoFOAM como segundo experimento de pesquisa — mas sem expectativa de cobrir workload real ainda.
