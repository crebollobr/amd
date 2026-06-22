# amd

# Todo 05/06/2026


  Plano sugerido pra hoje: DeePMD-kit → GPAW → ABINIT (os três novos que batem na placa). Se quiser jogar mais seguro, intercala com CP2K ou QMCPACK do seu
  TODO, que têm container AMD pronto e fecham rápido.


## Ordem sugerida, do mais fácil ao mais trabalhoso:

1. CP2K — suporte HIP/ROCm maduro, container pronto no AMD Infinity Hub, benchmarks publicados pela AMD. Smoke test
rápido com H2O-64 ou LiH-HFX.
2. QMCPACK — port HIP nativo, AMD investiu pesado (é case-study oficial deles para MI series). Build limpo,
exemplos NiO clássicos.
3. NWChem — tem ROCm via offload, mas é o mais chato dos três: TCE/CCSD nem todos os módulos têm kernel HIP.
Funciona, mas exige escolher o módulo certo (DFT e plane-wave são os caminhos mais limpos).


```
  ┌──────────┬─────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
  │ Software │    Sua nota     │                                    Realidade                                    │
  ├──────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ VASP     │ "AMD GPU -      │ ❌ Só GPU via OpenACC = NVIDIA HPC SDK exclusivo. Em AMD: só CPU. Há tentativas │
  │          │ HIP/ROCm"       │  de OpenMP-target offload, ainda não produção.                                  │
  ├──────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ GAMESS   │ (só CUDA)       │ ❌ Confirmado, só NVIDIA.                                                       │
  ├──────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Yambo    │ (só CUDA)       │ ❌ Confirmado, CUDA Fortran via NVHPC. Sem ROCm.                                │
  ├──────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Gaussian │ (só CUDA)       │ ❌ Proprietário, suporte GPU apenas NVIDIA (Tesla/A100/H100 homologados). Não   │
  │          │                 │ roda em AMD GPU por design.                                                     │
  └──────────┴─────────────────┴─────────────────────────────────────────────────────────────────────────────────┘


```


## CP2K 
```

NVidia GPU - CUDA
AMD GPU -  HIP/ROCm

OBS: Testado pela AMD (AMD Infinity

```

## Amber 

```

AMBER -   Dinâmica Molecular  e Análise de Sistemas Biomoleculares ( Proprietário/Licença )

NVidia GPU - CUDA
AMD GPU -  HIP/ROCm

OBS: Testado pela AMD (AMD Infinity Hub)


```

## NWChem

```
NWChem - Dinâmica Molecular e Mecânica Quântica ("Free")

NVidia GPU - CUDA
AMD GPU -  HIP/ROCm

OBS: Testado pela AMD (AMD Infinity Hub)

```
## TensorFlow OK teste06.md

```
TensorFlow - Biblioteca para solução de problemas em "Deep Learning", "Machine Learning" e IA  ("Free")

NVidia GPU - CUDA
AMD GPU -  HIP/ROCm

OBS: Testado pela AMD (AMD Infinity Hub)
```
## PyTorch
```
PyTorch - "Machine Learning Framewor for Deep Learning applications" ("Free")

NVidia GPU - CUDA
AMD GPU -  HIP/ROCm

OBS: Testado pela AMD (AMD Infinity Hub)
```
## OpenFOAM NOK teste07.md
```
OpenFOAM - Dinâmica de Fluidos Computacionais (CFD) ("Free")

           NVidia GPU - CUDA
           AMD GPU -  HIP/ROCm

           OBS: Testado pela AMD (AMD Infinity Hub)
```

## QMCPACK

```
QMCPACK - Cálculo de Estruturas Eletrônica com diversos algritmos de Quantum Monte Carlo (QMC)  ("Free")

           NVidia GPU - CUDA
           AMD GPU -  HIP/ROCm
```

## VASP
```
VASP - Simulação AB-Iniitio de Dinâmica dos Fluidos (Proprietário)

           NVidia GPU - OpenACC
           AMD GPU -  HIP/ROCm
```

## GAMESS

```
GAMESS - Análise de Estrutura Eletrônica Molecular e Atômica ("Free")

           NVidia GPU - CUDA

```

## Espresso
```
Quantum Espresso - Cálculos de Estrutura Eletrônica ("Free")

           NVidia GPU - CUDA
```
## Yambo
```
Yambo -  Física do Estado Sólido e Molecular ("Free")

           NVidia GPU - CUDA
```
## Gaussian
```

Gaussian - Dinâmica Molecular - (Proprietário)

          NVidia GPU - CUDA

```



# teste00.md
## PyTorch OK
## GROMACS OK

# teste02.md
## NAMD OK

# teste03.md
## LAMMPS OK

# teste04.md
## Amber e AmberTools OK 

# teste05.md
## Quantum ESPRESSO NOK

# teste06.md

# teste07.md
# teste06.md
## TensorFlow na AMD Instinct MI300X OK
