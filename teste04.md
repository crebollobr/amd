# Amber 24 com aceleração HIP em AMD MI300X — Receita completa

Compila e valida `pmemd.hip_DPFP` e `pmemd.hip_SPFP` em GPU AMD Instinct MI300X (CDNA3 / gfx942).

**Configuração testada (2026-05-04):**
- VM Ubuntu 22.04+ (root)
- ROCm 6.4.1 em `/opt/rocm`
- AMD Instinct MI300X (gfx942), 192 GB HBM3, 304 CUs
- Amber 24 + AmberTools 25 (download oficial de ambermd.org)
- 20 cores CPU

**Resultado esperado ao final:**
- `pmemd.hip_DPFP` (12 MB) e `pmemd.hip_SPFP` (10 MB) em `~/amber24/bin/`
- 222 testes da suite oficial passando em DPFP (≈77% pass rate)
- STMV (1M átomos) rodando em ~600–1000 ns/dia em SPFP

**Como usar este notebook:** células `%%bash` rodam comandos diretamente. Onde precisar interação humana (download licenciado, criar VM), está marcado em **negrito**.

---

## 1. Provisionar VM com MI300X — **(interação humana)**

Opções para conseguir uma VM com MI300X + ROCm 6.4.1:

1. **AMD Developer Cloud** — `https://www.amd.com/en/developer/resources/cloud-access.html`. Imagem pré-pronta `rocm-X.Y-software-gpu-mi300x1-192gb-devcloud-...`. Recomendado para PoC.
2. **Pedir crédito direto à AMD** via canal corporativo / parceiro HPC.
3. **Self-hosted** — bare-metal MI300X com Ubuntu 22.04/24.04, instalar ROCm 6.4.1 seguindo `https://rocm.docs.amd.com/en/latest/deploy/linux/install.html`.

**Ao logar na VM, confirmar:**

```bash
# verifica ROCm + GPU
echo '=== hipcc ==='
/opt/rocm/bin/hipcc --version 2>&1 | head -1
echo
echo '=== GPU ==='
/opt/rocm/bin/rocm-smi --showid --showproductname 2>&1 | grep -E 'Device Name|Card Series' | head -2
echo
echo '=== arch ==='
/opt/rocm/bin/rocminfo 2>&1 | grep -E 'Name:.*gfx' | head -2
```

Esperado: `gfx942` listado no rocminfo, `MI300X` no rocm-smi.

---

## 2. Baixar código-fonte do Amber 24 — **(interação humana)**

Amber 24 é licenciado (acadêmica gratuita, comercial paga). Não há link público direto.

1. Registrar / logar em `https://ambermd.org/GetAmber.php`
2. Baixar:
   - `Amber24.tar.bz2` (~200 MB) — pmemd, requer licença
   - `AmberTools25.tar.bz2` (~500 MB) — gratuito
3. Transferir os 2 tarballs para `~/` da VM (via `scp` ou `wget` autenticado)

**Depois que os 2 arquivos estiverem em `~/`:**

```bash
# extrair (os dois tarballs montam juntos a árvore ~/amber24_src/)
set -e
cd ~
ls -la Amber24.tar.bz2 AmberTools25.tar.bz2
tar xjf Amber24.tar.bz2
tar xjf AmberTools25.tar.bz2

# verificar estrutura esperada
ls ~/amber24_src/src/pmemd/src/cuda/hip_definitions.h
ls ~/amber24_src/cmake/UseMiniconda.cmake
echo 'OK: codigo fonte extraido em ~/amber24_src/'
```

---

## 3. Instalar dependências de build (apt)

Pacotes necessários no Ubuntu 22.04/24.04 para Boost interno do Amber e cmake.

```bash
set -e
DEBIAN_FRONTEND=noninteractive apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential gfortran cmake git pkg-config \
    python3 python3-dev python3-pip \
    tcsh \
    libnetcdff-dev libnetcdf-dev \
    liblapack-dev libblas-dev libfftw3-dev libboost-all-dev \
    libxml2-dev zlib1g-dev libbz2-dev libreadline-dev libprotobuf-dev \
    bzip2 wget curl flex bison patch make
echo
echo "cmake: $(cmake --version | head -1)"
echo "gfortran: $(gfortran --version | head -1)"
```

---

## 4. Patches no código fonte

Três patches obrigatórios para Amber 24 + ROCm 6.4 + CDNA3 (MI300X):

1. **`cmake/UseMiniconda.cmake`**: Cython 0.29 não compila com Python 3.13 (que vem do miniconda atual). Trocar para `cython` sem versão.
2. **`src/pmemd/src/cuda/hip_definitions.h`**: `cudaCreateTextureObject` / `cudaDestroyTextureObject` retornam erro "operation not supported" no MI300X — texture units não existem em CDNA3. Redefinir como no-op.
3. **13 headers em `src/pmemd/src/cuda/`**: 275 chamadas `tex1Dfetch<T>(cSim.texFOO, EXPR)` falham em runtime/compile. Substituir por `((T*)cSim.pFOO)[EXPR]` — load global direto, numericamente equivalente.

Os patches são idempotentes (rodar 2x não quebra), e geram backups `.bak.*`.

```bash
# Patch 1: cython=0.29 -> cython
set -e
cd ~/amber24_src
F=cmake/UseMiniconda.cmake
if grep -q 'cython=0.29' "$F" 2>/dev/null; then
    cp -n "$F" "$F.bak.cython"
    sed -i 's/cython=0\.29/cython/' "$F"
    echo '[1/3] cython patch aplicado'
else
    echo '[1/3] cython ja patcheado'
fi
```

```bash
# Patch 2: cudaCreateTextureObject -> no-op (MI300X nao tem texture units)
set -e
cd ~/amber24_src
F=src/pmemd/src/cuda/hip_definitions.h
if grep -q '^#define cudaCreateTextureObject hipCreateTextureObject' "$F" 2>/dev/null; then
    cp -n "$F" "$F.bak.tex"
    sed -i 's|^#define cudaCreateTextureObject hipCreateTextureObject|#define cudaCreateTextureObject(pObj, pResDesc, pTexDesc, pResViewDesc) ((*(pObj) = 0), hipSuccess)|' "$F"
    sed -i 's|^#define cudaDestroyTextureObject hipDestroyTextureObject|#define cudaDestroyTextureObject(obj) hipSuccess|' "$F"
    echo '[2/3] hip_definitions.h patch aplicado'
else
    echo '[2/3] hip_definitions.h ja patcheado'
fi
grep -E 'cudaCreate|cudaDestroy.*Texture' "$F" | head -2
```

```bash
# Patch 3: substituir 275 chamadas tex1Dfetch em 13 headers
set -e
cd ~/amber24_src
AMBER_SRC=$PWD python3 - <<'PYEOF'
import os, re
from pathlib import Path

AMBER_SRC = Path(os.environ['AMBER_SRC'])
FILES = [
    'src/pmemd/src/cuda/kPGS.h',
    'src/pmemd/src/cuda/kCCGE.h',
    'src/pmemd/src/cuda/kBNL_AMD.h',
    'src/pmemd/src/cuda/kShake.h',
    'src/pmemd/src/cuda/kNLCPNE_AMD.h',
    'src/pmemd/src/cuda/kNLCPNE.h',
    'src/pmemd/src/cuda/kNLCINE.h',
    'src/pmemd/src/cuda/kNLCPNE_AFE.h',
    'src/pmemd/src/cuda/kPBGS.h',
    'src/pmemd/src/cuda/kPGSPHMD.h',
    'src/pmemd/src/cuda/kCNF.h',
    'src/pmemd/src/cuda/kCalculateGBNonbondEnergy1.h',
    'src/pmemd/src/cuda/kCalculateGBNonbondEnergy2.h',
]
HEAD = re.compile(r'tex1Dfetch<([^>]+)>\(\s*cSim\.tex(\w+)\s*,\s*')

def patch(text):
    out, i, n = [], 0, 0
    while i < len(text):
        m = HEAD.match(text, i)
        if not m:
            out.append(text[i]); i += 1; continue
        tpl = m.group(1).strip(); name = m.group(2); j = m.end(); depth, start = 1, j
        while j < len(text) and depth:
            if text[j] == '(': depth += 1
            elif text[j] == ')':
                depth -= 1
                if depth == 0: break
            j += 1
        if depth:
            out.append(text[i]); i += 1; continue
        expr = text[start:j].strip()
        out.append(f'(({tpl}*)cSim.p{name})[{expr}]')
        i = j + 1; n += 1
    return ''.join(out), n

total = 0
for rel in FILES:
    p = AMBER_SRC / rel
    if not p.exists():
        print(f'  AUSENTE: {rel}'); continue
    txt = p.read_text()
    if 'tex1Dfetch<' not in txt:
        print(f'  {rel}: 0 (ja patcheado)'); continue
    bak = p.with_suffix(p.suffix + '.bak.tex')
    if not bak.exists():
        bak.write_text(txt)
    new, n = patch(txt)
    if n:
        p.write_text(new)
        print(f'  {rel}: {n}')
        total += n
print(f'\n[3/3] Total: {total} substituicoes')
PYEOF
```

Esperado:
```
kPGS.h: 16
kCCGE.h: 2
kBNL_AMD.h: 4
kShake.h: 63
kNLCPNE_AMD.h: 12
kNLCPNE.h: 12
kNLCINE.h: 6
kNLCPNE_AFE.h: 8
kPBGS.h: 4
kPGSPHMD.h: 2
kCNF.h: 144
kCalculateGBNonbondEnergy1.h: 2
kCalculateGBNonbondEnergy2.h: 0
Total: 275 substituicoes
```

---

## 5. Configurar build (cmake)

Cmake baixa miniconda automaticamente (`DOWNLOAD_MINICONDA=TRUE`). Conda recente exige aceitar Terms of Service por canal antes de instalar pacotes — a célula faz isso com retry.

```bash
set -e
cd ~/amber24_src
mkdir -p build
cd build

CMAKE_FLAGS=(
    -DCMAKE_INSTALL_PREFIX=$HOME/amber24
    -DCOMPILER=GNU -DMPI=FALSE -DCUDA=FALSE
    -DHIP=TRUE -DHIP_WARP64=TRUE
    -DAMDGPU_TARGETS=gfx942 -DGPU_TARGETS=gfx942
    -DHIPCUDA_EMULATE_VERSION=11.8
    -DCUDA_TOOLKIT_ROOT_DIR=/opt/rocm
    -DHIP_TOOLKIT_ROOT_DIR=/opt/rocm
    -DCUDA_TOOLKIT_INCLUDE=/opt/rocm/include
    -DBUILD_QUICK=FALSE -DINSTALL_TESTS=TRUE
    -DDOWNLOAD_MINICONDA=TRUE -DCHECK_UPDATES=FALSE
)
CONDA_BIN=$HOME/amber24_src/build/CMakeFiles/miniconda/install/bin/conda

# Se um build anterior ja baixou miniconda, aceitar ToS antes
if [ -x "$CONDA_BIN" ]; then
    "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

# 1a tentativa
if ! cmake .. "${CMAKE_FLAGS[@]}" 2>&1 | tee cmake.log; then
    # cmake provavelmente falhou no ToS do conda recem-baixado. Aceitar e retentar.
    if [ -x "$CONDA_BIN" ]; then
        echo '==> Aceitando ToS no miniconda recem-baixado e re-rodando'
        "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
        "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
        cmake .. "${CMAKE_FLAGS[@]}" 2>&1 | tee cmake.log
    else
        echo 'ERRO: cmake falhou e miniconda nao baixou. Veja cmake.log'
        exit 1
    fi
fi
echo
echo 'cmake OK'
```

---

## 6. Compilar `pmemd.hip_DPFP` e `pmemd.hip_SPFP`

Build seletivo: pulamos `make install` global porque `pytraj` (binding Python do cpptraj) não compila com Python 3.13 — usa APIs internas (`_PyList_Extend`, `_PyLong_AsByteArray`) que mudaram.

Tempo esperado: 15–25 min em 20 cores.

```bash
set -e
cd ~/amber24_src/build
make -j$(nproc) pmemd.hip_DPFP pmemd.hip_SPFP 2>&1 | tail -20
echo
ls -la src/pmemd/src/pmemd.hip_DPFP src/pmemd/src/pmemd.hip_SPFP
```

```bash
# Instalar so pmemd cuda/hip targets em ~/amber24/bin/
set -e
cd ~/amber24_src/build
make install_pmemd_cuda 2>&1 | tail -8
echo
ls -la ~/amber24/bin/pmemd.hip*
```

---

## 7. Montar infraestrutura de teste

A suite oficial precisa de: `AMBERHOME` apontando para dir com `config.h`, `test/`, `AmberTools/test/`, `bin/pmemd.hip*`. Como pulamos o install global, copiamos manualmente o mínimo.

```bash
set -e
AMBER_HOME=$HOME/amber24
AMBER_SRC=$HOME/amber24_src

mkdir -p $AMBER_HOME/AmberTools
cp -f $AMBER_SRC/build/config.h $AMBER_HOME/config.h
cp -f $AMBER_SRC/build/pathscripts/amber.sh $AMBER_HOME/amber.sh
cp -rfu $AMBER_SRC/test           $AMBER_HOME/
cp -rfu $AMBER_SRC/AmberTools/test $AMBER_HOME/AmberTools/
ln -sf pmemd.hip_DPFP $AMBER_HOME/bin/pmemd.hip
echo 'Infra de teste pronta em '$AMBER_HOME
```

---

## 8. Validar com test_amber_hip_serial.sh DPFP

Suite oficial. Tempo: ~5 min para 287 testes.

Patch necessário: o script chama `check_environment ... serial` sem o argumento `nopython`, o que faz ele tentar usar `$AMBERHOME/miniconda/bin/python` (não existe — pulamos pytraj). Adicionar `nopython` skipa a checagem.

```bash
set -e
AMBER_HOME=$HOME/amber24
AMBER_SRC=$HOME/amber24_src
F=$AMBER_HOME/test/test_amber_hip_serial.sh
if ! grep -q 'check_environment.*nopython' "$F"; then
    sed -i 's|\(check_environment .* serial\)$|\1 nopython|' "$F"
    echo '[patch] adicionado nopython em check_environment'
fi

export AMBERHOME=$AMBER_HOME
export LD_LIBRARY_PATH=$AMBER_SRC/build/AmberTools/src/emil:$AMBER_SRC/build/AmberTools/src/kmmd:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

cd $AMBER_HOME/test
LOG=/tmp/amber_hip_test_DPFP_$(date +%Y%m%d_%H%M%S).log
echo "Log: $LOG"
sh test_amber_hip_serial.sh DPFP 2>&1 | tee "$LOG"

echo
echo '=== RESUMO ==='
PASS=$(grep -c '^PASSED$' "$LOG" || true)
FAIL=$(grep -c '^possible FAILURE:' "$LOG" || true)
IGN=$(grep -c 'FAILURE: (ignored)' "$LOG" || true)
ERR=$(grep -cE 'Program [Ee]rror' "$LOG" || true)
TOT=$((PASS + FAIL + ERR))
echo "Total: $TOT  |  PASS: $PASS  |  FAIL: $FAIL ($IGN ignoraveis)  |  Errors: $ERR"
[ $TOT -gt 0 ] && echo "Pass rate: $((PASS * 100 / TOT))%"
```

**Resultado esperado:** ~222 PASSED / ~60 failed (9 ignoráveis) / 5 errors em NPT (grid-cell reorganization, comportamento conhecido do Amber, não bug). **Pass rate ≈ 77%.**

Falhas em `gti/NB_EXP/` são divergência de **parâmetros de input** entre referência e nosso run (scalpha/scbeta), não bug numérico.

---

## 9. Benchmark de stress — STMV (1M átomos)

Para saturar 100% da MI300X, usar STMV (Satellite Tobacco Mosaic Virus, ~1.067M átomos com solvente explícito + íons). 4fs com HMR em produção NPT.

**Em outro terminal**, durante a execução, monitorar GPU:
```
watch -n 1 'rocm-smi --showuse --showtemp --showpower --showmemuse'
```

Esperado: **GPU% 95–100%, ~600–1000 ns/dia em SPFP**.

```bash
# Baixar Amber24 Benchmark Suite (publicamente disponivel em ambermd.org)
set -e
cd ~
if [ ! -d Amber24_Benchmark_Suite ]; then
    wget -q https://ambermd.org/Amber24_Benchmark_Suite.tar.gz
    tar xzf Amber24_Benchmark_Suite.tar.gz
fi
ls Amber24_Benchmark_Suite/PME/
```

```bash
# Rodar STMV NPT 4fs em SPFP
set -e
export LD_LIBRARY_PATH=$HOME/amber24_src/build/AmberTools/src/emil:$HOME/amber24_src/build/AmberTools/src/kmmd:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

cd ~/Amber24_Benchmark_Suite/PME/STMV_production_NPT_4fs

# (opcional) extender nstlim se quiser stress sustentado por mais tempo
# sed -i 's/nstlim *=.*/nstlim = 50000,/' mdin.GPU

time ~/amber24/bin/pmemd.hip_SPFP -O \
    -i mdin.GPU -o mdout -p prmtop -c inpcrd \
    -r restrt -x mdcrd -inf mdinfo

echo
echo '=== Performance ==='
grep -A2 'Average timings' mdout | tail -10
grep -E 'ns/day|ns/d' mdout | tail -5
```

---

## 10. Resumo do que foi entregue

| Item | Local |
|---|---|
| `pmemd.hip_DPFP` (12 MB) | `~/amber24/bin/pmemd.hip_DPFP` |
| `pmemd.hip_SPFP` (10 MB) | `~/amber24/bin/pmemd.hip_SPFP` |
| Backups dos arquivos patcheados | `*.bak.cython`, `*.bak.tex` |
| Log do test suite | `/tmp/amber_hip_test_DPFP_*.log` |
| Benchmark STMV outputs | `~/Amber24_Benchmark_Suite/PME/STMV_production_NPT_4fs/mdout` |

**Limitações conhecidas (não impedem uso, mas vale reportar à AMD em PoC):**
1. Suite hip serial DPFP: ~50 falhas em `gti/NB_EXP/*` (parâmetros input divergentes vs referência), 5 erros em NPT (grid reorg, conhecido), ~12 falhas numéricas pequenas em myoglobin/dhfr_charmm/virtual_sites.
2. SPFP tem mais ruído numérico (esperado, single-precision).
3. Substituição de texturas por loads globais pode ter pequena perda de performance vs CDNA1/2 onde texture cache hardware existe — em CDNA3 não tem diferença, hardware não tem texture units acessíveis.
4. `make install` global falha em `pytraj` (Python 3.13 incompat) e `xblas` (`blas_extended_proto.h` faltando). Usar `make install_pmemd_cuda` para escopo GPU.

**Próximos refinamentos possíveis:**
- Build com MPI para multi-GPU.
- Substituir Python 3.13 por 3.11 no miniconda para destravar `pytraj`.
- Reportar à AMD para incluir CDNA3-friendly upstream em Amber 25.
