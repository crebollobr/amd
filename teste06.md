# Validação do TensorFlow na AMD Instinct MI300X (ROCm)

Registro da validação do TensorFlow rodando na GPU AMD Instinct MI300X via ROCm,
em VM na DigitalOcean. Objetivo: instalar no padrão de mercado e rodar uma carga
que prenda a GPU em ~100% por 5+ minutos.

---

## 1. Ambiente

| Item     | Valor                                            |
|----------|--------------------------------------------------|
| GPU      | AMD Instinct **MI300X VF** · `gfx942` · 192 GB   |
| ROCm     | **7.2.0**                                         |
| SO       | Ubuntu 24.04.4 LTS · kernel 6.8.0-106-generic    |
| Python   | 3.12.3 (host)                                     |
| Docker   | 29.3.0                                            |
| Host     | `rocm-7-2-software-gpu-mi300x1-192gb-devcloud-atl1` |

Comandos de diagnóstico:

```bash
rocm-smi --showproductname
cat /opt/rocm/.info/version
grep PRETTY_NAME /etc/os-release
uname -r
python3 --version
docker --version
```

> Nota: é uma MI300X **VF** (virtual function — GPU virtualizada no hypervisor).
> Quando ociosa, o `rocm-smi` reporta *low-power state*; a placa "acorda" sob carga.

---

## 2. Instalação (padrão de mercado: container oficial AMD)

Usamos a imagem oficial mantida pela AMD, que já traz TensorFlow compilado para
ROCm + toolchain casado. O container usa apenas o driver de kernel do host (KFD),
que já está em ROCm 7.2.0 — casamento perfeito com a tag `rocm7.2`.

**Imagem:** `rocm/tensorflow:rocm7.2-py3.12-tf2.20-dev`
(ROCm 7.2 + Python 3.12 + TensorFlow 2.20)

```bash
docker pull rocm/tensorflow:rocm7.2-py3.12-tf2.20-dev
```

### Acesso à GPU dentro do container

Esta VM **não tem o grupo `render` por nome** (comum em setups VF). Por isso, em
vez de `--group-add render`, passamos os **GIDs numéricos** dos device nodes:

```bash
--device=/dev/kfd --device=/dev/dri \
--group-add $(stat -c '%g' /dev/kfd) \
--group-add $(stat -c '%g' /dev/dri/renderD128) \
--security-opt seccomp=unconfined
```

---

## 3. Validação: detecção da GPU

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add $(stat -c '%g' /dev/kfd) \
  --group-add $(stat -c '%g' /dev/dri/renderD128) \
  --security-opt seccomp=unconfined \
  rocm/tensorflow:rocm7.2-py3.12-tf2.20-dev \
  python3 -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Resultado (OK ✅):**

```
TF 2.20.0-dev0+selfbuilt
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 4. Burn de GPU (~100% por 6 min)

Carga sustentada de GEMM (matmul) em fp32, encadeada e sincronizada a cada passo,
para manter a GPU saturada.

### Script `/root/tf_burn.py`

```python
import tensorflow as tf, time

print("TF:", tf.__version__, flush=True)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus, flush=True)
assert gpus, "Nenhuma GPU detectada — abortando."

N = 8192            # tamanho das matrizes
INNER = 20          # matmuls encadeados por passo
DURATION = 360      # segundos de carga (6 min)

with tf.device('/GPU:0'):
    a = tf.random.normal((N, N), dtype=tf.float32)
    b = tf.random.normal((N, N), dtype=tf.float32)

@tf.function
def step(a, b):
    c = a
    for _ in range(INNER):
        c = tf.matmul(c, b)
    return c

_ = step(a, b).numpy()   # warmup (compila o grafo)

print(f"Iniciando burn: N={N}, {INNER} matmuls/passo, {DURATION}s", flush=True)
t0 = time.time(); iters = 0
while time.time() - t0 < DURATION:
    _ = step(a, b).numpy()
    iters += 1
    if iters % 10 == 0:
        el = time.time() - t0
        flop = iters * INNER * 2.0 * (N**3)
        print(f"  {el:6.1f}s  passos={iters}  ~{flop/el/1e12:6.1f} TFLOP/s", flush=True)

el = time.time() - t0
flop = iters * INNER * 2.0 * (N**3)
print(f"Concluido: {iters} passos em {el:.1f}s  (~{flop/el/1e12:.1f} TFLOP/s fp32)", flush=True)
```

### Monitoramento (segundo terminal, no host)

```bash
watch -n 2 rocm-smi
```

Acompanhar: **GPU%** (→ ~100%), **Temp**, **Power**, **VRAM%**.

### Disparar o burn

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add $(stat -c '%g' /dev/kfd) \
  --group-add $(stat -c '%g' /dev/dri/renderD128) \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -v /root:/work -w /work \
  rocm/tensorflow:rocm7.2-py3.12-tf2.20-dev \
  python3 /work/tf_burn.py
```

### Resultado

> _(preencher após a execução)_

- Linha final do script (TFLOP/s fp32 sustentado): `...`
- Pico de GPU% no `rocm-smi`: `...`
- Temp / Power sob carga: `...`

---

## 5. Conclusão

> _(preencher: TensorFlow 2.20 validado na MI300X via ROCm 7.2 — detecção OK,
> burn sustentado por 6 min sem erros.)_
