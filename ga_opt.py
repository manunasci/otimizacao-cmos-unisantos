# ga_opt.py
# GA robusto para otimizar W, L, Vbias de um NMOS com LTspice + modelo level-3 (~1um)
# Requisitos: Python 3.9+, deap, numpy
# Vars de ambiente (opcional):
#  - LTSPICE_EXE: caminho do executável (ex: C:\Program Files\LTspice\XVIIx64.exe)
#  - CMOSEDU_MODELS: caminho do cmosedu_models.txt (ex: C:\...\cmosedu_models.txt)

import os, sys, math, time, tempfile, shutil, subprocess, re, json, random
from pathlib import Path
import numpy as np

from deap import base, creator, tools

# =========================
# Configs de ambiente
# =========================
LTSPICE_EXE = os.environ.get("LTSPICE_EXE", r"C:\Program Files\ADI\LTspice\LTspice.exe")
CMOSEDU_MODELS = os.environ.get("CMOSEDU_MODELS")
if not CMOSEDU_MODELS:
    # fallback: tenta arquivo em pasta do projeto
    local_models = Path(__file__).parent / "cmosedu_models.txt"
    CMOSEDU_MODELS = str(local_models) if local_models.exists() else r"C:\Users\holam\OneDrive - Sociedade Visconde de São Leopoldo\Área de Trabalho\otimizacao-cmos-unisantos"

IS_WINDOWS = os.name == "nt"
LTSPICE_ARGS = ["-run", "-b", "-ascii"]  # modo batch + ASCII .raw para facilitar
TIMEOUT_S = 25  # tempo máximo por simulação

# =========================
# Espaço de busca (safe)
# =========================
W_MIN, W_MAX = 5e-6, 5e-5
L_MIN, L_MAX = 1.0e-6, 2.0e-6   # <- crítico pro teu modelo level-3 (evita L efetivo < 0)
V_MIN, V_MAX = 0.85, 1.25

# Objetivos: max gain_dB, max UGBW (Hz), min Power (W)
creator.create("FitnessMulti", base.Fitness, weights=(+1.0, +1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_W", random.uniform, W_MIN, W_MAX)
toolbox.register("attr_L", random.uniform, L_MIN, L_MAX)
toolbox.register("attr_V", random.uniform, V_MIN, V_MAX)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_W, toolbox.attr_L, toolbox.attr_V), 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =========================
# Netlist template (estável)
# =========================
NETLIST_TMPL = r"""
* GA CMOS Amp | W={W} L={L} Vbias={V}
.param W={W} L={L} Vbias={V}

.options numdgt=6
.options reltol=1e-3 abstol=1e-9 chgtol=1e-14 vabstol=1e-6 iabstol=1e-12
.options gmin=1e-12 itl1=500 itl4=200 method=trap
*.option noopiter

.include "{CMOSEDU_MODELS}"

* NMOS simples com carga resistiva
* Evita nós flutuantes para .op: resistores de 1G a terra
Rg_leak g 0 1G
Rout_leak out 0 1G

* Dispositivo
M1 out g 0 0 n_1u W={W} L={L}

* Carga e fontes
Rload out vdd 10k
Vg    vg  0  {V}
Rg    vg  g  1k
Vdd   vdd 0  1.8

* Pontos iniciais gentis (ajudam DC)
.ic V(out)=0 V(g)=0 V(vg)={V} V(vdd)=1.8

* Primeiro garante .op; só depois roda .ac
.op
.meas op Idd FIND I(Vdd)
*.meas op Id1 FIND I(M1) ; (LTspice não dá I(M1) direto; usar Idd e gm do log)

* AC e métricas
.ac dec 100 1 1e9
* ganho em 1 kHz e 1 MHz para evitar "MAX" estranho
.meas ac GDB_1k param db(v(out)/v(g)) at=1k
.meas ac GDB_1M param db(v(out)/v(g)) at=1Meg

* Ganho "melhor caso" dentro da varredura (se existir)
.meas ac GAIN_DB  MAX  db(v(out)/v(g))
.meas ac GAIN_LIN MAX  mag(v(out)/v(g))

* UGBW (= |H| cruza 1), pode falhar -> tratamos como 0 no parser
.meas ac FC FIND freq WHEN mag(v(out)/v(g))=1 CROSS=1

.end
""".lstrip("\n")

# =========================
# Helpers de simulação
# =========================
MEAS_RE = {
    "GAIN_DB": re.compile(r'gain_db:\s*MAX\(db\(v\(out\)/v\(g\)\)\)=\(([-+0-9.eE]+)dB', re.I),
    "GAIN_LIN": re.compile(r'gain_lin:\s*MAX\(mag\(v\(out\)/v\(g\)\)\)=\(([-+0-9.eE]+)dB', re.I),  # LTspice formata dB mesmo no MAX
    "GDB_1k": re.compile(r'GDB_1k\s*=\s*([-+0-9.eE]+)', re.I),
    "GDB_1M": re.compile(r'GDB_1M\s*=\s*([-+0-9.eE]+)', re.I),
    "FC": re.compile(r'fc\s*=\s*([-+0-9.eE]+)', re.I),
    "IDD": re.compile(r'Idd\s*=\s*([-+0-9.eE]+)', re.I),
}

FATAL_MARKERS = (
    "Fatal Error", "Failed to find DC operating point",
    "Gmin stepping failed", "Could not converge to DC",
    "effective channel length less than zero"
)

def run_ltspice(netlist_text: str, tag: str):
    work = Path(tempfile.mkdtemp(prefix="ga_amp_"))
    cir = work / f"{tag}.asc"  # extensão pode ser .cir também; LTspice aceita ambos
    log = work / f"{tag}.log"
    raw = work / f"{tag}.raw"

    cir.write_text(netlist_text, encoding="utf-8")

    cmd = [LTSPICE_EXE, *LTSPICE_ARGS, str(cir)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return work, cir, log, raw, "TIMEOUT", "", ""

    stdout, stderr = proc.stdout.decode("utf-8", errors="ignore"), proc.stderr.decode("utf-8", errors="ignore")
    status = "OK" if log.exists() else f"RC={proc.returncode}"
    return work, cir, log, raw, status, stdout, stderr

def parse_log(log_path: Path):
    txt = log_path.read_text(errors="ignore")
    # checa fatais/convergência
    for m in FATAL_MARKERS:
        if m.lower() in txt.lower():
            return dict(ok=False, reason=m)

    vals = {}
    # tentativas de leitura
    m = MEAS_RE["GAIN_DB"].search(txt)
    if m:
        vals["gain_db"] = float(m.group(1))
    else:
        # fallback: usa 1kHz/1MHz
        g1k = MEAS_RE["GDB_1k"].search(txt)
        g1m = MEAS_RE["GDB_1M"].search(txt)
        if g1k:
            vals["gain_db"] = float(g1k.group(1))
        if g1m:
            vals["gain_db"] = max(vals.get("gain_db", -360.0), float(g1m.group(1)))
    fc = MEAS_RE["FC"].search(txt)
    if fc:
        try:
            vals["fc"] = float(fc.group(1))
        except:
            pass
    idd = MEAS_RE["IDD"].search(txt)
    if idd:
        vals["idd"] = float(idd.group(1))

    return dict(ok=True, **vals)

def fitness_eval(ind):
    W, L, V = ind
    # filtros rápidos (antes de simular) — mantêm o GA no espaço físico
    if not (W_MIN <= W <= W_MAX and L_MIN <= L <= L_MAX and V_MIN <= V <= V_MAX):
        return (-360.0, 0.0, 1e3)  # ruim

    # penaliza extremos onde Vgs << Vth (~0.8) ou muito alto (ex: >1.3), mas sem travar
    soft_penalty = 0.0
    if V < 0.85: soft_penalty += (0.85 - V) * 200.0
    if V > 1.25: soft_penalty += (V - 1.25) * 200.0

    tag = f"W{W:.3e}_L{L:.3e}_V{V:.3f}".replace("+", "").replace("-", "m")
    net = NETLIST_TMPL.format(W=W, L=L, V=V, CMOSEDU_MODELS=CMOSEDU_MODELS)

    work, cir, log, raw, status, so, se = run_ltspice(net, tag)

    try:
        if status != "OK" or not log.exists():
            # sem .log — falha dura
            return (-360.0, 0.0, 1e3)

        parsed = parse_log(log)
        if not parsed.get("ok", False):
            # falha de convergência — fitness ruim
            return (-360.0, 0.0, 1e3)

        gain_db = float(parsed.get("gain_db", -360.0))
        ugbw = float(parsed.get("fc", 0.0))
        idd = abs(float(parsed.get("idd", 0.0)))
        power = 1.8 * idd  # W

        # aplica penalidade suave ao ganho
        gain_db_adj = gain_db - soft_penalty

        # protege contra NaN
        if not math.isfinite(gain_db_adj): gain_db_adj = -360.0
        if not math.isfinite(ugbw): ugbw = 0.0
        if not math.isfinite(power): power = 1e3

        return (gain_db_adj, ugbw, power)

    finally:
        # limpeza (comenta se quiser manter)
        try:
            shutil.rmtree(work, ignore_errors=True)
        except: pass

toolbox.register("evaluate", fitness_eval)
toolbox.register("mate", tools.cxBlend, alpha=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.5)
toolbox.register("select", tools.selNSGA2)

def clip_params(ind):
    ind[0] = float(np.clip(ind[0], W_MIN, W_MAX))
    ind[1] = float(np.clip(ind[1], L_MIN, L_MAX))
    ind[2] = float(np.clip(ind[2], V_MIN, V_MAX))

def main(seed=42, pop_size=20, ngen=12, cxpb=0.6, mutpb=0.4):
    random.seed(seed)
    pop = toolbox.population(n=pop_size)

    # “semeia” alguns pontos centróides válidos
    anchors = [
        [2.0e-5, 1.2e-6, 1.00],
        [1.0e-5, 1.5e-6, 0.95],
        [3.0e-5, 1.6e-6, 1.10],
        [5.0e-6, 1.2e-6, 1.05],
    ]
    for a in anchors:
        pop.append(creator.Individual(a))

    # corrige limites
    for ind in pop: clip_params(ind)

    # Avalia inicial
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("min", np.min, axis=0)

    hof = tools.ParetoFront()

    print(f"{'gen':<7}{'nevals':<8}{'avg':<30}{'max':<30}{'min':<30}")
    for gen in range(ngen + 1):
        if gen > 0:
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # cruzamento & mutação
            for i in range(0, len(offspring), 2):
                if random.random() < cxpb and i+1 < len(offspring):
                    toolbox.mate(offspring[i], offspring[i+1])
                    del offspring[i].fitness.values, offspring[i+1].fitness.values
            for i, ind in enumerate(offspring):
                if random.random() < mutpb:
                    toolbox.mutate(ind)
                    clip_params(ind)
                    del ind.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, pop_size)

        record = stats.compile(pop)
        print(f"{gen:<7}{len(pop):<8}{np.array2string(record['avg'], precision=3, suppress_small=True):<30}"
              f"{np.array2string(record['max'], precision=3, suppress_small=True):<30}"
              f"{np.array2string(record['min'], precision=3, suppress_small=True):<30}")
        hof.update(pop)

    # escolhe melhor por função escalar (pode ser ajustado)
    def scalarize(f):  # ganho (dB) pesa mais, depois ugbw, penaliza potência
        return f[0] + 1e-6 * f[1] - 1e3 * f[2]
    best = max(pop, key=lambda ind: scalarize(ind.fitness.values))

    W, L, V = best
    gain_db, ugbw, power = best.fitness.values
    print("\n===== RESULTADO =====")
    print(f"Melhor individuo [W, L, Vbias] = [{W}, {L}, {V}]")
    print(f"Metricas (Gain_dB, UGBW_Hz, Power_W): ({gain_db}, {ugbw}, {power})")

if __name__ == "__main__":
    # permite override simples via args
    kwargs = {}
    if len(sys.argv) > 1:
        try:
            cfg = json.loads(sys.argv[1])
            kwargs.update(cfg)
        except Exception:
            pass
    main(**kwargs)
