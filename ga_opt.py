# ga_opt.py
# GA robusto adaptado para otimizar Wn, Wp, Vbias de um estágio Classe AB
# Requisitos: Python 3.9+, deap, numpy
# Vars de ambiente (opcional):
#  - LTSPICE_EXE: caminho do executável (ex: C:\Program Files\LTspice\XVIIx64.exe)
#  - LIB_PATH: caminho do C5_models_SPICE.txt

import os, sys, math, time, tempfile, shutil, subprocess, re, json, random
from pathlib import Path
import numpy as np

from deap import base, creator, tools

# =========================
# Configs de ambiente
# =========================
LTSPICE_EXE = os.environ.get("LTSPICE_EXE", r"C:\Program Files\ADI\LTspice\LTspice.exe")
LIB_PATH = os.environ.get("LIB_PATH", r"C:\Users\holam\OneDrive - Sociedade Visconde de São Leopoldo\Área de Trabalho\otimizacao-cmos-unisantos\C5_models_SPICE.txt")

if not Path(LTSPICE_EXE).exists():
    print(f"Atenção: LTspice.exe NÃO ENCONTRADO EM: {LTSPICE_EXE}")
    
if not Path(LIB_PATH).exists():
    print(f"Atenção: Biblioteca de modelos NÃO ENCONTRADA EM: {LIB_PATH}")

IS_WINDOWS = os.name == "nt"
LTSPICE_ARGS = ["-run", "-b", "-ascii"]
TIMEOUT_S = 25

# =========================
# Espaço de busca (ADAPTADO para Wn, Wp, Vbias)
# =========================
# WN_MIN alterado para 3e-6 para respeitar o "minimum W is 3 um" do C5_models_SPICE.txt
WN_MIN, WN_MAX = 3e-6, 20e-6   # Range de busca para W de M2, M8 (original era 5u)
WP_MIN, WP_MAX = 5e-6, 50e-6   # Range de busca para W de M3, M6 (original era 10u)
V_MIN, V_MAX   = 1.0, 5.0      # Range de busca para Vbias (baseado no sweep do .asc)

# Objetivos: max gain_dB, max UGBW (Hz), min Power (W)
creator.create("FitnessMulti", base.Fitness, weights=(+1.0, +1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# Genes adaptados: Wn, Wp, Vbias
toolbox.register("attr_Wn", random.uniform, WN_MIN, WN_MAX)
toolbox.register("attr_Wp", random.uniform, WP_MIN, WP_MAX)
toolbox.register("attr_V", random.uniform, V_MIN, V_MAX)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_Wn, toolbox.attr_Wp, toolbox.attr_V), 1) # 3 genes
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =========================
# Netlist template (ADAPTADO para o circuito .asc)
# =========================
NETLIST_TMPL = r"""
* GA CMOS Amp Classe AB | Wn={Wn} Wp={Wp} Vbias={Vbias}
.param Wn={Wn} Wp={Wp} Vbias={Vbias}

.options numdgt=6
.options reltol=1e-3 abstol=1e-9 chgtol=1e-14 vabstol=1e-6 iatol=1e-12
.options gmin=1e-12 itl1=500 itl4=200 method=trap

.include "{LIB_PATH}"

* Componentes (Baseado no .asc)
* L=0.6u fixo
* Modelos alterados para NMOS e PMOS (maiúsculas)
M8   out bias 0 0     NMOS l=0.6u w={Wn}
M6   out in   Vdd Vdd PMOS l=0.6u w={Wp}
M3   in  in   Vdd Vdd PMOS l=0.6u w={Wp}
M2   in  bias 0 0     NMOS l=0.6u w={Wn}

* Fontes (Baseado no .asc)
V4    bias 0      DC {Vbias}
Vdd   Vdd  0      DC 5
V2    in   0      AC 1
Vdd1  out  0      DC 0

* Pontos iniciais (ajudam DC)
.ic V(out)=2.5 V(in)=0 V(bias)={Vbias} V(vdd)=5

* Primeiro garante .op; só depois roda .ac
.op
.meas op Idd FIND I(Vdd)

* AC e métricas (Range do .asc)
.ac dec 100 1e8 1e9

* Ganho "melhor caso" dentro da varredura
.meas ac GAIN_DB  MAX  db(v(out)/v(in))
.meas ac GAIN_LIN MAX  mag(v(out)/v(in))

* UGBW (= |H| cruza 1)
.meas ac FC FIND freq WHEN mag(v(out)/v(in))=1 CROSS=1

.end
""".lstrip("\n")

# =========================
# Helpers de simulação
# =========================
# Regex da versão anterior (que estava falhando)
MEAS_RE = {
    "GAIN_DB": re.compile(r'gain_db:\s*MAX\(db\(v\(out\)/v\(in\)\)\)=\(([-+0-9.eE]+)', re.I),
    "GAIN_LIN": re.compile(r'gain_lin:\s*MAX\(mag\(v\(out\)/v\(in\)\)\)=\(([-+0-9.eE]+)', re.I),
    "FC": re.compile(r'fc\s*=\s*([-+0-9.eE]+)', re.I),
    "IDD": re.compile(r'Idd\s*=\s*([-\d.eE]+)', re.I), 
}

FATAL_MARKERS = (
    "Fatal Error", "Failed to find DC operating point",
    "Gmin stepping failed", "Could not converge to DC",
    "effective channel length less than zero",
    "Unknown model",
    "Can't open"
)

def run_ltspice(netlist_text: str, tag: str):
    work = Path(tempfile.mkdtemp(prefix="ga_amp_"))
    
    # Correção: Alterado de .asc para .cir
    cir = work / f"{tag}.cir"
    log = work / f"{tag}.log"
    raw = work / f"{tag}.raw"

    cir.write_text(netlist_text, encoding="utf-8")

    cmd = [LTSPICE_EXE, *LTSPICE_ARGS, str(cir)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return work, cir, log, raw, "TIMEOUT", "", ""
    except FileNotFoundError:
        return work, cir, log, raw, "NOT_FOUND", "", ""

    stdout, stderr = proc.stdout.decode("utf-8", errors="ignore"), proc.stderr.decode("utf-8", errors="ignore")
    status = "OK" if log.exists() else f"RC={proc.returncode}"
    return work, cir, log, raw, status, stdout, stderr

def parse_log(log_path: Path):
    if not log_path.exists():
        return dict(ok=False, reason="Log file not created")
        
    txt = log_path.read_text(errors="ignore")
    for m in FATAL_MARKERS:
        if m.lower() in txt.lower():
            # Retorna o motivo da falha
            return dict(ok=False, reason=m) 

    vals = {}
    m_gain = MEAS_RE["GAIN_DB"].search(txt)
    if m_gain:
        vals["gain_db"] = float(m_gain.group(1))
    
    m_fc = MEAS_RE["FC"].search(txt)
    if m_fc:
        try:
            vals["fc"] = float(m_fc.group(1))
        except:
            pass
            
    m_idd = MEAS_RE["IDD"].search(txt)
    if m_idd:
        vals["idd"] = float(m_idd.group(1))

    # Checa se medições essenciais falharam (LTspice escreve 'FAILED')
    if "gain_db: FAILED" in txt.lower():
        return dict(ok=False, reason="GAIN_DB measurement FAILED")

    return dict(ok=True, **vals)

def fitness_eval(ind):
    Wn, Wp, V = ind
    if not (WN_MIN <= Wn <= WN_MAX and WP_MIN <= Wp <= WP_MAX and V_MIN <= V <= V_MAX):
        return (-360.0, 0.0, 1e3)

    tag = f"Wn{Wn:.3e}_Wp{Wp:.3e}_V{V:.3f}".replace("+", "").replace("-", "m")
    net = NETLIST_TMPL.format(Wn=Wn, Wp=Wp, Vbias=V, LIB_PATH=LIB_PATH)

    work, cir, log, raw, status, so, se = run_ltspice(net, tag)

    try:
        if status != "OK" or not log.exists():
            # <-- MODO DE DEPURAÇÃO: Imprime falha de execução
            print(f"!!! FALHA DE EXECUÇÃO: Status={status}, Log Existe={log.exists()}, Arquivo={log.parent.name}")
            return (-360.0, 0.0, 1e3)

        parsed = parse_log(log)
        if not parsed.get("ok", False):
            # <-- MODO DE DEPURAÇÃO: Imprime falha de simulação (ex: convergência DC)
            print(f"!!! FALHA DE SIMULAÇÃO (do .log): {parsed.get('reason')} em {log.parent.name}")
            return (-360.0, 0.0, 1e3)

        # Processa os valores
        gain_db = float(parsed.get("gain_db", -360.0))
        ugbw = float(parsed.get("fc", 0.0)) # Default é 0.0, não é uma falha
        idd = abs(float(parsed.get("idd", 1e9))) # Default é alta potência
        
        if idd > 1e3:
             power = 1e3
        else:
             power = 5.0 * idd

        # Protege contra NaN
        if not math.isfinite(gain_db): gain_db = -360.0
        if not math.isfinite(ugbw): ugbw = 0.0
        if not math.isfinite(power): power = 1e3
        
        # Só consideramos falha se o GANHO não for medido.
        if gain_db == -360.0:
            # <-- MODO DE DEPURAÇÃO: Imprime falha de medição
            print(f"!!! FALHA DE MEDIÇÃO: gain_db não foi encontrado no log {log.parent.name}")
            return (-360.0, 0.0, 1e3)

        # Se chegou aqui, a simulação é válida
        return (gain_db, ugbw, power)

    finally:
        try:
            # <-- MODO DE DEPURAÇÃO: Limpeza DESATIVADA
            # shutil.rmtree(work, ignore_errors=True)
            pass
        except: pass

toolbox.register("evaluate", fitness_eval)
toolbox.register("mate", tools.cxBlend, alpha=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.5)
toolbox.register("select", tools.selNSGA2)

def clip_params(ind):
    ind[0] = float(np.clip(ind[0], WN_MIN, WN_MAX)) # Wn
    ind[1] = float(np.clip(ind[1], WP_MIN, WP_MAX)) # Wp
    ind[2] = float(np.clip(ind[2], V_MIN, V_MAX))  # Vbias

def main(seed=42, pop_size=20, ngen=12, cxpb=0.6, mutpb=0.4):
    random.seed(seed)
    pop = toolbox.population(n=pop_size)

    anchors = [
        [5e-6, 10e-6, 1.5], 
        [10e-6, 20e-6, 2.0],
        [3e-6, 15e-6, 2.5], # Wn=3um (agora válido)
        [8e-6, 8e-6, 3.0],
    ]
    for a in anchors:
        pop.append(creator.Individual(a))

    for ind in pop: clip_params(ind)

    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    pop = toolbox.select(pop, pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("min", np.min, axis=0)

    hof = tools.ParetoFront()

    print(f"{'gen':<7}{'nevals':<8}{'avg':<30}{'max':<30}{'min':<30}")
    for gen in range(ngen + 1):
        if gen > 0:
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring.copy()] # .copy() é mais seguro

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
        print(f"{'gen':<7}{len(pop):<8}{np.array2string(record['avg'], precision=3, suppress_small=True):<30}"
              f"{np.array2string(record['max'], precision=3, suppress_small=True):<30}"
              f"{np.array2string(record['min'], precision=3, suppress_small=True):<30}")
        hof.update(pop)

    def scalarize(f):
        return f[0] + 1e-6 * f[1] - 1e3 * f[2]
    best = max(pop, key=lambda ind: scalarize(ind.fitness.values))

    Wn, Wp, V = best
    gain_db, ugbw, power = best.fitness.values
    print("\n===== RESULTADO =====")
    print(f"Melhor individuo [W_n, W_p, Vbias] = [{Wn}, {Wp}, {V}]")
    print(f"Metricas (Gain_dB, UGBW_Hz, Power_W): ({gain_db}, {ugbw}, {power})")

if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) > 1:
        try:
            cfg = json.loads(sys.argv[1])
            kwargs.update(cfg)
        except Exception:
            pass
    main(**kwargs)