import os
import glob
import subprocess
import random
import numpy as np
import csv
from deap import base, creator, tools, algorithms

# -------------------------------------------------------------------
# 1) DEFINIÇÕES DE CAMINHOS
# -------------------------------------------------------------------
LIB_PATH = r"C:\Users\holam\OneDrive - Sociedade Visconde de São Leopoldo\Área de Trabalho\otimizacao-cmos-unisantos\transistor_model.lib"

# Localiza LTspice.exe automaticamente
candidates = glob.glob(r"C:\Program Files\ADI\LTspice\LTspice.exe")
if not candidates:
    raise FileNotFoundError("Não achei LTspice.exe em 'C:\\Program Files\\ADI'.")
LTSPICE_EXE = candidates[0]

WORKDIR = "simulacoes"
os.makedirs(WORKDIR, exist_ok=True)

# -------------------------------------------------------------------
# 2) TEMPLATE DE NETLIST COM .measure
# -------------------------------------------------------------------
NETLIST_TEMPLATE = f"""* Amplificador CMOS – W={{W}} L={{L}} Vbias={{Vbias}}
.param W={{W}} L={{L}} Vbias={{Vbias}}

* Modelos de fallback
.model nmos NMOS
.model pmos PMOS

* Biblioteca de processo
.include "{LIB_PATH}"

* Dispositivos
M1    out in   bias 0   nmos W={{W}} L={{L}}
Rload out vdd       10k
Vbias bias 0        DC {{Vbias}}
Vin   in   0        AC 1
Vdd   vdd  0        DC 1.8

* Análises
.op
.ac dec 100 1 1e9

* Medições
.measure ac GAIN_LIN MAX mag(v(out)/v(in))
.measure ac GAIN_DB  MAX db(v(out)/v/in))
.measure ac FC       WHEN mag(v(out)/v(in))=GAIN_LIN/sqrt(2)
.measure op  Idd     FIND I(Vdd)

.end
"""

# -------------------------------------------------------------------
# 3) FUNÇÃO DE SIMULAÇÃO
# -------------------------------------------------------------------
def run_simulation(W, L, Vb):
    tag      = f"W{W:.0e}_L{L:.0e}_V{int(Vb*1e3)}mV"
    cir_path = os.path.join(WORKDIR, f"amp_{tag}.cir")
    log_path = os.path.join(WORKDIR, f"amp_{tag}.log")

    # Gera netlist
    with open(cir_path, "w") as f:
        f.write(NETLIST_TEMPLATE.format(W=W, L=L, Vbias=Vb))

    # Executa LTspice em modo batch/ASCII
    try:
        subprocess.run([LTSPICE_EXE, "-ascii", "-b", cir_path, "-log", log_path], check=True)
    except subprocess.CalledProcessError:
        # Penalidade para simulações inválidas
        print(f"⚠️ Simulação falhou para {tag}, atribuindo fitness penalizado.")
        return 0.0, 0.0, 1e6

    # Parse das medições no .log
    gain_db = fc = Idd = None
    with open(log_path, "r") as lf:
        for line in lf:
            line = line.strip()
            if line.startswith("GAIN_DB"):
                try:
                    gain_db = float(line.split("=")[1].split()[0])
                except:
                    gain_db = 0.0
            elif line.startswith("FC"):
                try:
                    fc = float(line.split("=")[1].split()[0])
                except:
                    fc = 0.0
            elif line.startswith("Idd"):
                try:
                    Idd = float(line.split("=")[1].split()[0])
                except:
                    Idd = 0.0
    # Garante valores numéricos
    gain_db = gain_db if gain_db is not None else 0.0
    fc      = fc      if fc      is not None else 0.0
    Idd     = Idd     if Idd     is not None else 0.0
    power   = 1.8 * Idd
    return gain_db, fc, power

# -------------------------------------------------------------------
# 4) CONFIGURAÇÃO DO AG COM DEAP
# -------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Genes: W1, L1, W2, L2, Vbias
toolbox.register("attr_W", random.uniform, 1e-6, 50e-6)
toolbox.register("attr_L", random.uniform, 0.1e-6, 1e-6)
toolbox.register("attr_Vb", random.uniform, 0.5, 1.5)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_W, toolbox.attr_L,
                  toolbox.attr_W, toolbox.attr_L,
                  toolbox.attr_Vb), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Avaliação usando função de simulação
def evaluate(ind):
    gain, fc, power = run_simulation(ind[0], ind[1], ind[4])
    return gain, fc, power

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian,
                 mu=0, sigma=1e-6, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("max", np.max, axis=0)
stats.register("min", np.min, axis=0)

# -------------------------------------------------------------------
# 5) LOOP PRINCIPAL DO AG
# -------------------------------------------------------------------
def main():
    pop, hof = toolbox.population(n=20), tools.HallOfFame(1)
    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                       cxpb=0.7, mutpb=0.3,
                                       ngen=10, stats=stats,
                                       halloffame=hof, verbose=True)

    best = hof[0]
    print("Melhor indivíduo:", best)
    print("Métricas (Gain_dB, fc(Hz), Power_W):", best.fitness.values)

if __name__ == "__main__":
    main()
