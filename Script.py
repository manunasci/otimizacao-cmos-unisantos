import os
import glob
import subprocess
import itertools
import numpy as np
import csv

# -------------------------------------------------------------------
# 1) CAMINHO ABSOLUTO DA SUA BIBLIOTECA .lib
# -------------------------------------------------------------------
LIB_PATH = r"C:\Users\holam\OneDrive - Sociedade Visconde de São Leopoldo\Área de Trabalho\otimizacao-cmos-unisantos\transistor_model.lib"

# -------------------------------------------------------------------
# 2) LOCALIZAÇÃO DO EXECUTÁVEL LTspice
# -------------------------------------------------------------------
candidates = glob.glob(r"C:\Program Files\ADI\LTspice\LTspice.exe")
if not candidates:
    raise FileNotFoundError("Não achei LTspice.exe em 'C:\\Program Files\\ADI'.")
LTSPICE_EXE = candidates[0]

# -------------------------------------------------------------------
# 3) CONFIGURAÇÃO DE PARÂMETROS
# -------------------------------------------------------------------
WORKDIR     = "simulacoes"
os.makedirs(WORKDIR, exist_ok=True)

W_vals     = [10e-6, 20e-6, 30e-6]
L_vals     = [0.18e-6, 0.5e-6]
Vbias_vals = [0.7, 1.0, 1.2]

# -------------------------------------------------------------------
# 4) TEMPLATE DE NETLIST COM .measure (nível principal)
# -------------------------------------------------------------------
NETLIST_TEMPLATE = f"""* Amplificador CMOS – W={{W}} L={{L}} Vbias={{Vbias}}
.param W={{W}} L={{L}} Vbias={{Vbias}}

* Modelos de fallback
.model nmos NMOS
.model pmos PMOS

* Biblioteca de processo
.include "{LIB_PATH}"

* Componentes
M1    out in   bias 0   nmos W={{W}} L={{L}}
Rload out vdd       10k
Vbias bias 0        DC {{Vbias}}
Vin   in   0        AC 1
Vdd   vdd  0        DC 1.8

* Análises
.op
.ac dec 100 1 1e9

* Medições no log
.measure ac GMAX MAX mag(v(out)/v(in))
.measure ac FC   WHEN mag(v(out)/v(in))=GMAX/sqrt(2)

.end
"""

# -------------------------------------------------------------------
# 5) VARREDURA E COLETA
# -------------------------------------------------------------------
results = []
for W, L, Vb in itertools.product(W_vals, L_vals, Vbias_vals):
    tag      = f"W{W:.0e}_L{L:.0e}_V{int(Vb*1e3)}mV"
    cir_path = os.path.join(WORKDIR, f"amp_{tag}.cir")
    log_path = os.path.join(WORKDIR, f"amp_{tag}.log")

    # Gera netlist
    with open(cir_path, "w") as f:
        f.write(NETLIST_TEMPLATE.format(W=W, L=L, Vbias=Vb))

    # Roda LTspice em batch/ASCII
    cmd = [LTSPICE_EXE, "-ascii", "-b", cir_path, "-log", log_path]
    print("⏳ Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Parse das medições no log
    Gmax = np.nan
    fc   = np.nan
    Idd  = np.nan
    with open(log_path, "r") as lf:
        for line in lf:
            if line.startswith("GMAX"):
                # Ex.: GMAX: MAX mag(v(out)/v(in)) = 24.3 dB at 1.00e+06 Hz
                parts = line.split("=")[1].split()
                Gmax  = float(parts[0])
            elif line.startswith("FC"):
                # Ex.: FC: WHEN ... = 3.16e+05
                parts = line.split("=")[1].split()
                fc    = float(parts[0])
            elif "I(Vdd)" in line:
                try:
                    Idd = abs(float(line.split()[1]))
                except:
                    pass

    Power = 1.8 * Idd
    results.append((W, L, Vb, Gmax, fc, Power))

# -------------------------------------------------------------------
# 6) SALVA NO CSV
# -------------------------------------------------------------------
csv_path = os.path.join(WORKDIR, "simulation_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["W (m)", "L (m)", "Vbias (V)", "Gain (dB)", "fc (Hz)", "Power (W)"])
    w.writerows(results)

print("✅ Pronto! Veja:", csv_path)
