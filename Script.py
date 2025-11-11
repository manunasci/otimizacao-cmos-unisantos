import os
import glob
import subprocess
import itertools
import numpy as np
import csv

# -------------------------------------------------------------------
# 1) CAMINHO ABSOLUTO DA SUA BIBLIOTECA .lib
# -------------------------------------------------------------------
LIB_PATH = r"C:\Users\holam\OneDrive - Sociedade Visconde de São Leopoldo\Área de Trabalho\otimizacao-cmos-unisantos\C5_models_SPICE.txt"

# -------------------------------------------------------------------
# 2) LOCALIZAÇÃO DO EXECUTÁVEL LTspice
# -------------------------------------------------------------------
candidates = glob.glob(r"C:\Program Files\ADI\LTspice\LTspice.exe")
if not candidates:
    raise FileNotFoundError("Não achei LTspice.exe em 'C:\\Program Files\\ADI'.")
LTSPICE_EXE = candidates[0]

# -------------------------------------------------------------------
# 3) CONFIGURAÇÃO DE PARÂMETROS (AJUSTADO PARA O .asc)
# -------------------------------------------------------------------
WORKDIR     = "simulacoes"
os.makedirs(WORKDIR, exist_ok=True)

# O .asc varre Vbias de 1 a 5V com passos de 0.5V 
# W e L são fixos no novo circuito 
Vbias_vals = np.arange(1.0, 5.0 + 0.5, 0.5)

# -------------------------------------------------------------------
# 4) TEMPLATE DE NETLIST (AJUSTADO PARA O .asc)
# -------------------------------------------------------------------
NETLIST_TEMPLATE = f"""* Netlist adaptada do 'CMOS class AB Output STAGES_FINAL.asc'
.param Vbias={{Vbias}}

* Modelos de fallback
.model nmos NMOS
.model pmos PMOS

* Biblioteca de processo
.include "{LIB_PATH}"

* Componentes (Baseado no .asc) [cite: 1, 2, 3, 4]
M8   out bias 0 0     nmos l=0.6u w=5u
M6   out in   Vdd Vdd pmos l=0.6u w=10u
M3   in  in   Vdd Vdd pmos l=0.6u w=10u
M2   in  bias 0 0     nmos l=0.6u w=5u

* Fontes (Baseado no .asc) 
V4    bias 0      DC {{Vbias}}
Vdd   Vdd  0      DC 5
V2    in   0      AC 1
Vdd1  out  0      DC 0  * Sonda de corrente para medição (como no .asc)

* Análises
.op
.ac dec 1000 1e8 1e9 * Range do .asc 

* Medições no log (Medições do script original)
.measure ac GMAX MAX mag(v(out)/v(in))
.measure ac FC   WHEN mag(v(out)/v(in))=GMAX/sqrt(2)

.end
"""

# -------------------------------------------------------------------
# 5) VARREDURA E COLETA (AJUSTADO PARA VARRER SÓ Vbias)
# -------------------------------------------------------------------
results = []
# Loop alterado para varrer apenas Vb
for Vb in Vbias_vals:
    tag      = f"V{int(Vb*1e3)}mV"
    cir_path = os.path.join(WORKDIR, f"amp_{tag}.cir")
    log_path = os.path.join(WORKDIR, f"amp_{tag}.log")

    # Gera netlist
    with open(cir_path, "w") as f:
        # Format alterado para passar apenas Vbias
        f.write(NETLIST_TEMPLATE.format(Vbias=Vb))

    # Roda LTspice em batch/ASCII
    cmd = [LTSPICE_EXE, "-ascii", "-b", cir_path, "-log", log_path]
    print("⏳ Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Parse das medições no log (Lógica mantida)
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
                    # Mede a corrente da fonte Vdd (que agora é 5V)
                    Idd = abs(float(line.split()[1]))
                except:
                    pass

    # Cálculo de potência ajustado para Vdd=5V 
    Power = 5.0 * Idd
    # Resultado alterado para refletir a nova varredura
    results.append((Vb, Gmax, fc, Power))

# -------------------------------------------------------------------
# 6) SALVA NO CSV (AJUSTADO PARA NOVOS DADOS)
# -------------------------------------------------------------------
csv_path = os.path.join(WORKDIR, "simulation_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    # Colunas do CSV alteradas
    w.writerow(["Vbias (V)", "Gain (dB)", "fc (Hz)", "Power (W)"])
    w.writerows(results)

print("✅ Pronto! Veja:", csv_path)