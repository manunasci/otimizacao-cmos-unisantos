import os
import numpy as np
from PyLTSpice import SimRunner, RawRead

LTSPICE_EXE = r"C:\Users\holam\OneDrive - Sociedade Visconde de São Leopoldo\Área de Trabalho\Iniciação Cientifica"

CIRCUIT_FILE =  'CMOS class AB Output STAGES.asc'

def analisar_circuito_saida(wn_out_val, wp_out_val, wn_bias_val, wp_bias_val, bias_val):

    """
    Roda a simulação para um conjunto de parâmetros do amplificador Classe AB
    e extrai as métricas de desempenho.
    
    Args:
        wn_out_val (float): Largura do transistor NMOS de saída (M2) em micrômetros.
        wp_out_val (float): Largura do transistor PMOS de saída (M3) em micrômetros.
        wn_bias_val (float): Largura do transistor NMOS de bias (M8) em micrômetros.
        wp_bias_val (float): Largura do transistor PMOS de bias (M6) em micrômetros.
        bias_val (float): Tensão de polarização (V4) em Volts.
    """
    
    print("="*60)
    print(f"Analisando com: Wn_out={wn_out_val:.1f}u, Wp_out={wp_out_val:.1f}u, "
          f"Wn_bias={wn_bias_val:.1f}u, Wp_bias={wp_bias_val:.1f}u, Bias={bias_val:.2f}V")
    print("="*60)

    try:
        runner = SimRunner(output_folder='./temp', simulator=LTSPICE_EXE)

        runner.set_parameter('Wn_out', f"{wn_out_val}u")
        runner.set_parameter('Wp_out', f"{wp_out_val}u")
        runner.set_parameter('Wn_bias', f"{wn_bias_val}u")
        runner.set_parameter('Wp_bias', f"{wp_bias_val}u")
        runner.set_parameter('bias', bias_val)

        print("Rodando simulações (.ac e .op) no LTSpice...")
        runner.run(CIRCUIT_FILE)
        print("Simulação concluída.")

        if not os.path.exists(raw_file):
            print("\nERRO: Arquivo de resultado não encontrado. A simulação pode ter falhado.")
            return None
        
        data = RawRead(raw_file)

        ac_data = data.get_trace("V(out)")
        freqs = data.get_axis()

        gain_complex = ac_data.get_point(1e3) 
        gain_magnitude = np.abs(gain_complex)
        gain_db = 20 * np.log10(gain_magnitude) if gain_magnitude > 0 else

        gain_max_lin = np.max(np.abs(ac_data.get_wave()))

        try:
              freq_3db_idx = np.where(np.abs(ac_data.get_wave()) < gain_max_lin / np.sqrt(2))[0][0]
            bw_hz = freqs[freq_3db_idx]
        except IndexError:
            bw_hz = freqs[-1] if len(freqs) > 0 else 0 
