import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def post_processing(file_name, parasitic_file="parasitics.txt",
                    sg_window=11, sg_poly=2):
    try:
        df = pd.read_csv(file_name + ".csv", comment="%", header=None)
        df.columns = ['frequency_hz', 'ch1_dbm', 'ch1_phase_deg',
                      'ch2_dbm', 'ch2_phase_deg']

        freq = df["frequency_hz"].values

        # --- Suavização nas grandezas físicas (reduz amplificação do ruído) ---
        def sg(x):
            w = min(sg_window, len(x) if len(x) % 2 == 1 else len(x) - 1)
            return savgol_filter(x, w, sg_poly)

        p1  = sg(df["ch1_dbm"].values)
        p2  = sg(df["ch2_dbm"].values)
        ph1 = sg(df["ch1_phase_deg"].values)
        ph2 = sg(df["ch2_phase_deg"].values)

        # --- Tensões e impedância medida ---
        v1 = np.sqrt(2 / 5 * (10 ** (p1 / 10)))
        v2 = np.sqrt(2 / 5 * (10 ** (p2 / 10)))

        delta_phase   = np.radians(ph2 - ph1)
        voltage_ratio = v2 / v1

        z_r = (voltage_ratio * np.cos(delta_phase) - 1) * 50
        z_i = (voltage_ratio * np.sin(delta_phase)) * 50
        z_m = z_r + 1j * z_i                            # Z medida (complexa)

        # --- Correção de parasitas ---
        df_par = pd.read_csv(parasitic_file)

        # Suporta colunas complexas escritas como string "(a+bj)"
        # ou colunas reais/imaginárias separadas
        if "Zp" in df_par.columns and "Zs" in df_par.columns:
            z_p = df_par["Zp"].apply(complex).to_numpy()
            z_s = df_par["Zs"].apply(complex).to_numpy()
        else:
            raise ValueError("parasitics.txt deve ter colunas 'Zp' e 'Zs'")

        # Verificação de dimensões
        if len(z_p) != len(freq) or len(z_s) != len(freq):
            raise ValueError(
                f"parasitics.txt tem {len(z_p)} pontos mas o CSV tem {len(freq)}. "
                "As medidas de calibração devem usar a mesma grade de frequência."
            )

        # ✅ Fórmula correta: modelo série-paralelo
        #    Z_meas = Zs + Zp ∥ (Zs + Z_dut)
        #    Z_dut  = Zp·(Z_meas - Zs) / (Zp + Zs - Z_meas)  -  Zs
        denom = z_p + z_s - z_m
        
        # Avisa se o denominador estiver próximo de zero (ressonância do fixture)
        denom_min = np.abs(denom).min()
        if denom_min < 1.0:
            idx_bad = np.argmin(np.abs(denom))
            print(f"[AVISO] Denominador próximo de zero em "
                  f"{freq[idx_bad]:.1f} Hz (|denom|={denom_min:.3f}). "
                  "Resultado instável nessa frequência.")

        z_dut = z_p * (z_m - z_s) / denom - z_s

        zr_dut = np.real(z_dut)
        zi_dut = np.imag(z_dut)

        # --- Plotagem ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].semilogx(freq, zr_dut, 'b-', lw=1.5)
        axes[0].set_title("Real Impedance (Ω)")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].grid(True, which='both', alpha=0.3)

        axes[1].semilogx(freq, zi_dut, 'b-', lw=1.5)
        axes[1].set_title("Imaginary Impedance (Ω)")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].grid(True, which='both', alpha=0.3)

        # ✅ Nyquist usa Z corrigida
        axes[2].plot(zr_dut, zi_dut, 'r-', lw=1.5)
        axes[2].plot(zr_dut, zi_dut, 'r.', ms=3, alpha=0.5)
        axes[2].set_title("Nyquist Plot Z")
        axes[2].set_xlabel("Z' (Ω)")
        axes[2].set_ylabel("Z'' (Ω)")
        axes[2].axhline(0, color='k', lw=0.5, alpha=0.4)
        axes[2].set_aspect('equal', adjustable='datalim')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"[OK] Z_real médio: {zr_dut.mean():.1f} Ω  |  "
              f"Z_imag médio: {zi_dut.mean():.1f} Ω")

    except Exception as e:
        print(f'[ERROR] Post-Processing failed: {e}')
        raise
    
post_processing("10k_moku_10cycles_20260513_165008_Traces")
