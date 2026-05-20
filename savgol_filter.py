import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def post_processing(file_name, sg_window=11, sg_poly=2, freq_min=None, freq_max=None):
    """
    Processa dados do Moku FRA e plota impedância.

    Parâmetros
    ----------
    file_name : str
        Caminho do CSV sem extensão.
    sg_window : int
        Janela do filtro Savitzky-Golay (ímpar). Default 11.
        Aumentar reduz ruído mas pode distorcer ressonâncias estreitas.
    sg_poly : int
        Grau do polinômio do SG. Default 2.
    freq_min, freq_max : float ou None
        Recortar faixa de frequência (Hz) antes de processar.
        Útil para excluir extremos onde o Moku perde fase.
    """
    try:
        df = pd.read_csv(file_name + ".csv", comment="%", header=None)
        df.columns = ['frequency_hz', 'ch1_dbm', 'ch1_phase_deg',
                      'ch2_dbm', 'ch2_phase_deg']

        # --- Recorte de frequência (opcional) ---
        if freq_min is not None:
            df = df[df['frequency_hz'] >= freq_min]
        if freq_max is not None:
            df = df[df['frequency_hz'] <= freq_max]
        df = df.sort_values('frequency_hz', ascending=False).reset_index(drop=True)

        freq = df["frequency_hz"].values

        # ------------------------------------------------------------------ #
        # SUAVIZAÇÃO NAS GRANDEZAS FÍSICAS (amplitude e fase)                 #
        # antes de calcular Z evita amplificação do ruído pela razão V2/V1    #
        # ------------------------------------------------------------------ #
        def sg(x):
            # Garante que window não ultrapasse o tamanho do array
            w = min(sg_window, len(x) if len(x) % 2 == 1 else len(x) - 1)
            return savgol_filter(x, w, sg_poly)

        p1_s  = sg(df["ch1_dbm"].values)
        p2_s  = sg(df["ch2_dbm"].values)
        ph1_s = sg(df["ch1_phase_deg"].values)
        ph2_s = sg(df["ch2_phase_deg"].values)

        # --- Tensões (V_rms em sistema 50 Ω) ---
        v1 = np.sqrt(2 / 5 * (10 ** (p1_s / 10)))
        v2 = np.sqrt(2 / 5 * (10 ** (p2_s / 10)))

        # --- Delta de fase (rad) ---
        delta_phase = np.radians(ph2_s - ph1_s)
        voltage_ratio = v2 / v1

        # --- Impedância complexa ---
        z_r = (voltage_ratio * np.cos(delta_phase) - 1) * 50
        z_i = (voltage_ratio * np.sin(delta_phase)) * 50

        z_mod = np.sqrt(z_r**2 + z_i**2)
        z_ang = np.degrees(np.arctan2(z_i, z_r))

        # --- Impedância teórica do RC ---
        r = 200
        c = 1e-9
        n = 1 # harmonic number
        freq_read = n * freq
        denom = 1 + 4 * (np.pi**2) * (freq_read**2) * (r**2) * (c**2)  
        z_rc_real = r / denom
        z_rc_imag = -(2*np.pi*freq_read*(r**2)*c) / denom

        # ------------------------------------------------------------------ #
        # PLOTAGEM                                                             #
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(file_name.split("/")[-1], fontsize=11, y=1.01)

        # Linha 1: grandezas originais do Moku (diagnóstico)
        axes[0, 0].semilogx(freq, df["ch1_dbm"].values, 'gray', lw=0.8,
                            label='Raw')
        axes[0, 0].semilogx(freq, df["ch2_dbm"].values, 'gray', lw=0.8)
        axes[0, 0].semilogx(freq, p1_s, 'b-', lw=1.5, label='CH1 suav.')
        axes[0, 0].semilogx(freq, p2_s, 'r-', lw=1.5, label='CH2 suav.')
        axes[0, 0].set_title('Amplitude CH1 e CH2 (dBm)')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, which='both', alpha=0.3)

        delta_raw = df["ch2_phase_deg"].values - df["ch1_phase_deg"].values
        axes[0, 1].semilogx(freq, delta_raw, 'gray', lw=0.8, label='Raw')
        axes[0, 1].semilogx(freq, np.degrees(delta_phase), 'b-', lw=1.5,
                            label='Suavizado')
        axes[0, 1].set_title('Δφ = φ₂ − φ₁ (graus)')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, which='both', alpha=0.3)

        axes[0, 2].semilogx(freq, z_mod, 'b-', lw=1.5)
        axes[0, 2].set_title('|Z| (Ω)')
        axes[0, 2].grid(True, which='both', alpha=0.3)

        # Linha 2: impedância calculada
        z_r_raw_plot = (df["ch2_dbm"].values - df["ch1_dbm"].values)  # proxy visual
        # Recalcula Z raw para comparação no plot
        vr_raw = np.sqrt(2/5*(10**(df["ch2_dbm"].values/10))) / \
                 np.sqrt(2/5*(10**(df["ch1_dbm"].values/10)))
        dp_raw = np.radians(df["ch2_phase_deg"].values - df["ch1_phase_deg"].values)
        zr_raw = (vr_raw * np.cos(dp_raw) - 1) * 50
        zi_raw = (vr_raw * np.sin(dp_raw)) * 50

        axes[1, 0].semilogx(freq, zr_raw, 'gray', lw=0.8, alpha=0.6,
                            label='Raw')
        axes[1, 0].semilogx(freq, z_r, 'b-', lw=1.5, label='Suavizado')
        axes[1, 0].semilogx(freq, z_rc_real, 'r--', lw=1.5, label='Teórica')
        axes[1, 0].set_title("Z' real (Ω)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, which='both', alpha=0.3)

        axes[1, 1].semilogx(freq, zi_raw, 'gray', lw=0.8, alpha=0.6,
                            label='Raw')
        axes[1, 1].semilogx(freq, z_i, 'b-', lw=1.5, label='Suavizado')
        axes[1, 1].semilogx(freq, z_rc_imag, 'r--', lw=1.5, label='Teórica')
        axes[1, 1].set_title("Z'' imag (Ω)")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, which='both', alpha=0.3)

        axes[1, 2].plot(z_r, z_i, 'b-', lw=1.5)
        axes[1, 2].plot(z_r, z_i, 'b.', ms=3, alpha=0.5)
        axes[1, 2].plot(z_rc_real, z_rc_imag, 'r--', lw=1.5)
        axes[1, 2].set_title("Nyquist Z")
        axes[1, 2].set_xlabel("Z' (Ω)")
        axes[1, 2].set_ylabel("Z'' (Ω)")
        axes[1, 2].set_aspect('equal', adjustable='datalim')
        axes[1, 2].axhline(0, color='k', lw=0.5, alpha=0.4)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        fig_cap, axes_2 = plt.subplots(1, 2, figsize=(16, 9))
        fig.suptitle(file_name.split("/")[-1], fontsize=11, y=1.01)

        # Linha 1: Capacitância
        C = np.clip(z_i, a_min=None, a_max=0) # termo negativo da impedancia imaginaria
        C_teorica = np.clip(z_rc_imag, a_min=None, a_max=0)
        axes_2[0].semilogx(freq, C, 'b-', lw=0.8,
                            label='Raw')
        axes_2[0].semilogx(freq, C_teorica, 'r--', lw=0.8,
                            label='Teórico')
        axes_2[0].set_title('Capacitância')
        axes_2[0].legend(fontsize=8)
        axes_2[0].grid(True, which='both', alpha=0.3)

        epslon = None
        axes_2[1].semilogx(freq, C, 'b-', lw=0.8,
                            label='Raw')
        axes_2[1].semilogx(freq, C_teorica, 'r--', lw=0.8,
                            label='Teórico')
        axes_2[1].set_title('Permissividade')
        axes_2[1].legend(fontsize=8)
        axes_2[1].grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"     |Z| médio: {z_mod.mean():.1f} Ω  (std: {z_mod.std():.1f} Ω)")
        print(f"     Z_real:    {z_r.mean():.1f} Ω  (std: {z_r.std():.1f} Ω)")
        print(f"     Z_imag:    {z_i.mean():.1f} Ω  (std: {z_i.std():.1f} Ω)")

    except Exception as e:
        print(f'[ERROR] Post-Processing failed: {e}')
        raise


# ── Uso ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    post_processing(
        "pvk_cell1_marcos_1harm_light_20260519_183109_Traces",
        sg_window=11,  
        sg_poly=2,
        freq_min=1,    
        freq_max=1e6,   
    )