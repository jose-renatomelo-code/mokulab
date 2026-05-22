from impedance.models.circuits import Randles, CustomCircuit
from impedance.visualization import plot_nyquist, plot_residuals
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

def processing_mokuFRA(file_name):
    try:
        df = read_csv(file_name + ".csv", comment="%", header=None)
        df.columns = ['frequency_hz', 'ch1_dbm', 'ch1_phase_deg',
                      'ch2_dbm', 'ch2_phase_deg']
        p1 = df['ch1_dbm'].values
        p2 = df['ch2_dbm'].values
        ph1 = df['ch1_phase_deg'].values
        ph2 = df['ch2_phase_deg'].values
        freq = df['frequency_hz'].values

         # --- Tensões (V_rms em sistema 50 Ω) ---
        v1 = np.sqrt(2 / 5 * (10 ** (p1 / 10)))
        v2 = np.sqrt(2 / 5 * (10 ** (p2 / 10)))

        # --- Delta de fase (rad) ---
        delta_phase = np.radians(ph2 - ph1)
        voltage_ratio = v2 / v1

        # --- Impedância complexa ---
        z_r = (voltage_ratio * np.cos(delta_phase) - 1) * 50
        z_i = (voltage_ratio * np.sin(delta_phase)) * 50

        z_mod = np.sqrt(z_r**2 + z_i**2)
        z_ang = np.degrees(np.arctan2(z_i, z_r))

        # Capacitância
        mascara_cap = z_i < 0
        C = np.full_like(z_i, np.nan, dtype=float)
        C = -1 / (2 * np.pi * freq[mascara_cap] * z_i[mascara_cap])

        # === PLOT DADOS MOKU + IMPEDÂNCIA CALCULADA === 
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Dados do FRA e impedância calculada", fontsize=11)

        # Linha 1: grandezas originais do Moku (diagnóstico)
        axes[0, 0].semilogx(freq, p1, 'r-', lw=0.8,
                            label='Ch1')
        axes[0, 0].semilogx(freq, p2, 'b-', lw=0.8, label='Ch2')
        axes[0, 0].set_title('Amplitude CH1 e CH2 (dBm)')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, which='both', alpha=0.3)

        delta_raw = ph1 - ph2
        axes[0, 1].semilogx(freq, delta_raw, 'b-', lw=0.8)
        axes[0, 1].set_title('Δφ = φ₂ − φ₁ (graus)')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, which='both', alpha=0.3)

        axes[0, 2].semilogx(freq, z_mod, 'b-', lw=0.8)
        axes[0, 2].set_title('|Z| (Ω)')
        axes[0, 2].grid(True, which='both', alpha=0.3)

        # Linha 2: gráficos de impedância
        axes[1, 0].semilogx(freq, z_r, 'b-', lw=1.0, alpha=0.6)
        axes[1, 0].set_title("Z' real (Ω)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, which='both', alpha=0.3)

        axes[1, 1].semilogx(freq, z_i, 'b-', lw=1.0)
        axes[1, 1].set_title("Z'' imag (Ω)")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, which='both', alpha=0.3)

        axes[1, 2].semilogx(z_r, z_i, 'b--', lw=1.0)
        axes[1, 2].set_title("Nyquist")
        axes[1, 2].set_xlabel("Z'")
        axes[1, 2].set_ylabel("Z''")
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        #plt.show()
    
    except Exception as e:
        print(f'[ERROR] pre-processing failed: {e}')
    
    return freq, np.asarray(z_r + 1j*z_i, dtype='complex128')

def fitting_impedance(freq, z, predict):
    randles = Randles(initial_guess=[.01, .005, .001, 200, .1])
    randlesCPE = Randles(initial_guess=[50.0, 1500.0, 300.0, 0.1, 1e-7, 0.85], CPE=True)
    customCircuit1 = CustomCircuit(initial_guess=[25.0, 1000.0, 100.0, 300.0, 0.1, 1e-7, 1e-9, 200.0],
                                  circuit='R_0-p(R_1-p(R_2-Wo_1, C1), C_2, R_3)')
    customCircuit2 = CustomCircuit(initial_guess=[25.0, 400.0, 1e-5, 0.95, 300.0, 1e-4, 0.85], 
                                   circuit='R_0-p(R_1, CPE_0)-p(R_2, CPE_1)')
    customCircuit3 = CustomCircuit(initial_guess=[25.0, 500.0, 1e-7, 0.85, 500.0, 0.1], 
                                   circuit='R_0-p(R_1,CPE_1)-Wo_1')

    # Primeiro faz o fit (ajusta parâmetros in-place)
    randles.fit(freq, z)
    randlesCPE.fit(freq, z)
    customCircuit1.fit(freq, z)
    low_bounds =  [10.0,  350.0, 1e-7, 0.85, 200.0,  1e-5, 0.50]
    high_bounds = [50.0,  500.0, 1e-3, 1.00, 5000.0, 1e-1, 0.95]
    customCircuit2.fit(freq, z, bounds=(low_bounds, high_bounds), weight_by_modulus=True)

    #print(randlesCPE)
    print(customCircuit2)

    # Define frequências para predição
    f_plot = np.logspace(np.log10(freq.max()), np.log10(freq.min()), 200) if predict else freq

    # Gera impedâncias preditas
    randles_fit            = randles.predict(f_plot)
    randles_CPE_fit        = randlesCPE.predict(f_plot)
    customCircuit_fit      = customCircuit2.predict(f_plot)


    # Cálculo de resíduos
    res_real_randles = (z - randles_CPE_fit).real/np.abs(z)
    res_imag_randles = (z - randles_CPE_fit).imag/np.abs(z)
    res_real_custom = (z - customCircuit_fit).real/np.abs(z)
    res_imag_custom = (z - customCircuit_fit).imag/np.abs(z)

    
    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    plot_nyquist(z, ax=ax[0])
    #plot_nyquist(randles_fit, fmt='-', ax=ax)
    plot_nyquist(randles_CPE_fit, fmt='-', ax=ax[0])
    plot_nyquist(customCircuit_fit, fmt='-', ax=ax[0])
    ax[0].legend(["Data", "Randles CPE", "Custom"])
    plot_residuals(ax=ax[1], f=freq, res_real=res_real_randles, res_imag=res_imag_randles, y_limits=((-20, 20)))
    plot_residuals(ax=ax[1], f=freq, res_real=res_real_custom, res_imag=res_imag_custom, y_limits=((-20, 20)))


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    freq, z = processing_mokuFRA("Data/pvk_dev1_cell1_day2_light_20260520_145155_Traces")
    fitting_impedance(freq, z, predict=False)