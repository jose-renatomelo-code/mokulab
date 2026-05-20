from impedance.models.circuits import Randles, CustomCircuit
from impedance.visualization import plot_nyquist
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

        axes[1, 2].plot(z_r, z_i, 'b.', lw=0.8)
        axes[1, 2].set_title("Nyquist Z")
        axes[1, 2].set_xlabel("Z' (Ω)")
        axes[1, 2].set_ylabel("Z'' (Ω)")
        axes[1, 2].set_aspect('equal', adjustable='datalim')
        axes[1, 2].axhline(0, color='k', lw=0.5, alpha=0.4)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].invert_yaxis()

        plt.tight_layout()
        #plt.show()
    
    except Exception as e:
        print(f'[ERROR] pre-processing failed: {e}')
    
    return freq, np.asarray(z_r + 1j*z_i, dtype='complex128')

def fitting_impedance(freq, z, predict):
    randles = Randles(initial_guess=[.01, .005, .001, 200, .1])
    randlesCPE = Randles(initial_guess=[.01, .005, .001, 200, .1, .9], CPE=True)
    customCircuit = CustomCircuit(initial_guess=[.01, .005, .1, .005, .1, .001, 200],
                                  circuit='R_0-p(R_1,C_1)-p(R_2,C_2)-Wo_1')
    customConstantCircuit = CustomCircuit(initial_guess=[None, .005, .1, .005, .1, .001, None],
                                          constants={'R_0': 0.02, 'Wo_1_1': 200},
                                          circuit='R_0-p(R_1,C_1)-p(R_2,C_2)-Wo_1')

    # Primeiro faz o fit (ajusta parâmetros in-place)
    randles.fit(freq, z)
    randlesCPE.fit(freq, z)
    customCircuit.fit(freq, z)
    customConstantCircuit.fit(freq, z)

    # Define frequências para predição
    f_plot = np.logspace(np.log10(freq.max()), np.log10(freq.min()), 200) if predict else freq

    # Gera impedâncias preditas
    randles_fit            = randles.predict(f_plot)
    randles_CPE_fit        = randlesCPE.predict(f_plot)
    customCircuit_fit      = customCircuit.predict(f_plot)
    customConstantCircuit_fit = customConstantCircuit.predict(f_plot)

    print(randles_CPE_fit)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_nyquist(z, ax=ax)
    #plot_nyquist(randles_fit, fmt='-', ax=ax)
    plot_nyquist(randles_CPE_fit, fmt='-', ax=ax)
    #plot_nyquist(customCircuit_fit, fmt='-', ax=ax)
    #plot_nyquist(customConstantCircuit_fit, fmt='-', ax=ax)
    #ax.legend(["Data", "Randles", "Randles CPE", "CustomCircuit", "CustomConstantCircuit"])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    freq, z = processing_mokuFRA("Data/pvk_cell1_marcos_1harm_light_20260519_183109_Traces")
    fitting_impedance(freq, z, predict=False)