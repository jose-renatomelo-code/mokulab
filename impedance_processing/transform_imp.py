import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def post_processing(file_name):
    try:
        df_moku =pd.read_csv(file_name + ".csv", comment="%", header=None)

        # Define nomes das colunas
        df_moku.columns = [
            'frequency_hz',
            'ch1_dbm',
            'ch1_phase_deg',
            'ch2_dbm',
            'ch2_phase_deg'
        ]
        freq = df_moku["frequency_hz"]
        p1 = df_moku["ch1_dbm"]
        p2 = df_moku["ch2_dbm"]
        phase_rad = np.radians(df_moku["ch1_phase_deg"])
        v1 = np.sqrt(2/5 * (10**(p1/10)))
        v2 = np.sqrt(2/5 * (10**(p2/10)))
        
        # Calcular impedância 
        z_mod = ((v2/v1) - 1) * 50
        z_i = z_mod * np.sin(phase_rad)
        z_r = z_mod * np.cos(phase_rad)


        # PLOTAGEM
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        axes = axes.flatten()

        plot_cfg = [
            ('Impedance module (Ω)', freq, z_mod, True), 
            ('Imaginary Impedance (Ω)', freq, z_i, True),
            ('Nyquist Plot Z', z_r, z_i, False)
        ]

        for idx, (title, x, y, is_log) in enumerate(plot_cfg):
            if is_log:
                axes[idx].semilogx(x, y, 'b-')
            else:
                axes[idx].plot(x, y, 'r-')
            axes[idx].set_title(title)
            axes[idx].grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f'[ERROR] Post-Processing failed: {e}')

post_processing("10k_rawmoku_20260512_185105_Traces")
