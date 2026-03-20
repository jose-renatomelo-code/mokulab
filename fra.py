from moku.instruments import FrequencyResponseAnalyzer as fra
import time
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
i = fra('192.168.73.1', ignore_busy=True, force_connect=True)
i.claim_ownership(force_connect=True)

def convert_to_impedance(Pdbm1, Pdbm2, phi):
    # Forçar conversão para float64 e tratar como array NumPy
    p1 = np.asarray(Pdbm1, dtype=np.float64)
    p2 = np.asarray(Pdbm2, dtype=np.float64)
    p_deg = np.asarray(phi, dtype=np.float64)
    
    # Cálculo de Tensão (RMS)
    v1_volt = np.sqrt(2/5 * (10**(p1 / 10)))
    v2_volt = np.sqrt(2/5 * (10**(p2 / 10)))
    
    # Cálculo da Impedância
    # Z = ((V2/V1) - 1) * 50
    ratio = np.divide(v2_volt, (v1_volt + 1e-12))
    z_mod = (ratio - 1) * 50 
    
    phi_rad = np.radians(p_deg)
    z_real = z_mod * np.cos(phi_rad)
    z_img = z_mod * np.sin(phi_rad)
    
    # Admitância
    denom = (z_mod**2 + 1e-12)
    adm_real = z_real / denom
    adm_img = -z_img / denom
    
    return z_real, z_img, adm_real, adm_img, v1_volt, v2_volt

def plot_RealTime(num_points):
    fig = plt.figure(figsize=(14, 9))
    axes = []
    lines = []
    
    plot_cfg = [
        ('Magnitude (dBm)', 231, True), ('Phase (deg)', 232, True),
        ('Real Impedance (Ω)', 233, True), ('Imaginary Impedance (Ω)', 234, True),
        ('Nyquist Plot', 235, False), ('Imaginary Admittance (S)', 236, True)
    ]
    
    for title, sub, is_log in plot_cfg:
        ax = fig.add_subplot(sub)
        if is_log:
            l1, = ax.semilogx([], [], label='Ch1')
            l2, = ax.semilogx([], [], label='Ch2') if sub in [231, 232] else (None,)
        else:
            l1, = ax.plot([], [])
            l2 = None
        
        ax.set_title(title, fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        lines.append((l1, l2))
        axes.append(ax)

    plt.ion()
    plt.show()

    try:
        while True:
            data = i.get_data()
            ch1, ch2 = data['ch1'], data['ch2']
            
            freq = np.asarray(ch1['frequency'], dtype=np.float64)
            if len(freq) == 0:
                plt.pause(0.1)
                continue

            # Chama a álgebra corrigida
            z_r, z_i, a_r, a_i, v1, v2 = convert_to_impedance(
                ch1['magnitude'], ch2['magnitude'], ch2['phase']
            )

            # Atualização dos eixos
            plot_data = [
                (freq, ch1['magnitude'], freq, ch2['magnitude']),
                (freq, ch1['phase'], freq, ch2['phase']),
                (freq, z_r, None, None),
                (freq, z_i, None, None),
                (z_r, z_i, None, None),
                (freq, a_i, None, None)
            ]

            for idx, (x1, y1, x2, y2) in enumerate(plot_data):
                lines[idx][0].set_data(x1, y1)
                if lines[idx][1] is not None:
                    lines[idx][1].set_data(x2, y2)
                axes[idx].relim()
                axes[idx].autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # CONDIÇÃO DE PARADA
            if len(freq) >= num_points:
                i.stop_sweep()
                print(f"\nVarredura finalizada com {len(freq)} pontos.")
                break
            
            plt.pause(0.1)

        # Exportação final
        final_stack = np.column_stack((freq, ch1['magnitude'], ch2['magnitude'], v1, v2, z_r, z_i, a_r, a_i))
        np.savetxt("dados_finais.txt", final_stack, header="Freq,Mag1,Mag2,V1,V2,Zr,Zi,Ar,Ai")
        fig.savefig("fra_plots.png", dpi=300)
        print("\nArquivos finais salvos com sucesso!")

    except Exception as e:
        print(f"Erro: {e}")

def ProcessSweepAndPlot(num_points):
    last_len = 0
    # Imprime o cabeçalho apenas UMA vez antes de começar
    print(f"\n{'Ponto':<6} | {'Freq (Hz)':<12} | {'Mag CH1':<10} | {'Mag CH2':<10} | {'Fase CH2':<10}")
    print("-" * 60)
    
    try:
        while True:
            data = i.get_data()
            ch1, ch2 = data['ch1'], data['ch2']
            
            freq = np.asarray(ch1['frequency'], dtype=np.float64)
            current_len = len(freq)

            # === Update Table (Imprime apenas as linhas novas) === 
            if current_len > last_len:
                for idx in range(last_len, current_len):
                    # Acessando o índice [idx] de cada campo para a tabela
                    f = freq[idx]
                    m1 = ch1['magnitude'][idx]
                    m2 = ch2['magnitude'][idx]
                    p2 = ch2['phase'][idx]
                    print(f"{idx+1:<6} | {f:<12.1f} | {m1:<10.2f} | {m2:<10.2f} | {p2:<10.2f}")
                
                last_len = current_len

            # CONDIÇÃO DE PARADA
            if current_len >= num_points:
                i.stop_sweep()
                print(f"\nVarredura finalizada com {current_len} pontos.")
                break
            
            time.sleep(0.1) # Evita sobrecarga de pooling na rede

        # ==========================================
        #       GERAÇÃO DO PLOT COMPLETO (FINAL)
        # ==========================================
        print("Gerando gráficos e arquivos...")
        
        # Chama a álgebra corrigida (uma única vez para todo o array)
        z_r, z_i, a_r, a_i, v1, v2 = convert_to_impedance(
            ch1['magnitude'], ch2['magnitude'], ch2['phase']
        )

        fig = plt.figure(figsize=(15, 10))
        plot_cfg = [
            ('Real Impedance (Ω)', 231, True, freq, z_r), 
            ('Imaginary Impedance (Ω)', 232, True, freq, z_i),
            ('Nyquist Plot Z', 233, False, z_r, z_i), 
            ('Real Admittance (S)', 234, True, freq, a_r),
            ('Imaginary Admittance (S)', 235, True, freq, a_i), 
            ('Nyquist Plot A', 236, False, a_r, a_i)
        ]

        for title, sub, is_log, x_data, y_data in plot_cfg:
            ax = fig.add_subplot(sub)
            if is_log:
                ax.semilogx(x_data, y_data, 'b-', label='Data')
            else:
                ax.plot(x_data, y_data, 'r-', label='Data')
            
            ax.set_title(title, fontsize=10)
            ax.grid(True, which='both', alpha=0.3)
            
            # Ajuste de labels específicos para Nyquist
            if "Nyquist" in title:
                ax.set_xlabel("Real")
                ax.set_ylabel("Imaginary")

        # Exportação final
        final_stack = np.column_stack((freq, ch1['magnitude'], ch2['magnitude'], v1, v2, z_r, z_i, a_r, a_i))
        np.savetxt("dados_finais.txt", final_stack, 
                   header="Freq,Mag1,Mag2,V1,V2,Zr,Zi,Ar,Ai", delimiter=",")
        
        plt.tight_layout()
        fig.savefig("fra_plots.png", dpi=300)
        print("Arquivos 'dados_finais.txt' e 'fra_plots.png' salvos!") 
        plt.show()

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise e
try: 
    # Modo Input 
    i.measurement_mode("In")
    # =========================== 
    #    Configurar Canal 1
    # =========================== 
    # Input 1
    i.set_frontend(channel=1, impedance='50Ohm', coupling='DC',
                   range='1Vpp', bandwidth='200MHz', strict=True)
    # Output 1
    i.set_output(channel=1, amplitude=0.05, enable_amplitude=True)
    i.set_output_termination(channel=1, termination='50Ohm', strict=True)
    
    # ===========================
    #    Configurar Canal 2
    # ===========================   
    # Input 2
    i.set_frontend(channel=2, impedance='50Ohm', coupling='DC',
                   range='1Vpp', bandwidth='200MHz', strict=True)
    # Output 2
    i.set_output(channel=2, amplitude=0.05, enable_amplitude=True)
    i.set_output_termination(channel=2, termination='50Ohm', strict=True)
    
    # Config Sweep
    num_points = 128
    i.set_sweep(start_frequency=1, stop_frequency=1e6, num_points=num_points,
                averaging_time=2e-5, averaging_cycles=1, settling_time=1e-5,
                settling_cycles=1, dynamic_amplitude=False
                )
    # Frontend Parameters
    sweep_info = i.get_sweep()
    estimated_time = sweep_info['estimated_sweep_time']
    min, seg = divmod(estimated_time, 60)
    print(f"Tempo estimado da varredura: {int(min)}min {int(seg)}s")
    continuar = input("Prosseguir? (s/n) ")
    if continuar == 's':
        # === Select Harmonic === 
        harm = int(input("Digite o harmônico desejado: "))
        i.set_harmonic_multiplier(multiplier=harm)
        print(i.summary())

        # ===========================================
        #            RUN SWEEP AND PLOT 
        # ===========================================
        print("\nExecutando varredura na frequência...")
        r_t = input("Plot Data in Real-Time? (s/n) ")
        i.start_sweep()
        if r_t == 's':
            # Heavy function
            plot_RealTime(num_points)
        # Get Sweep Data and plot in the end
        ProcessSweepAndPlot(num_points)
        
except Exception as e: 
    print(f"Erro na execução do FRA: {e}")
    
finally:
    # Close outputs
    i.disable_output(channel=1)
    i.disable_output(channel=2)

    # Fecha API
    print("Fechando conexão API...")
    i.relinquish_ownership()
    print("Conexão encerrada com sucesso!")