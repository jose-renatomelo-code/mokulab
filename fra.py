from moku.instruments import FrequencyResponseAnalyzer as fra
import time
import threading
from queue import Queue
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

ipv4 = "192.168.73.1"
ipv6 = "[fe80::7269:79ff:feb7:fed%3]"
i = fra(ipv4, ignore_busy=True, force_connect=True)
i.claim_ownership(force_connect=True)


def convert_to_impedance(Pdbm1, Pdbm2, phi):
    # Forçar conversão para float64 e tratar como array NumPy
    p1 = np.asarray(Pdbm1, dtype=np.float64)
    p2 = np.asarray(Pdbm2, dtype=np.float64)
    p_deg = np.asarray(phi, dtype=np.float64)
    
    # Cálculo de Tensão 
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

def streaming_moku(num_points):
    fig = plt.figure(figsize=(14, 9))
    final_data = None
    
    # Magnitude (dbm) vs frequência (Hz)
    ax1 = fig.add_subplot(211)
    
    (l1,) = ax1.semilogx([], [], label='Ch1')
    (l2,) = ax1.semilogx([], [], label='Ch2') 
    
    ax1.set_title("Magnitude vs Frequency", fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)

    # Phase (degrees) vs Frequency (Hz)
    ax2 = fig.add_subplot(212)
    
    (l3,) = ax2.semilogx([], [], label='Ch1')
    (l4,) = ax2.semilogx([], [], label='Ch2') 
    
    ax2.set_title("Phase vs Frequency", fontsize=10)
    ax2.grid(True, which='both', alpha=0.3)


    plt.ion()
    plt.show()

    try:
        while True:
            data = i.get_data()
            ch1, ch2 = data['ch1'], data['ch2']
            l1.set_data(ch1["frequency"], ch1["magnitude"])
            l2.set_data(ch2["frequency"], ch2["magnitude"])
            l3.set_data(ch1["frequency"], ch1["phase"])
            l4.set_data(ch2["frequency"], ch2["phase"])

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # CONDIÇÃO DE PARADA
            freq = np.asarray(ch1['frequency'], dtype=np.float64)
            mag1 = np.asarray(ch1["magnitude"], dtype=np.float64)
            mag2 = np.asarray(ch2["magnitude"], dtype=np.float64)

            has_nan = (
                np.isnan(mag1).any() or
                np.isnan(mag2).any()
            )
            if len(freq) >= num_points and not has_nan:
                i.stop_sweep()
                print(f"\nVarredura finalizada com {len(freq)} pontos, pronto para plotagem.")
                plt.close()
                final_data = data
                break

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            plt.draw()
            plt.pause(1)

    except Exception as e:
        print(f"Erro: {e}")
    return final_data, fig

def batching_moku(num_points, estimated_time):
    print(f"Aguardando término da varredura (estimado: {estimated_time:.1f}s)...")
    try:
        while True:
            # 1. A REQUISIÇÃO 
            data = i.get_data(timeout=estimated_time+60)
            freq = np.asarray(data['ch1']['frequency'], dtype=np.float64)
            if len(freq) >= num_points:
                i.stop_sweep()
                break
            
        ch1 = data['ch1']
        ch2 = data['ch2']
        current_len = len(freq)

        # 2. MOSTRAR TABELA NO TERMINAL
        print(f"\n{'Ponto':<6} | {'Freq (Hz)':<12} | {'Mag CH1':<10} | {'Mag CH2':<10} | {'Fase CH2':<10}")
        print("-" * 65)
        
        for idx in range(current_len):
            f = freq[idx]
            m1 = ch1['magnitude'][idx]
            m2 = ch2['magnitude'][idx]
            p2 = ch2['phase'][idx]
            print(f"{idx+1:<6} | {f:<12.1f} | {m1:<10.2f} | {m2:<10.2f} | {p2:<10.2f}")
        
        print(f"\nVarredura finalizada com {len(freq)} pontos.")

        # 3. PROCESSAMENTO MATEMÁTICO
        print("\nGerando gráficos finais e arquivos...")
        z_r, z_i, a_r, a_i, v1, v2 = convert_to_impedance(
            ch1['magnitude'], ch2['magnitude'], ch2['phase']
        )

        # 4. PLOTAGEM
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        plot_cfg = [
            ('Real Impedance (Ω)', freq, z_r, True), 
            ('Imaginary Impedance (Ω)', freq, z_i, True),
            ('Nyquist Plot Z', z_r, z_i, False), 
            ('Real Admittance (S)', freq, a_r, True),
            ('Imaginary Admittance (S)', freq, a_i, True), 
            ('Nyquist Plot A', a_r, a_i, False)
        ]

        for idx, (title, x, y, is_log) in enumerate(plot_cfg):
            if is_log:
                axes[idx].semilogx(x, y, 'b-')
            else:
                axes[idx].plot(x, y, 'r-')
            axes[idx].set_title(title)
            axes[idx].grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        fig.savefig("fra_plots.png", dpi=300)
        
        # 5. EXPORTAÇÃO CSV
        df = pd.DataFrame({
            'Freq': freq, 'Mag1': ch1['magnitude'], 'Mag2': ch2['magnitude'],
            'Zr': z_r, 'Zi': z_i, 'Ar': a_r, 'Ai': a_i
        })
        df.to_csv("dados_finais.csv", index=False)
        
        print("[SUCESSO] Arquivos 'dados_finais.csv' e 'fra_plots.png' salvos!")
        plt.show()

    except Exception as e:
        print(f"\n[ERRO] Falha na comunicação ou processamento: {e}")


# === NEW STRUCTURE (Threading) ===
def acquire_data(i, data_queue, stop_event):
    last_len = 0
    while not stop_event.is_set():
        try: 
            # O FRA pode demorar para responder se o settling_time for alto
            data = i.get_data(timeout=5) 

            if data and 'ch1' in data:
                freq = np.asarray(data['ch1']['frequency'])
                current_len = len(freq)
                
                # Só coloca na fila se o hardware de fato adicionou pontos novos
                if current_len > last_len:
                    try:
                        data_queue.put_nowait(data)
                        last_len = current_len
                    except:
                        pass 

            time.sleep(0.1) 

        except Exception as e:
            # Se for timeout da API do Moku, ignoramos e tentamos de novo
            time.sleep(0.2)

def process_data(num_points, data_queue):
    final_data = None
    while True:
        try: 
            data = data_queue.get(timeout=2)
        except:
            print('[WARN] Sem novos dados (timeout local)')
            continue

        ch1 = data['ch1']
        freq = np.asarray(ch1['frequency'], dtype=np.float64)
        print(f"Pontos adquiridos: {len(freq)}", end='\r')
        if len(freq) >= num_points:
            final_data = data
            break

    return final_data

def post_processing(final_data, file_name, fig_moku):
    try:
        ch1 = final_data['ch1']
        ch2 = final_data['ch2']
        freq = np.asarray(ch1['frequency'], dtype=np.float64)
        print("\nGerando gráficos finais e arquivos...")
        z_r, z_i, a_r, a_i, v1, v2 = convert_to_impedance(
            ch1['magnitude'], ch2['magnitude'], ch2['phase']
        )

        # PLOTAGEM
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        plot_cfg = [
            ('Real Impedance (Ω)', freq, z_r, True), 
            ('Imaginary Impedance (Ω)', freq, z_i, True),
            ('Nyquist Plot Z', z_r, z_i, False), 
            ('Real Admittance (S)', freq, a_r, True),
            ('Imaginary Admittance (S)', freq, a_i, True), 
            ('Nyquist Plot A', a_r, a_i, False)
        ]

        for idx, (title, x, y, is_log) in enumerate(plot_cfg):
            if is_log:
                axes[idx].semilogx(x, y, 'b-')
            else:
                axes[idx].plot(x, y, 'r-')
            axes[idx].set_title(title)
            axes[idx].grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        # Save impedance plot
        fig.savefig(file_name + '.png', dpi=300)
        # Save moku data plot
        fig.savefig(file_name + 'moku-data' +'.png', dpi=300)

        
        # EXPORTAÇÃO CSV
        df = pd.DataFrame({
            'Freq': freq, 'Mag1': ch1['magnitude'], 'Mag2': ch2['magnitude'],
            'Zr': z_r, 'Zi': z_i, 'Ar': a_r, 'Ai': a_i
        })
        df.to_csv(file_name + '.csv', index=False)
        
        print("[SUCESSO] Arquivos finais salvos!")

    except Exception as e:
        print(f'[ERROR] Post-Processing failed: {e}')

def threading(): 
    data_queue = Queue(maxsize=100)
    stop_event = threading.Event()
    acq_thread = threading.Thread(target=acquire_data, args=(i, data_queue, stop_event), daemon=True)
    acq_thread.start()
    final_data = process_data(num_points, data_queue)
    if final_data: 
        i.stop_sweep()
        stop_event.set()
        acq_thread.join()
        fig_name = input('Nome do arquivo PNG: ')
        csv_name = input('Nome do arquivo TXT: ')
        post_processing(final_data, fig_name, csv_name)
        
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
    i.set_output(channel=1, amplitude=2*0.2, enable_amplitude=True)
    i.set_output_termination(channel=1, termination='50Ohm', strict=True)
    
    # ===========================
    #    Configurar Canal 2
    # ===========================   
    # Input 2
    i.set_frontend(channel=2, impedance='50Ohm', coupling='DC',
                   range='1Vpp', bandwidth='200MHz', strict=True)
    # Output 2
    i.set_output(channel=2, amplitude=2*0.2, enable_amplitude=True)
    i.set_output_termination(channel=2, termination='50Ohm', strict=True)
    
    # Config Sweep
    num_points = 128
    i.set_sweep(start_frequency=1e6, stop_frequency=1, num_points=num_points,
                averaging_time=1e-1, averaging_cycles=10, settling_time=1e-1,
                settling_cycles=10, dynamic_amplitude=False, linear_scale=False
                )
    # Frontend Parameters
    sweep_info = i.get_sweep()
    estimated_time = sweep_info['estimated_sweep_time']
    min, seg = divmod(estimated_time, 60)
    print(f"Tempo estimado da varredura: {int(min)}min {int(seg)}s")
    print(i.summary())
    continuar = input("Prosseguir? (s/n) ")
    if continuar == 's':
        # === Select Harmonic === 
        harm = int(input("Digite o harmônico desejado: "))
        i.set_harmonic_multiplier(multiplier=harm)
        # ===========================================
        #            RUN SWEEP PROCESS 
        # ===========================================
        print("\nExecutando varredura na frequência...")
        i.stop_sweep()
        i.start_sweep()
        time.sleep(0.5)
        final_data, fig_moku = streaming_moku(num_points)
        if final_data:
            file_name = input("Nome do arquivo: ")
            post_processing(final_data, file_name, fig_moku)


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
    