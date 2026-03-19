from moku.instruments import FrequencyResponseAnalyzer as fra
import time
import pandas as pd
import matplotlib.pyplot as plt 
i = fra('192.168.73.1')
i.claim_ownership(force_connect=True)
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
    i.set_sweep(start_frequency=1, stop_frequency=1e6, num_points=128,
                averaging_time=2e-6, averaging_cycles=1, settling_time=1e-6,
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
        #         RUN SWEEP AND PLOT 
        # ===========================================
        print("\nExecutando varredura na frequência...")
        i.start_sweep()

        # Magnitude Plot Parameters
        plt.subplot(211)
        (line1,) = plt.semilogx([])
        (line2,) = plt.semilogx([])
        ax1 = plt.gca()
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dBm)')
        ax1.grid()
        # Phase Plot Parameters
        plt.subplot(212)
        (line3,) = plt.semilogx([])
        (line4,) = plt.semilogx([])
        ax2 = plt.gca()
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.grid()

        plt.ion()
        plt.show()

        while True: 
            #=== Real-Time Plot ===
            moku_data = i.get_data(wait_complete=True)

            plt.subplot(211)
            ch1_data = moku_data['ch1']
            ch2_data = moku_data['ch2']

            line1.set_xdata(ch1_data['frequency'])
            line1.set_ydata(ch1_data['magnitude'])
            line2.set_xdata(ch2_data['frequency'])
            line2.set_ydata(ch2_data['magnitude'])

            plt.subplot(212)
            line3.set_xdata(ch1_data['frequency'])
            line3.set_ydata(ch1_data['phase'])
            line4.set_xdata(ch2_data['frequency'])
            line4.set_ydata(ch2_data['phase'])

            ax1.set_xlim(min(ch1_data['frequency']), max(ch1_data['frequency']))
            ax1.relim()
            ax1.autoscale_view()
            ax2.set_xlim(min(ch2_data['frequency']), max(ch2_data['frequency']))
            ax2.relim()
            ax2.autoscale_view()

            plt.draw()
            plt.pause(0.001)

except Exception as e: 
    print(f"Erro na execução do FRA: {e}")
    
finally:
    # Stop receiving data
    time.sleep(5)
    i.stop_sweep()
    # Close outputs
    i.disable_output(channel=1)
    i.disable_output(channel=2)

    # Save txt
    df_moku = pd.DataFrame(moku_data)
    df_moku.to_csv('moku_fra_data.txt', sep='\t')

    # Fecha API
    print("Fechando conexão API...")
    i.relinquish_ownership()
    print("Conexão encerrada com sucesso!")