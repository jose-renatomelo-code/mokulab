from moku.instruments import FrequencyResponseAnalyzer as fra
import time
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
        #         RUN SWEEP AND GET MOKU DATA
        # ===========================================
        print("\nExecutando varredura na frequência...")
        i.start_sweep()
        moku_data = i.get_data(wait_complete=True)
        print(moku_data['ch1']['frequency'],
            moku_data['ch1']['magnitude'],
            moku_data['ch1']['phase'])
        print(moku_data['ch2']['frequency'],
            moku_data['ch2']['magnitude'],
            moku_data['ch2']['phase'])
    

except Exception as e: 
    print(f"Erro na execução do FRA: {e}")
    
finally:
    # Stop receiving data
    time.sleep(5)
    i.stop_sweep()
    # Close outputs
    i.disable_output(channel=1)
    i.disable_output(channel=2)

    # Fecha API
    print("Fechando conexão API...")
    i.relinquish_ownership()
    print("Conexão encerrada com sucesso!")