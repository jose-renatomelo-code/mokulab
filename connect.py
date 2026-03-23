from moku.instruments import FrequencyResponseAnalyzer

ip = '192.168.73.1'

try: 
    moku = Oscilloscop(ip, force_connect=True, ignore_busy=True)

    print(f'Conectado com sucesso ao: {moku.name()}')

except Exception as e:
    print(f'Falha na conexão: {e}')

finally:
    # Close outputs
    moku.disable_output(channel=1)
    moku.disable_output(channel=2)

    # Fecha API
    print("Fechando conexão API...")
    moku.relinquish_ownership()
    print("Conexão encerrada com sucesso!")