from moku.instruments import Oscilloscope

ip = '[fe80::7269:79ff:feb7:fed%3]'

try: 
    moku = Oscilloscope(ip, force_connect=True)

    print(f'Conectado ocm sucesso ao: {moku.name()}')

except Exception as e:
    print(f'Falha na conexão: {e}')