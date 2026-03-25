from moku.instruments import MultiInstrument, FrequencyResponseAnalyzer, SpectrumAnalyzer

m = MultiInstrument('192.168.73.1', platform_id=2, force_connect=True, ignore_busy=True)

# Main
try:
    fra = m.set_instrument(1, FrequencyResponseAnalyzer)
    sa = m.set_instrument(2, SpectrumAnalyzer)

    connections = [
        dict(source="Input1", destination="Slot1InA"),
        dict(source="Input2", destination="Slot1InB"),
        dict(source="Slot1OutA", destination="Slot2InA"),
        dict(source="Slot1OutB", destination="Slot2InB")
    ]

    print(m.set_connections(connections=connections))

    # ===========================================
    #            SA configuration
    # ===========================================

    sa.set_span(1, 100e6)
    sa.set_rbw("Auto")

    data_sa = sa.get_data()
    # data["ch1"], data["ch2"] e data["frequency"]




except Exception as e:
    print("[ERROR] Multi-instrument failed")

finally:
    # Close outputs
    fra.disable_output(channel=1)
    fra.disable_output(channel=2)

    # Fecha API
    print("Fechando conexão API...")
    m.relinquish_ownership()
    print("Conexão encerrada com sucesso!")
    
