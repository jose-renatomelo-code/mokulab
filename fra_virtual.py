import matplotlib.pyplot as plt
import control as ctrl

# Exemplo: Filtro passa-baixas com Função de Transferência H(s) = 10 / (s^2 + 3s + 10)
num = [10]                  # Numerador
den = [1, 3, 10]            # Denominador
sistema = ctrl.tf(num, den)

# Plota o diagrama de Bode
plt.figure()
ctrl.bode_plot(sistema, dB=True, Phase=True, Plot=True)
plt.show()