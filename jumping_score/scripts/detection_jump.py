# Este script se encarga de detectar saltos en las secuencias de movimiento de las personas
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import find_peaks, savgol_filter

BASE = os.getenv("DATA_DIR", "../csv")  # valor por defecto dentro del contenedor
CSV = os.path.join(BASE, "A15_-_skip_to_stand_stageii_xz_openpose.csv")


def is_jumping(sequence, threshold=0.1, prominence=0.08, distance=10, separar_saltos=False):
    """
    Detecta si una secuencia de keypoints indica un salto significativo usando find_peaks robusto.
    """
    if len(sequence) < 2:
        return False, None

    trajectory = []
    for seq in sequence:
        # Centro de masa de las columnas impares (por ejemplo, coordenada Y)
        odd_columns = seq[1::2]
        center_of_mass = float(np.mean(odd_columns))
        trajectory.append(center_of_mass)

    trajectory = np.array(trajectory)

    # Suavizado para reducir el ruido
    if len(trajectory) > 7:
        trajectory_smooth = savgol_filter(trajectory, window_length=7, polyorder=2)
    else:
        trajectory_smooth = trajectory

    # Detectar máximos y mínimos significativos
    peaks, prop_peaks = find_peaks(
        trajectory_smooth, prominence=prominence, distance=distance
    )
    valleys, prop_valleys = find_peaks(
        -trajectory_smooth, prominence=prominence, distance=distance
    )

    print("Índices de máximos:", peaks)
    print("Índices de mínimos:", valleys)

    #plot_trajectory(trajectory_smooth, peaks, valleys)
    detected = False
    if len(peaks) > 0 and len(valleys) > 0:
        max_val = np.max(trajectory_smooth[peaks])
        min_val = np.min(trajectory_smooth[valleys])
        print(f"Altura máxima detectada: {max_val:.3f}, mínima: {min_val:.3f}")

        detected = bool(abs((max_val - min_val)) > threshold)

    if separar_saltos:
        # Separamos la secuencia por saltos, es decir, cada dos mínimos
        saltos = []
        if len(valleys) == 1 and len(peaks) == 1:
            # Se trata de tomar el salto simetricamente respecto al pico
            distance = valleys[0] - peaks[0] 
            prev_valley = peaks[0] - distance
            if distance > 0 and prev_valley >= 0:
                print ("Mínimo virtual detectado")
                saltos.append((sequence[prev_valley:valleys[0]]))

            return detected, saltos

        for i in range(len(valleys) - 1):
            if valleys[i + 1] - valleys[i] > 40:  # Si hay al menos 40 frame entre saltos
                print ("Salto de", valleys[i], "a", valleys[i + 1])
                saltos.append((sequence[valleys[i]:valleys[i + 1]]))
        return detected, saltos

    else:
        # Considera salto si hay al menos un máximo y un mínimo y la diferencia supera el umbral
        return detected, None

def plot_trajectory(trajectory, peaks=None, valleys=None):
    plt.figure(figsize=(10, 5))
    plt.plot(trajectory, label='Trayectoria (suavizada)')
    if peaks is not None and len(peaks) > 0:
        plt.plot(peaks, trajectory[peaks], "ro", label="Máximos")
    if valleys is not None and len(valleys) > 0:
        plt.plot(valleys, trajectory[valleys], "go", label="Mínimos")
    plt.title('Trayectoria de la persona')
    plt.xlabel('Frame')
    plt.ylabel('Centro de masa (Y)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("trayectoria.png")
    plt.show()

    
if __name__ == "__main__":
    # Ejemplo de trayectoria (reemplaza por tus datos reales)
    # Detectar máximos y mínimos usando find_peaks

    secuencia = []
    with open(CSV, 'r') as f:
        lector = csv.reader(f)
        next(lector)  # Saltamos la cabecera
        datos = list(lector)

    for fila in datos:
        frame = int(fila[0])
        keypoints_csv = np.array([
            float(x) if x.strip() != '' else 0.0
            for x in fila[1:]
        ])

        secuencia.append(keypoints_csv)

    is_jump, saltos = is_jumping(secuencia, separar_saltos=True)
    print(f"¿La persona está saltando? {'Sí' if is_jump else 'No'}")
    print ("Numero de saltos", len(saltos))
    for i in saltos:
        print("Salto detectado:", len(i), type(i), saltos.shape())
