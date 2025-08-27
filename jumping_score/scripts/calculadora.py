from detection_jump import is_jumping


def calcular_puntuacion(historial):

    
    '''
        Historial: LISTA de keypoints para el ID dado
    '''
    # Aquí podríamos procesar el historial de saltos de cada persona
    # Por ejemplo, podríamos imprimirlo o almacenarlo en una base de datos

    # Mandamos la trayectoria a la función que la separará por saltos
    _, saltos = is_jumping(historial, separar_saltos=True)

    # saltos es una lista de listas, donde cada elemento contiene un salto. 
    # Son estas secuencias de keypoints las que se tienen que pasar al modelo de puntuación
    print("Saltos", len(saltos))
    return  saltos