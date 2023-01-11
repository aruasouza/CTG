import numpy as np
import pandas as pd

def distribution(media,minimo,maximo,n):
    size = maximo - minimo
    size += size * 0.1
    dist = (np.random.rand(n) * size) + minimo
    return list(map(lambda x: x + (np.random.normal(0.5,0.17) * (media - x)),dist))

def gerar_cenarios(n,simulacao):
    # IPCA
    cenarios_ipca = distribution(simulacao.loc['medio','ipca'],simulacao.loc['min','ipca'],simulacao.loc['max','ipca'],n)
    np.random.shuffle(cenarios_ipca)
    # CDI
    cenarios_cdi = distribution(simulacao.loc['medio','cdi'],simulacao.loc['min','cdi'],simulacao.loc['max','cdi'],n)
    np.random.shuffle(cenarios_cdi)
    # Cambio
    cenarios_cambio = distribution(simulacao.loc['medio','cambio'],simulacao.loc['min','cambio'],simulacao.loc['max','cambio'],n)
    np.random.shuffle(cenarios_cambio)

    # Resposta
    return pd.DataFrame({'ipca':cenarios_ipca,'cdi':cenarios_cdi,'cambio':cenarios_cambio})

