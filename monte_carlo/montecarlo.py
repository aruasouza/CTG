import numpy as np
import pandas as pd
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from azure.datalake.store import core, lib, multithread

class NoLogError(Exception):
    pass

tenant = '6e2475ac-18e8-4a6c-9ce5-20cace3064fc'
RESOURCE = 'https://datalake.azure.net/'
client_id = "0ed95623-a6d8-473e-86a7-a01009d77232"
client_secret = "NC~8Q~K~SRFfrd4yf9Ynk_YAaLwtxJST1k9S4b~O"
adlsAccountName = 'deepenctg'

adlCreds = lib.auth(tenant_id = tenant,
                client_secret = client_secret,
                client_id = client_id,
                resource = RESOURCE)

adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

# Pega o log do azure e sobescreve com novas saídas do log
def get_log(mes,ano):
    logfile_name = f'log_{mes}_{ano}.csv'
    try:
        multithread.ADLDownloader(adlsFileSystemClient, lpath=logfile_name, 
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
        log = pd.read_csv(logfile_name)
        files_list = log.loc[log['error'] == 'no errors']
    except FileNotFoundError:
        raise NoLogError('Não foi possível recuperar um log do datalake')

# Cria a simulação de monte carlo com base em uma distribuição normal
def distribution(media,std,minimo,maximo,n):
    dist = list(np.random.normal(media,std,n))
    for i,valor in enumerate(dist):
        if not (minimo <= valor <= maximo):
            dist[i] = (random.random() * (maximo - minimo)) + minimo
    return dist
