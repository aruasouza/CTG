import numpy as np
import pandas as pd
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from azure.datalake.store import core, lib, multithread

class NoRecordsError(Exception):
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

try:
    multithread.ADLDownloader(adlsFileSystemClient, lpath='records.csv', 
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)

except FileNotFoundError:
    raise NoRecordsError('Não foi possível recuperar um arquivo "records" do datalake')

records = pd.read_csv('records.csv')

inflacao = records.loc[[x.find('INFLACAO') != -1 for x in records['file_name']]]
juros = records.loc[[x.find('JUROS') != -1 for x in records['file_name']]]
cambio = records.loc[[x.find('CAMBIO') != -1 for x in records['file_name']]]


try:
    inflacao_name = inflacao.values.ravel()[-1]
    multithread.ADLDownloader(adlsFileSystemClient, lpath=inflacao_name, 
        rpath=f'DataLakeRiscoECompliance/PrevisionData/Variables/INFLACAO/AI/{inflacao_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    df_inflacao = pd.read_csv(inflacao_name)
except IndexError:
    print('Não existem simulações de inflação')

try:
    juros_name = juros.values.ravel()[-1]
    multithread.ADLDownloader(adlsFileSystemClient, lpath=juros_name, 
        rpath=f'DataLakeRiscoECompliance/PrevisionData/Variables/JUROS/AI/{juros_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    df_juros = pd.read_csv(juros_name)
except IndexError:
    print('Não existem simulações de juros')

try:
    cambio_name = cambio.values.ravel()[-1]
    multithread.ADLDownloader(adlsFileSystemClient, lpath=cambio_name, 
        rpath=f'DataLakeRiscoECompliance/PrevisionData/Variables/CAMBIO/AI/{cambio_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    df_cambio = pd.read_csv(cambio_name)
except IndexError:
    print('Não existem simulações de câmbio')

def distribution(media,std,minimo,maximo,n):
    dist = list(np.random.normal(media,std,n))
    for i,valor in enumerate(dist):
        if not (minimo <= valor <= maximo):
            dist[i] = (random.random() * (maximo - minimo)) + minimo
    return dist

def simulate_inflacao(mes,ano):
    data = f'{ano}-{mes}'
    linha = df_inflacao.loc[df_inflacao['date'] == data]
    media = linha['prediction'].iloc[0]
    maximo = linha['superior'].iloc[0]
    minimo = linha['inferior'].iloc[0]
    std = df_inflacao.loc[0,'std']
    simulation = distribution(media,std,minimo,maximo,1000)
    return simulation

def simulate_juros(mes,ano):
    data = f'{ano}-{mes}'
    linha = df_juros.loc[df_inflacao['date'] == data]
    media = linha['prediction'].iloc[0]
    maximo = linha['superior'].iloc[0]
    minimo = linha['inferior'].iloc[0]
    std = df_juros.loc[0,'std']
    simulation = distribution(media,std,minimo,maximo,1000)
    return simulation

def simulate_cambio(mes,ano):
    data = f'{ano}-{mes}'
    linha = df_cambio.loc[df_inflacao['date'] == data]
    media = linha['prediction'].iloc[0]
    maximo = linha['superior'].iloc[0]
    minimo = linha['inferior'].iloc[0]
    std = df_cambio.loc[0,'std']
    simulation = distribution(media,std,minimo,maximo,1000)
    return simulation