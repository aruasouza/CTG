import tkinter as tk
from tkinter import filedialog
from azure.datalake.store import core, lib, multithread
import pandas as pd
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


# Capturando o caminho do arquivo csv
def captura_arquivo ():
    # Pedindo o input do nome da previsão
    Risco=input("Qual o tipo de previsão? (CAMBIO,INFLACAO,JUROS)")
    # Escolha do arquivo dentro do computador
    root = tk.Tk()
    root.withdraw()
    print("Escolha o arquivo para upload")
    file_path = filedialog.askopenfilename()
    # Capturando o arquivo e renomeando para o padrão
    output = pd.read_csv(file_path)
    time = datetime.now()
    time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
    file_name = f'{Risco}_{time_str}.csv'
    # Colocando o output para csv e encaminhando-o para o data lake
    output.to_csv('output/' + file_name)
    upload_file_to_directory(file_name,f'DataLakeRiscoECompliance/PrevisionData/Variables/{Risco}/AI')
    log = pd.read_csv(logfile_name)
    log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
    log.to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)    
    records = pd.read_csv('records.csv')
    records = pd.concat([records,pd.DataFrame({'file_name':[file_name]})])
    records.to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)    