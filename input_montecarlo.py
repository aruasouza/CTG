import tkinter as tk
from tkinter import filedialog
from azure.datalake.store import core, lib, multithread
import pandas as pd
from datetime import datetime
import time
import os
import re

# Talvez tenha bugado porque não tem tkinter

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

today = datetime.now()
logfile_name = f'log_{today.month}_{today.year}.csv'

try:
    multithread.ADLDownloader(adlsFileSystemClient, lpath=logfile_name, 
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)

except FileNotFoundError:
    pd.DataFrame({'time':[],'output':[],'error':[]}).to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

try:
    multithread.ADLDownloader(adlsFileSystemClient, lpath='records.csv', 
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)

except FileNotFoundError:
    pd.DataFrame({'file_name':[],'origin':[]}).to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

def upload_file_to_directory(file_name,directory):
    multithread.ADLUploader(adlsFileSystemClient, lpath=file_name,
        rpath=f'{directory}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    time.sleep(1)
    os.remove('output/' + file_name)

# Capturando o caminho do arquivo csv
def captura_arquivo():
    # Pedindo o input do nome da previsão

    # Escolha do arquivo dentro do computador
    root = tk.Tk()
    var = tk.StringVar()

    def on_select():
        choice = var.get()
        # do something with the choice
        root.destroy()

    option1 = tk.Radiobutton(root, text="CAMBIO", variable=var, value="CAMBIO",command = on_select)
    option1.pack()
    option2 = tk.Radiobutton(root, text="INFLACAO", variable=var, value="INFLACAO",command = on_select)
    option2.pack()
    option3 = tk.Radiobutton(root, text="JUROS", variable=var, value="JUROS",command = on_select)
    option3.pack()
    root.mainloop()
    Risco = var.get()
    print("Escolha o arquivo para upload")
    file_path = filedialog.askopenfilename()
    root.withdraw()
    # Capturando o arquivo e renomeando para o padrão
    try:
        output = pd.read_csv(file_path)
    except Exception:
        print('O arquivo selecionado não é do formato correto.')
        return
    if len(output) > 72:
        print('O tamanho do arquivo excede o limite máximo')
        return
    if list(output.columns) != ['date', 'prediction', 'superior', 'inferior', 'std']:
        print('As colunas do arquivo devem ser (nessa ordem): date, prediction, superior, inferior, std')
        return
    date_sample = output.loc[0,'date'] 
    if not re.match("^\d{4}-\d{2}$", date_sample):
        print('As datas não estão no formato correto (YYYY-mm). Exemplo: 2020-04')
        return
    time = datetime.now()
    time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
    file_name = f'{Risco}_{time_str}.csv'
    # Colocando o output para csv e encaminhando-o para o data lake
    upload_file_to_directory(file_path,f'DataLakeRiscoECompliance/PrevisionData/Variables/{Risco}/Manual')
    log = pd.read_csv(logfile_name)
    log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
    log.to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)    
    records = pd.read_csv('records.csv')
    records = pd.concat([records,pd.DataFrame({'file_name':[file_name],'origin':['Manual']})])
    records.to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)