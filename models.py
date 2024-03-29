import pandas as pd
import numpy as np
from bcb import sgs,currency
from fredapi import Fred
from sklearn.metrics import mean_squared_error
from datetime import date,datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler
from sklearn.linear_model import LinearRegression
import math
from retry import retry
from darts.models import NBEATSModel
from bcb import Expectativas
from sklearn.ensemble import RandomForestRegressor

# Função que converte a variação mensal do IPCA em IPCA absoluto (A série de ipca deve começar em janeiro de 2000)
def absolute(serie):
    valor_atual = 1598.41
    yield valor_atual
    for valor in serie[1:]:
        valor = valor / 100
        valor_atual += (valor_atual * valor)
        yield valor_atual

# Função que puxa as séries temporais usadas pra prever o IPCA
@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_ipca(start_date):
    dados = {'selic':432,'emprego':28763,'producao':21859,'comercio':1455,'energia':1406,'IPCA_change':433}
    try:
        dataframe = sgs.get(dados,start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    dataframe = dataframe.resample('m').mean()
    dataframe['indice'] = [valor for valor in absolute(dataframe['IPCA_change'].values)]
    del(dataframe['IPCA_change'])
    dataframe = dataframe.dropna()
    return dataframe

# Função que separa os dados em treino e teste
def train_test_split(xdata,ydata,horizonte):
    meses = horizonte * 12
    y_train = ydata.iloc[:-meses]
    x_train = xdata.iloc[-meses - len(y_train):-meses]
    return x_train,y_train

# Função que puxa os dados usados pra prever o câmbio
@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_cambio(start_date):
    dados = {'selic':432,'ipca':13522,'pib':1208,'emprego':28763}
    try:
        dataframe = sgs.get(dados,start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    cy = currency.get('USD', start = start_date,end = str(date.today()))
    dataframe['cambio'] = cy['USD']
    try:
        api_key = '5beeb88b7a5cdd7d4fd8b976e138b52e'
        fred = Fred(api_key = api_key)
        gdp_eua = fred.get_series(series_id = 'GDPC1',observation_start = start_date)
        cpi = fred.get_series(series_id = 'USACPIALLMINMEI',observation_start = f'{int(start_date[:4]) - 1}{start_date[4:]}')
        inflacion_rate = pd.Series(data = (cpi.values[12:] - cpi.values[:-12]) / cpi.values[:-12],index = cpi.iloc[12:].index)
        employment = fred.get_series(series_id = 'CE16OV',observation_start = start_date)
        interest = fred.get_series(series_id = 'INTDSRUSM193N',observation_start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Federal Reserve')
    dataframe['pib_eua'] = gdp_eua
    dataframe['cpi'] = inflacion_rate
    dataframe['employment'] = employment
    dataframe['interest_rates'] = interest
    dataframe = dataframe.fillna(method = 'ffill')
    dataframe = dataframe.resample('m').mean()
    return dataframe.iloc[:-2]
    
# Função para realizar a captura de dados para cálculo da taxa Selic
@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_selic(start_date):
    dados = {'selic':4391}
    try:
        dataframe = sgs.get(dados,start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    dataframe = dataframe.resample('m').mean()
    return dataframe.iloc[:-1]

# Classe utilizada para criar o modelo LSTM (IPCA)
class NBEATS:
    def __init__(self,main_serie,extra_series):
        self.last = main_serie.values[-1]
        self.data = TimeSeries.from_dataframe(main_serie)
        self.extra_data = TimeSeries.from_dataframe(extra_series)
        self.scaler_y = Scaler()
        self.transformed_data = self.scaler_y.fit_transform(self.data)
        self.scaler_x = Scaler()
        self.transformed_extra_data = self.scaler_x.fit_transform(self.extra_data)
    def fit(self,input_size,output_size):
        self.model_cov = NBEATSModel(
            input_chunk_length = input_size,
            output_chunk_length = output_size,
            n_epochs = 50
        )
        self.model_cov.fit(
            series = self.transformed_data,
            past_covariates = self.transformed_extra_data,
            verbose = False,
        )
        return self
    def predict(self,n):
        prediction = self.model_cov.predict(n = n,series = self.transformed_data, past_covariates = self.transformed_extra_data)
        converted_prediction = self.scaler_y.inverse_transform(prediction).values().ravel()
        difference = converted_prediction[0] - self.last
        prediction_final = converted_prediction - difference
        return prediction_final

class LSTM:
    def __init__(self,main_serie,extra_series):
        self.last = main_serie.values[-1]
        self.data = TimeSeries.from_dataframe(main_serie)
        self.extra_data = TimeSeries.from_dataframe(extra_series)
        self.scaler_y = Scaler()
        self.transformed_data = self.scaler_y.fit_transform(self.data)
        self.scaler_x = Scaler()
        self.transformed_extra_data = self.scaler_x.fit_transform(self.extra_data)
    def fit(self,input_size,output_size):
        self.model_cov = BlockRNNModel(
            model = "LSTM",
            input_chunk_length = input_size,
            output_chunk_length = output_size,
            n_epochs = 300,
        )
        self.model_cov.fit(
            series = self.transformed_data,
            past_covariates = self.transformed_extra_data,
            verbose = False,
        )
        return self
    def predict(self,n):
        prediction = self.model_cov.predict(n = n,series = self.transformed_data, past_covariates = self.transformed_extra_data)
        converted_prediction = self.scaler_y.inverse_transform(prediction).values().ravel()
        difference = converted_prediction[0] - self.last
        prediction_final = converted_prediction - difference
        return prediction_final

class Forest:
    def __init__(self,df):
        df = df.copy()
        self.last_date = df.index[-1]
        df['mes'] = df.index.month
        df['quarter'] = df.index.quarter
        ger,gf = df['Geração'].values,df['Garantia Física'].values
        x = df[['mes','quarter']].values
        self.model_ger = RandomForestRegressor(max_depth=10).fit(x,ger)
        self.model_gf = RandomForestRegressor(max_depth=10).fit(x,gf)
    def predict(self,n):
        date_range = pd.date_range(start = self.last_date + relativedelta(months = 1),periods = n,freq = 'MS')
        df = pd.DataFrame(index = date_range)
        df['mes'] = df.index.month
        df['quarter'] = df.index.quarter
        x_fut = df.values
        df['prediction'] = self.model_ger.predict(x_fut) / self.model_gf.predict(x_fut)
        return df['prediction'].values

# Função otilizada para aproximar a curva de câmbio
def simple_square(x,a):
    return (x ** 2) * a

def square(x,a,b,c):
    return ((x ** 2) * a) + (x * b) + c

def linear(x,a,b):
    return (x * a) + b

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def weight(points,expo):
    def ajust(i):
        return sigmoid(i/points) ** expo
    return ajust

def simple_model_predict(serie,projection_points):
    ma = serie.rolling(6).mean().dropna()
    values = ma.values
    last_dif = values[-1] - values[-2]
    line = LinearRegression().fit(np.arange(len(values)).reshape(-1,1),values).predict(np.arange(len(values),len(values) + projection_points).reshape(-1,1)) - values[-1]
    line_derivada = np.cumsum(np.array([last_dif] * projection_points))
    ajuster = weight(10,10)
    final = [(line_derivada[i] * (1 - ajuster(i))) + (line[i] * ajuster(i)) for i in range(projection_points)]
    prediction = final + values[-1]
    micro_diferenca = serie.values[-1] - prediction[0]
    prediction_final = np.array([prediction[i] + (micro_diferenca * (1 - (i / (len(prediction) - 1)))) for i in range(len(prediction))])
    return prediction_final

def get_expectations(anos = 0):
    data_inicial = str(datetime.today() - relativedelta(months = (1 + (anos * 12))))[:10]
    data_final = str(datetime.today() - relativedelta(months = (anos * 12)))[:10]
    em = Expectativas()
    ep = em.get_endpoint('ExpectativasMercadoTop5Anuais')
    ultima_data = ep.query().filter(ep.Indicador == 'Câmbio',ep.Data >= data_inicial,ep.Data <= data_final,ep.tipoCalculo == 'L').select(ep.Data).collect()['Data'].iloc[-1]
    df = ep.query().filter(ep.Indicador == 'Câmbio',ep.Data == ultima_data,ep.tipoCalculo == 'L').select(ep.DataReferencia,ep.Media).collect()
    df['DataReferencia'] = df['DataReferencia'].apply(int)
    return df.set_index('DataReferencia').rename({'Media':'expect'},axis = 1)

# Classe utilizada para criar o modelo de regressão + LSTM para o câmbio
class RegressionPlusLSTM:
    def __init__(self,target_data,extra_data,func):
        self.func = func
        self.target_data = target_data
        self.extra_data = extra_data
        self.values = target_data.values.ravel()
        self.x0 = len(target_data)
    
    def fit(self,input_size,output_size):
        # Regressão polinomial no tempo
        self.popt = curve_fit(self.func,list(range(self.x0)),self.values)[0]
        # Sazonalidade
        self.lstm = LSTM(self.target_data,self.extra_data).fit(input_size,output_size)
        return self

    def predict(self,n,peso):
        self.peso = peso
        trend_prediction = np.array([self.func(x,*self.popt) for x in range(self.x0,self.x0 + n)])
        secondary_prediction = self.lstm.predict(n)
        prediction_final = (trend_prediction * self.peso) + (secondary_prediction * (1 - self.peso))
        # Ajustes finais
        micro_diferenca = self.values[-1] - prediction_final[0]
        prediction_final = np.array([prediction_final[i] + (micro_diferenca * (1 - (i / (len(prediction_final) - 1)))) for i in range(len(prediction_final))])
        return prediction_final

# Função que prevê o IPCA
def predict_ipca(test = False,lags = None):
    # Obtendo os dados
    df = get_indicators_ipca('2000-01-01')
    ipca = df[['indice']].copy()
    df = df.drop(['indice'],axis = 1)
    # Treinando o modelo de IPCA
    if not test:
        lags = [5]
    results = {}
    for anos in lags:
        x_train,y_train = train_test_split(df,ipca,anos)
        model = RegressionPlusLSTM(y_train,x_train,square).fit(24,12 * anos)
        # Calculando o Erro
        prediction = model.predict(12 * anos,0.5)
        pred_df = ipca.copy()
        pred_df['prediction'] = [None for _ in range(len(pred_df) - len(prediction))] + list(prediction)
        results[anos] = pred_df
    if test:
        return results
    pred_df['res'] = ((pred_df['indice'] - pred_df['prediction']) / pred_df['indice']).apply(abs)
    pred = pred_df.dropna()
    std = mean_squared_error(pred['indice'],pred['prediction'],squared = False)
    res_max = pred['res'].max()
    # Treinando novamente o modelo e calculando o Forecast
    model = RegressionPlusLSTM(ipca,df,square).fit(24,12 * anos)
    prediction = model.predict(12 * anos,0.5)
    pred_df = pd.DataFrame({'prediction':prediction},
        index = pd.period_range(start = ipca.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
    pred_df['superior'] = [pred + (pred * res_max) for pred in prediction]
    pred_df['inferior'] = [pred - (pred * res_max) for pred in prediction]
    pred_df['std'] = std
    return pred_df

def predict_cambio(test = False,lags = None):
    regression_level = 0
    # Puxando os dados de câmbio
    df = get_indicators_cambio('2000-01-01')
    cambio = df[['cambio']].copy()
    df = df.drop(['cambio'],axis = 1)
    # Treinando o modelo de câmbio
    if not test:
        lags = [5]
    results = {}
    for anos in lags:
        expect = get_expectations(anos)
        x_train,y_train = train_test_split(df,cambio,anos)
        model = NBEATS(y_train,x_train).fit(24,12 * anos)
        # Calculando o Erro
        prediction = model.predict(12 * anos)
        prediction = pd.Series(prediction).rolling(6,1).mean().values
        pred_df = cambio.copy()
        pred_df['prediction'] = ([None] * (len(pred_df) - len(prediction))) + list(prediction)
        pred_df['ano'] = pred_df.index.year
        pred_df = pred_df.join(expect,on = 'ano')
        results[anos] = pred_df
    if test:
        return results
    pred_df['res'] = ((pred_df['cambio'] - pred_df['prediction']) / pred_df['cambio']).apply(abs)
    pred = pred_df.dropna()
    std = mean_squared_error(pred['cambio'],pred['prediction'],squared = False)
    res_max = pred['res'].max()
    # Treinando novamente o modelo e calculando o Forecast
    model = NBEATS(cambio,df,square).fit(72,12 * anos)
    prediction = model.predict(12 * anos,regression_level)
    prediction = pd.Series(prediction).rolling(6,1).mean().values
    pred_df = pd.DataFrame({'prediction':prediction},
        index = pd.period_range(start = cambio.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
    pred_df['superior'] = [pred + (pred * res_max) for pred in prediction]
    pred_df['inferior'] = [pred - (pred * res_max) for pred in prediction]
    pred_df['std'] = std
    return pred_df

def predict_selic(test = False,lags = None):
    # Puxando e plotando os dados de IPCA
    df = get_indicators_selic('2000-01-01')
    selic = df['selic']
    # Treinando o modelo de SELIC
    if not test:
        lags = [5]
    results = {}
    for anos in lags:
        y_train = selic.iloc[:-12 * anos]
        # Calculando o Erro
        prediction = simple_model_predict(y_train,12 * anos)
        pred_df = df.copy()
        pred_df['prediction'] = [None for _ in range(len(pred_df) - len(prediction))] + list(prediction)
        results[anos] = pred_df
    if test:
        return results
    pred_df['res'] = (pred_df['selic'] - pred_df['prediction']).apply(abs)
    pred = pred_df.dropna()
    std = math.sqrt(np.square(np.subtract(pred['selic'].values,pred['prediction'].values)).mean())
    res_max = pred['res'].max()
    # Treinando novamente o modelo e calculando o Forecast
    prediction = simple_model_predict(selic,12 * anos)
    pred_df = pd.DataFrame({'prediction':prediction},
        index = pd.period_range(start = selic.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
    pred_df['superior'] = [pred + res_max for pred in prediction]
    pred_df['inferior'] = [pred - res_max for pred in prediction]
    pred_df['std'] = std
    return pred_df

def predict_gsf(test = False,lags = None):
    mapa_meses = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
    new_names = {'Garantia física no centro de gravidade MW médios (GFIS_2p,j)':'Garantia Física',
        'Geração no Centro de Gravidade - MW médios (Gp,j)':'Geração'}
    df = pd.read_excel('MRE - Geração x Garantia Física - Mês.xlsx').set_index('Unnamed: 0').T.rename(new_names,axis = 1)
    df.columns.name = None
    df = df.reset_index()
    df['index'] = pd.to_datetime(df['index'].apply(lambda x: str(mapa_meses[x[:3]]) + x[3] + '20' + x[4:]),format = '%m/%Y')
    df = df.set_index('index')
    df['gsf'] = df['Geração'] / df['Garantia Física']
    gsf = df[['gsf']]
    df = df.drop('gsf',axis = 1)
    # Treinando o modelo de SELIC
    if not test:
        lags = [3]
    results = {}
    for anos in lags:
        y_train = df.iloc[:-anos * 12]
        model = Forest(y_train)
        # Calculando o Erro
        prediction = model.predict(12 * anos)
        pred_df = gsf.copy()
        pred_df['prediction'] = [None for _ in range(len(pred_df) - len(prediction))] + list(prediction)
        results[anos] = pred_df
    if test:
        return results
    pred_df['res'] = (pred_df['gsf'] - pred_df['prediction']).apply(abs)
    pred = pred_df.dropna()
    std = math.sqrt(np.square(np.subtract(pred['gsf'].values,pred['prediction'].values)).mean())
    res_max = pred['res'].max()
    # Treinando novamente o modelo e calculando o Forecast
    model = Forest(df)
    prediction = model.predict(12 * anos)
    pred_df = pd.DataFrame({'prediction':prediction},
        index = pd.period_range(start = gsf.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
    pred_df['superior'] = [pred + res_max for pred in prediction]
    pred_df['inferior'] = [pred - res_max for pred in prediction]
    pred_df['std'] = std