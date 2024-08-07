import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Carregar os dados
data = pd.read_csv('data/weather_2000.csv')

# Renomear colunas para facilitar o acesso
data.columns = [
    'date', 'time_utc', 'precipitation', 'pressure_station', 'pressure_max_prev', 'pressure_min_prev',
    'global_radiation', 'temperature', 'dew_point', 'max_temperature_prev', 'min_temperature_prev',
    'max_dew_point_prev', 'min_dew_point_prev', 'max_humidity_prev', 'min_humidity_prev',
    'humidity', 'wind_direction', 'max_wind_gust', 'wind_speed', 'station'
]

# Processar os dados
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data = data.dropna(subset=['precipitation'])

def get_hottest_day(year, year_compair = None, month=None):
    # Filtra os dados para o ano e o mês específicos, se o mês for fornecido
    if month != None:
        year_month_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]
    else:
        year_month_data = data[data['date'].dt.year == year]
        if year_compair != None:
            year_month_data_compair = data[data['date'].dt.year == year_compair]

    if len(year_month_data) == 0:
        return None

    # Encontra o dia mais quente
    hottest_day = year_month_data.loc[year_month_data['temperature'].idxmax()]

    if year_compair != None:
        if len(year_month_data_compair) == 0:
            return None
        hottest_day_compair = year_month_data_compair.loc[year_month_data_compair['temperature'].idxmax()]
        hottest_day['temperature'] = hottest_day['temperature'] - hottest_day_compair['temperature']

    return hottest_day['temperature']

def get_min_temperature(year, year_compair = None, month=None):
    # Filtra os dados para o ano e o mês específicos, se o mês for fornecido
    if month != None:
        year_month_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]
    else:
        year_month_data = data[data['date'].dt.year == year]
        if year_compair != None:
            year_month_data_compair = data[data['date'].dt.year == year_compair]

    if len(year_month_data) == 0:
        return None

    # Encontra a mínima temperatura
    min_temperature = year_month_data['temperature'].min()

    if year_compair != None:
        if len(year_month_data_compair) == 0:
            return None
        min_temperature_compair = year_month_data_compair['temperature'].min()
        min_temperature = min_temperature - min_temperature_compair

    return min_temperature

def get_average_temperature(year, year_compair = None, month=None):
    # Filtra os dados para o ano e o mês específicos, se o mês for fornecido
    if month != None:
        year_month_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]
    else:
        year_month_data = data[data['date'].dt.year == year]
        if year_compair != None:
            year_month_data_compair = data[data['date'].dt.year == year_compair]

    if len(year_month_data) == 0:
        return None
    
    mean = year_month_data['temperature'].mean()

    if year_compair != None:
        if len(year_month_data_compair) == 0:
            return None
        mean_compair = year_month_data_compair['temperature'].mean()
        mean = mean - mean_compair

    # Calcula a média de temperatura
    return mean


from dotenv import load_dotenv
import os

load_dotenv() 

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,
    max_tokens=100,
    timeout=10,
    max_retries=3
)

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

class RecordeMaximoTemperatura(BaseModel):
    """Busca a temperatura máxima para um ano e mês específicos"""
    year: int = Field(..., description="O ano em que o recorde de temperatura máxima foi registrado")
    year_compair: Optional[int] = Field(None, description="O ano em que o recorde de temperatura máxima é comparada")
    month: Optional[int] = Field(None, description="O mês em que o recorde de temperatura máxima foi registrado")

class RecordeMinimoTemperatura(BaseModel):
    """Busca a temperatura mínima para um ano e mês específicos"""
    year: int = Field(..., description="O ano em que o recorde de temperatura mínima foi registrado")
    year_compair: Optional[int] = Field(None, description="O ano em que o recorde de temperatura mínima é comparada")
    month: Optional[int] = Field(None, description="O mês em que o recorde de temperatura mínima foi registrado")

class MediaTemperatura(BaseModel):
    """Busca a temperatura média para um ano e mês específicos"""
    year: int = Field(..., description="O ano em que a média da temperatura foi calculada")
    year_compair: Optional[int] = Field(None, description="O ano em que a média da temperatura é comparada")
    month: Optional[int] = Field(None, description="O mês em que a média da temperatura foi calculada")

llm_with_tools = llm.bind_tools([RecordeMaximoTemperatura, RecordeMinimoTemperatura, MediaTemperatura])

messages = [
    ("system", ""),
]

while True:
  message = input('Você: ')
  if message == "exit()":
    break
  messages.append(HumanMessage(message))
  answered = False
  while not answered:
    response = llm_with_tools.invoke(messages)
    messages.append(AIMessage(response.content, tool_calls=response.tool_calls, invalid_tool_calls=response.invalid_tool_calls))
    if len(response.tool_calls) == 0:
        print('ChatGPT: ', response.content)
        answered = True
    else:
        for call in response.tool_calls:
            if call['name'] == 'RecordeMaximoTemperatura':
                call_response = get_hottest_day(**call['args'])
            elif call['name'] == 'RecordeMinimoTemperatura':
                call_response = get_min_temperature(**call['args'])
            elif call['name'] == 'MediaTemperatura':
                call_response = get_average_temperature(**call['args'])
            messages.append(ToolMessage(content = call_response if call_response != None else '?', name = call['name'], tool_call_id = call['id']))

