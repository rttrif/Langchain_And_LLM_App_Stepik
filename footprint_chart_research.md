# Исследование: Построение графика футпринт (Footprint) на Python

**Дата:** 14 ноября 2025 г.

## Введение

Этот документ представляет собой глубокое исследование по теме построения графиков футпринт (также известных как кластерные графики или графики потока ордеров) с использованием Python. Цель — дать исчерпывающий ответ на вопросы: какие данные необходимы и как реализовать такую визуализацию.

---

## Часть 1: Необходимые данные (Data Requirements)

### 1.1. Что такое данные для футпринта?

Стандартные свечные данные (OHLC - Open, High, Low, Close) **категорически не подходят**. Они показывают только 4 ценовых уровня за период, но не дают информации о том, *какой объем был проторгован на каждой конкретной цене внутри свечи*.

**Футпринт — это визуализация потока ордеров (Order Flow).** Для его построения нужны **тиковые данные (tick data)** — информация о каждой сделке, прошедшей на рынке.

### 1.2. Атрибуты данных

Каждая запись о сделке (тик) должна содержать как минимум:
*   `timestamp`: Точное время сделки (вплоть до миллисекунд).
*   `price`: Цена, по которой прошла сделка.
*   `volume` (или `quantity`): Объем сделки.
*   `side` (или `aggressor`): Направление сделки. Это самый важный и не всегда доступный атрибут. Он показывает, кто был инициатором:
    *   **Покупатель (Buy/Ask):** Сделка прошла по цене Ask или выше (рыночная покупка).
    *   **Продавец (Sell/Bid):** Сделка прошла по цене Bid или ниже (рыночная продажа).

### 1.3. Источники данных

1.  **Криптовалютные биржи:**
    *   **Преимущества:** Наиболее доступный способ получить качественные тиковые данные бесплатно через публичные API (например, Binance, Bybit).
    *   **Пример (Binance API):** В данных о сделках (`aggTrades`) есть флаг `m` (`isBuyerMaker`). Если `m=False`, покупатель был "тейкером", то есть **агрессивным покупателем**. Это именно то, что нужно для определения `side`.

2.  **Фондовый/фьючерсный рынок:**
    *   **Сложности:** Обычно это платная услуга. Бесплатные данные сильно ограничены.
    *   **Примеры:** API брокеров (Interactive Brokers) или специализированные поставщики данных (IQFeed, TickData).

### 1.4. Определение агрессора

Если прямого флага `side` нет, его можно определить по "правилу тика", сравнивая цену сделки с лучшими ценами спроса/предложения (Bid/Ask) в тот же момент. Однако это требует доступа к данным Level 1 (стакан котировок), что усложняет задачу.

**Вывод:** Проще всего использовать источники, где сторона-агрессор уже определена.

---

## Часть 2: Реализация на Python

### 2.1. Логика агрегации данных

Основная задача — преобразовать плоский список сделок в структурированный формат футпринта для каждой свечи.

**Процесс в Pandas:**
1.  **Ресемплинг:** Сгруппировать тики по временным интервалам (например, 5 минут) с помощью `df.resample('5T')`.
2.  **Агрегация:** Внутри каждой группы (свечи) сгруппировать данные по цене и стороне (`price`, `side`) и просуммировать объемы.
3.  **Формирование структуры:** Создать удобную структуру (например, словарь), содержащую OHLC и данные футпринта (`цена -> {объем продаж, объем покупок}`) для каждой свечи.

### 2.2. Библиотеки для визуализации

1.  **Plotly:** **Предпочтительный выбор.** Идеально подходит для создания интерактивных графиков. Логика заключается в отрисовке стандартного свечного графика (`go.Candlestick`), поверх которого с помощью `fig.add_annotation()` накладываются текстовые метки с объемами.
2.  **Matplotlib:** Максимально гибкий, но более трудоемкий вариант. Потребуется вручную рисовать каждый элемент: прямоугольник свечи, фитиль и все текстовые метки.
3.  **mplfinance:** Не подходит. Эта библиотека предназначена для быстрого создания *стандартных* графиков и не имеет функционала для добавления кастомных элементов внутрь свечей.

---

## Часть 3: Пошаговый пример кода (Proof-of-Concept)

Этот скрипт демонстрирует весь процесс: от генерации тестовых данных до построения интерактивного графика на Plotly.

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def generate_mock_ticks(num_ticks=5000, start_time=datetime(2023, 1, 1, 10, 0)):
    """Генерирует DataFrame с синтетическими тиковыми данными."""
    base_price = 100
    prices = []
    volumes = []
    sides = []
    timestamps = []

    current_time = start_time
    for i in range(num_ticks):
        # Симуляция движения цены
        price_movement = np.random.randn() * 0.1
        base_price += price_movement
        price = round(base_price, 2)
        
        # Симуляция объема и стороны
        volume = np.random.randint(1, 50)
        side = np.random.choice(['buy', 'sell'], p=[0.5 + price_movement * 2, 0.5 - price_movement * 2])
        
        # Время
        current_time += timedelta(milliseconds=np.random.randint(50, 500))
        
        prices.append(price)
        volumes.append(volume)
        sides.append(side)
        timestamps.append(current_time)

    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'side': sides
    })

def aggregate_ticks_to_footprint(ticks_df, freq='5T'):
    """Агрегирует тиковые данные в структуру футпринта."""
    if ticks_df.empty:
        return {}

    ticks_df = ticks_df.set_index('timestamp')
    resampler = ticks_df.resample(freq)
    
    footprint_data = {}

    for timestamp, group in resampler:
        if group.empty:
            continue

        # 1. Рассчитываем OHLC
        ohlc = {
            'open': group['price'].iloc[0],
            'high': group['price'].max(),
            'low': group['price'].min(),
            'close': group['price'].iloc[-1]
        }

        # 2. Агрегируем футпринт
        footprint = group.groupby(['price', 'side'])['volume'].sum().unstack(fill_value=0)
        
        # Убедимся, что оба столбца существуют
        if 'buy' not in footprint.columns:
            footprint['buy'] = 0
        if 'sell' not in footprint.columns:
            footprint['sell'] = 0
        
        footprint = footprint.rename(columns={'buy': 'ask_volume', 'sell': 'bid_volume'})
        
        # 3. Рассчитываем общие метрики
        total_bid_volume = footprint['bid_volume'].sum()
        total_ask_volume = footprint['ask_volume'].sum()
        total_volume = total_bid_volume + total_ask_volume
        delta = total_ask_volume - total_bid_volume

        footprint_data[timestamp] = {
            'ohlc': ohlc,
            'footprint': footprint,
            'total_volume': total_volume,
            'delta': delta
        }
        
    return footprint_data

def plot_footprint_chart(footprint_data):
    """Строит интерактивный футпринт-график на Plotly."""
    
    # Подготовка данных для свечного графика
    ohlc_df = pd.DataFrame({
        'timestamp': list(footprint_data.keys()),
        'open': [v['ohlc']['open'] for v in footprint_data.values()],
        'high': [v['ohlc']['high'] for v in footprint_data.values()],
        'low': [v['ohlc']['low'] for v in footprint_data.values()],
        'close': [v['ohlc']['close'] for v in footprint_data.values()],
        'delta': [v['delta'] for v in footprint_data.values()]
    })

    fig = go.Figure(data=go.Candlestick(
        x=ohlc_df['timestamp'],
        open=ohlc_df['open'],
        high=ohlc_df['high'],
        low=ohlc_df['low'],
        close=ohlc_df['close'],
        name='Candles',
        hovertext=[f"Delta: {d}" for d in ohlc_df['delta']]
    ))

    # Добавление аннотаций (текста футпринта)
    annotations = []
    for timestamp, data in footprint_data.items():
        footprint = data['footprint']
        for price, volumes in footprint.iterrows():
            bid_vol = int(volumes['bid_volume'])
            ask_vol = int(volumes['ask_volume'])
            
            # Не рисуем нулевые значения для чистоты
            if bid_vol == 0 and ask_vol == 0:
                continue

            annotations.append(go.layout.Annotation(
                x=timestamp,
                y=price,
                text=f"<b>{bid_vol} x {ask_vol}</b>",
                showarrow=False,
                font=dict(size=8, color="white"),
                align="center",
                bgcolor="rgba(0,0,0,0.5)", # Полупрозрачный фон для читаемости
                borderpad=0
            ))
            
    fig.update_layout(
        title="Интерактивный график футпринт (Footprint Chart)",
        xaxis_title="Время",
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        annotations=annotations,
        template="plotly_dark"
    )
    
    return fig

# --- Основной блок выполнения ---
if __name__ == '__main__':
    # 1. Генерируем тестовые данные
    mock_ticks = generate_mock_ticks(num_ticks=10000)
    
    # 2. Агрегируем данные в структуру футпринта
    # Используем частоту '15T' (15 минут) для более наглядных свечей
    aggregated_data = aggregate_ticks_to_footprint(mock_ticks, freq='15T')
    
    # 3. Строим и отображаем график
    if aggregated_data:
        footprint_fig = plot_footprint_chart(aggregated_data)
        footprint_fig.show()
    else:
        print("Недостаточно данных для построения графика.")

```

---

## Часть 4: Источники и полезные материалы

1.  **Plotly Documentation:**
    *   [Candlestick Charts](https://plotly.com/python/candlestick-charts/): Основа для построения свечного графика.
    *   [Text and Annotations](https://plotly.com/python/text-and-annotations/): Ключевая функциональность для добавления данных футпринта на график.
2.  **Pandas Documentation:**
    *   [Resampling](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling): Документация по группировке временных рядов.
    *   [Group By: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html): Основа для агрегации данных внутри каждой свечи.
3.  **Концептуальные материалы:**
    *   [Order Flow Trading: A Complete Guide (Axia Futures)](https://axiafutures.com/blog/order-flow-trading-a-complete-guide/): Статья, объясняющая концепцию потока ордеров, частью которой являются футпринт-графики.
    *   [Footprint Chart (Bookmap Wiki)](https://bookmap.com/wiki/Footprint_chart_and_advanced_volume_tools): Описание и примеры интерпретации футпринт-графиков.
