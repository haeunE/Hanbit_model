import pandas as pd
import numpy as np


def preprocess_air_weather(file_path):
# CSV 파일 읽기
    df = pd.read_csv(file_path)
    # 불필요한 컬럼 드랍
    columns_to_drop = ['STN', 'GST_WD', 'GST_WS', 'GST_TM', 'PT', 'PR', 'PV',
                    'RN_DAY', 'RN_JUN', 'RN_INT', 'SD_HR3', 'SD_DAY', 'WC', 'WP', 'WW',
                    'CH_MIN', 'CT_TYPE', 'CT_TOP', 'CT_MID', 'CT_LOW', 'SS', 'ST', 'GD_5',
                    'GD_10', 'GD_20', 'GD_30', 'SEA', 'WH', 'BF', 'IR', 'IX']

    df = df.drop(columns=columns_to_drop)

    # 컬럼명을 모두 소문자로 바꿈
    df.columns = df.columns.str.lower()

    # 'datetime' 컬럼을 datetime 형식으로 변환
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 'datetime' 컬럼에서 연도, 월, 일, 시간, 주 추출
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['week'] = df['datetime'].dt.weekday  # 0~6 (월요일=0, 일요일=6)

    df = df.drop(columns=["datetime"])
    district_coords = {
        "종로구": {"lat": 37.5703777777777, "lon": 126.981641666666},
        "중구": {"lat": 37.5610027777777, "lon": 126.999641666666},
        "용산구": {"lat": 37.5360944444444, "lon": 126.967522222222},
        "성동구": {"lat": 37.5606111111111, "lon": 127.039},
        "광진구": {"lat": 37.5357388888888, "lon": 127.084533333333},
        "동대문구": {"lat": 37.571625, "lon": 127.042141666666},
        "중랑구": {"lat": 37.6038055555555, "lon": 127.094777777777},
        "성북구": {"lat": 37.5863833333333, "lon": 127.020333333333},
        "강북구": {"lat": 37.6369555555555, "lon": 127.027719444444},
        "도봉구": {"lat": 37.6658333333333, "lon": 127.049522222222},
        "노원구": {"lat": 37.6514611111111, "lon": 127.058388888888},
        "은평구": {"lat": 37.5999694444444, "lon": 126.931241666666},
        "서대문구": {"lat": 37.5763666666666, "lon": 126.938897222222},
        "마포구": {"lat": 37.5607055555555, "lon": 126.910530555555},
        "양천구": {"lat": 37.5142305555555, "lon": 126.868708333333},
        "강서구": {"lat": 37.5481555555555, "lon": 126.851675},
        "구로구": {"lat": 37.49265, "lon": 126.889597222222},
        "금천구": {"lat": 37.4491083333333, "lon": 126.904197222222},
        "영등포구": {"lat": 37.5236111111111, "lon": 126.898341666666},
        "동작구": {"lat": 37.5096555555555, "lon": 126.941575},
        "관악구": {"lat": 37.4753861111111, "lon": 126.953844444444},
        "서초구": {"lat": 37.4807861111111, "lon": 127.034811111111},
        "강남구": {"lat": 37.514575, "lon": 127.049555555555},
        "송파구": {"lat": 37.5117555555555, "lon": 127.107930555555},
        "강동구": {"lat": 37.5273666666666, "lon": 127.125863888888}
    }

    # msrste_nm에 따라 lat, lon 값을 매칭
    df['lat'] = df['msrste_nm'].map(lambda x: district_coords.get(x, {}).get('lat'))
    df['lon'] = df['msrste_nm'].map(lambda x: district_coords.get(x, {}).get('lon'))
    df = df.drop(columns=["msrste_nm"])
    # 결측치 처리
    # 1. 'wd' (풍향): 결측치 제외 평균값 대입
    df['wd'] = df['wd'].replace(-9.0, np.nan)  # -9.0을 NaN으로 대체
    df['wd'] = df['wd'].fillna(df['wd'].mean())

    # 2. 'ws' (풍속): 결측치 제외 평균값 대입
    df['ws'] = df['ws'].replace(-9.0, np.nan)
    df['ws'] = df['ws'].fillna(df['ws'].mean())

    # 3. 'hm' (상대습도): 결측치 제외 평균값 대입
    df['hm'] = df['hm'].replace(-9.0, np.nan)
    df['hm'] = df['hm'].fillna(df['hm'].mean())

    # 4. 'rn' (강수량): 결측치 0 대입
    df['rn'] = df['rn'].replace(-9.0, 0)

    # 5. 'sd_tot' (적설): 결측치 0 대입
    df['sd_tot'] = df['sd_tot'].replace(-9.0, 0)

    # 6. 'ca_tot' (전운량): 해당 월의 최소값 대입
    df['ca_tot'] = df['ca_tot'].replace(-9.0, np.nan)
    df['ca_tot'] = df.groupby(df['month'])['ca_tot'].transform(lambda x: x.fillna(x.min()))

    # 7. 'ca_mid' (중하층운량): 결측치 제외 평균값 대입
    df['ca_mid'] = df['ca_mid'].replace(-9.0, np.nan)
    df['ca_mid'] = df['ca_mid'].fillna(df['ca_mid'].mean())

    # 8. 'vs' (시정): 결측치 제외 평균값 대입
    df['vs'] = df['vs'].replace(-9.0, np.nan)
    df['vs'] = df['vs'].fillna(df['vs'].mean())

    # 9. 'si' (일사): 하절기, 동절기 구분하여 결측치 처리
    df['si'] = df['si'].replace(-9.0, np.nan)

    # 하절기 (21~06월): 결측치 제외 평균값 대입
    df['si'] = df.apply(lambda row: row['si'] if (row['month'] < 7 or row['month']> 10) else row['si'], axis=1)
    df['si'] = df.groupby(df['month'])['si'].transform(lambda x: x.fillna(x.mean()))

    # 동절기 (19~08월): 낮시간대는 평균값, 밤은 0 대입
    df['si'] = df.apply(lambda row: 0 if (row['hour'] >= 19 or row['hour'] <= 8) else row['si'], axis=1)

    # 10. 'ps' (해면기압): 결측치 제외 평균값 대입
    df['ps'] = df['ps'].replace(-9.0, np.nan)
    df['ps'] = df['ps'].fillna(df['ps'].mean())

    # 11. 'pa' (현지기압): 결측치 제외 평균값 대입
    df['pa'] = df['pa'].replace(-9.0, np.nan)
    df['pa'] = df['pa'].fillna(df['pa'].mean())

    # 결측치 처리 후 결과 확인
    print(df.head())
    df.to_csv(file_path, index=False, encoding='utf-8-sig')