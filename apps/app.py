from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from .preprocess.air import seoul_time_air_quality_data_last_six_hours
from .preprocess.weather import seoul_time_air_quality_data_yesterday
from .preprocess.prepro import preprocess_air_weather
import threading
import os
from dotenv import load_dotenv
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


load_dotenv()
API_KEY_AIR = os.getenv("API_KEY_AIR")
API_KEY_WEATHER = os.getenv("API_KEY_WEATHER")
app = Flask(__name__)
CORS(app)  # CORS 허용
file_path = os.path.join(os.path.dirname(__file__), 'static', 'yesterday_seoul_dust.csv')
CITY = None

def run_preprocess():
    """ 🔥 1시간마다 실행되는 작업 (air.py → weather.py → prepro.py 순서대로 실행) """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"🚀 {now} - Running scheduled preprocessing tasks...")

    print("⏳ 주기적인 작업 실행 중...")

    # air.py 실행
    seoul_time_air_quality_data_last_six_hours(api_key=API_KEY_AIR, file_path=file_path, city=CITY)
    print("✅ air.py 실행 완료.")

    # weather.py 실행
    seoul_time_air_quality_data_yesterday(api_key=API_KEY_WEATHER, file_path=file_path)  # weather.py에 정의된 함수 호출
    print("✅ weather.py 실행 완료.")

    # prepro.py 실행
    preprocess_air_weather(file_path=file_path)  # prepro.py에 정의된 함수 호출
    print("✅ prepro.py 실행 완료.")

def start_scheduler():
    print("📌 스케줄러 실행 시도...")
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(run_preprocess, 'cron', minute=30)  # XX시 10분마다 실행
    scheduler.start()
    print("✅ 스케줄러 시작됨.")


def model_load():
    global rf_pm10, scaler, column_order
    print("모델 로드 시작")
    # 모델 및 전처리 도구 로드
    rf_pm10 = joblib.load("model/rf_pm10_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    column_order = joblib.load("model/column_order.pkl")  # 컬럼 순서 유지
    print("모델 로드 완료")

# 시간 시계열 피처 생성 함수
def create_time_series_features(df, past_hours=6):
    df = df.copy()

    # 과거 6시간 데이터 생성 (shift 적용)
    past_features = []
    for i in range(1, past_hours + 1):
        shifted = df[['pm10', 'pm25', 'wd', 'ws', 'ta', 'td', 'hm', 'rn', 'sd_tot', 'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa']].shift(i)
        shifted.columns = [f'{col}_lag{i}' for col in shifted.columns]
        past_features.append(shifted)

    df_transformed = pd.concat([df] + past_features, axis=1)
    print("시계열 부분 헤드")
    print(df_transformed)

    # NaN 값이 있는 행 제거 (최소 past_hours 만큼의 데이터가 필요)
    df_transformed.dropna(inplace=True)

    return df_transformed



@app.route("/dust/hour", methods=["POST"])
def run_model():
    try:
        global CITY  # 전역 변수 사용

        # 요청 본문에서 JSON 데이터 받기
        data = request.get_json()
        city = data.get("city")

        if CITY is None or CITY != city:  # 논리 연산자 수정
            CITY = city
            print(f"🔄 도시 변경됨: {CITY}, 전처리 실행")
            run_preprocess()  
    

        print("📌 데이터 로드 시작")

        # CSV 파일 로드
        df = pd.read_csv("apps/static/yesterday_seoul_dust.csv")
        if df.columns.size<=10:
            run_preprocess()  

        print("✅ 데이터 로드 완료. 컬럼 목록:", df.columns.tolist())

        # 1️⃣ 컬럼 정렬
        df_sorted = df.sort_values(by=['year', 'month', 'day', 'hour'])
        df_sorted = df_sorted[column_order]
        print("✅ 컬럼 정렬 완료. 정렬된 컬럼 목록:", df_sorted.columns.tolist())

        # 2️⃣ 요일(week) 컬럼 원핫 인코딩
        df_encoded = pd.get_dummies(df_sorted, columns=['week'], drop_first=False)
        print("✅ 원핫 인코딩 완료. 인코딩된 컬럼 목록:", df_encoded.columns.tolist())

        # 3️⃣ 학습 데이터에서 사용한 모든 컬럼을 보장하기 위해 컬럼 정리
        week_columns = [f'week_{i}' for i in range(7)] 
        expected_columns = [col for col in column_order if col != 'week'] + week_columns  
        missing_cols = set(expected_columns) - set(df_encoded.columns)

        # 부족한 컬럼을 0으로 추가
        for col in missing_cols:
            df_encoded[col] = False
        print(f"✅ 부족한 컬럼 {len(missing_cols)}개 추가 완료.")
        df_encoded = df_encoded[expected_columns]
        print(df_encoded.head())

        # 4️⃣ 정규화 (MinMaxScaler)
        cols_to_scale = [col for col in df_encoded.columns if col not in ['pm10', 'pm25', "week"]]
        df_encoded[cols_to_scale] = scaler.transform(df_encoded[cols_to_scale])
        print("✅ 정규화 완료.", df_encoded.columns.tolist())
        print(df_encoded.head())

        # 6️⃣ 시계열 데이터 변환 (과거 6시간, 미래 12시간 예측)
        df_transformed = create_time_series_features(df_encoded)
        print("✅ 시계열 변환 완료. 데이터 크기:", df_transformed.shape)

        if df_transformed.empty:
            print("⚠️ 변환된 데이터가 없습니다. 예측 불가.")
            return jsonify({"error": "Not enough past data for prediction."})

        # 7️⃣ 예측을 위한 X, y 준비
        X = df_transformed
        print("✅ 예측 데이터 준비 완료. 입력 데이터 크기:", X.shape)

        # 8️⃣ 모델 예측
        print("⏳ 모델 예측 중...")
        y_pm10_pred = rf_pm10.predict(X).tolist()
        print("✅ 모델 예측 완료. 예측값 예시:", y_pm10_pred[:5])

        response = {"pm10": y_pm10_pred}
        return jsonify(response)
    
    except Exception as e:
        print("🔴 오류 발생:", e)
        return jsonify({"error": str(e)})



def model_sw_load():
    global model_sw, scaler_sw, column_sw, season_sw, weekday_sw
    print("모델 로드 시작")
    
    
    model_sw = load_model('model/pm_lstm_sw01.keras')  # Keras 모델 파일 로드
    
    # pickle을 이용한 다른 파일들 로드
    with open('model/scaler_sw.pkl', 'rb') as f:
        scaler_sw = pickle.load(f)
        
    with open('model/columns_sw.pkl', 'rb') as f:
        column_sw = pickle.load(f)
        
    with open('model/season_encoded.pkl', 'rb') as f:
        season_sw = pickle.load(f)
        
    with open('model/week_encoded.pkl', 'rb') as f:
        weekday_sw = pickle.load(f)
        


    print("✅ 모델 및 전처리 도구 로드 완료.")


# 계절 컬럼 추가 및 원핫 인코딩 함수
def season_col(df):
    df['season'] = np.select(
        [
            (df['month'].isin([3, 4, 5])),  # 봄 (3, 4, 5월)
            (df['month'].isin([6, 7, 8])),  # 여름 (6, 7, 8월)
            (df['month'].isin([9, 10, 11])),  # 가을 (9, 10, 11월)
            (df['month'].isin([12, 1, 2]))  # 겨울 (12, 1, 2월)
        ],
        [1, 2, 3, 4],  # 각각 봄, 여름, 가을, 겨울을 1, 2, 3, 4로 할당
        default=0  # 기본값은 0 (예외 처리)
    )
    
    # 계절 원핫 인코딩
    season_encoded = pd.get_dummies(df['season'], prefix='season', drop_first=False)
    # df = pd.concat([df, season_encoded], axis=1)

    return df, season_encoded


# 요일 컬럼 추가 및 원핫 인코딩 함수
def dayOfTheWeek_col(df):
    df['weekend'] = df['week'].isin([5, 6]).astype(int)  # 주말
    df['weekday'] = df['week'].isin([0, 1, 2, 3, 4]).astype(int)  # 평일

    # 요일 원핫 인코딩
    week_encoded = pd.get_dummies(df['week'], prefix='week', drop_first=False)  # 요일 원핫 인코딩
    # df = pd.concat([df, week_encoded], axis=1)
    
    return df, week_encoded



# 데이터 전처리 및 예측 함수
@app.route("/dust/model_sw", methods=["POST"])
def run_model_sw():
    try:
        print("📌 데이터 로드 시작")

        # CSV 파일 로드
        df = pd.read_csv("apps/static/yesterday_seoul_dust.csv")
        df = df.head(1)
        print(df)
        print("✅ 데이터 로드 완료.")
        
        
        column_order = df.columns

        # 1️⃣ 계절(season) 컬럼 추가 및 원핫 인코딩
        df, season_encoded = season_col(df)
        print("✅ 계절(season) 컬럼 및 원핫 인코딩 완료.")

        # 2️⃣ 요일(week) 컬럼 원핫 인코딩
        df, week_encoded = dayOfTheWeek_col(df)
        print("✅ 요일(week) 컬럼 및 원핫 인코딩 완료.")
        
        # 3️⃣ df_sorted에 원핫 인코딩된 season과 week 컬럼 추가
        # df_sorted는 df를 기반으로 정렬 후 생성
        df = pd.concat([df, season_encoded, week_encoded], axis=1)  # season과 week 원핫 인코딩 컬럼 추가
        print("✅ 컬럼 정렬 완료.")
        
        print(df.columns)
        print(df.head())
        
        week_columns = [f'week_{i}' for i in range(7)] 
        season_columns = [f'season_{i}' for i in range(1,5)] 
        expected_columns = [col for col in column_order if col not in ['season', 'week']] + season_columns + week_columns
        missing_cols = set(expected_columns) - set(df.columns)
             
        # 4️⃣ 부족한 컬럼을 0으로 추가
        for col in missing_cols:
            df[col] = False  # 부족한 컬럼을 0으로 추가
        print(f"✅ 부족한 컬럼 {len(missing_cols)}개 추가 완료.")
        
        df = df.sort_values(by=['year', 'month', 'day', 'hour'])  # 시간 순 정렬
        print(f"week 컬럼 확인",df.columns)
        
        # 예측할 컬럼들 (출력값 y)
        output_columns = ['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']
        
        # 입력 컬럼들 (특성 X)
        input_columns = [col for col in df.columns if col not in output_columns]
        
        # 6️⃣ 예측을 위한 X 준비
        # X와 y 데이터 분리
        X = df[input_columns]  # 입력 특성 데이터
        # y = df[output_columns]  # 출력 예측 대상 데이터
        print("✅ 예측 데이터 준비 완료. 입력 데이터 크기:", X.shape)

                
        df = df[column_sw]  # 모델에서 사용한 컬럼 순서대로 정렬
        print("✅ 컬럼 순서 정렬 완료. 정렬된 컬럼 목록:", df.columns.tolist())
        
        # 0으로 초기화된 컬럼들 추가
        columns_to_add = ['pm10', 'pm25', 'no2', 'o3', 'co', 'so2', 'season', 'week', 'weekday', 'weekend']
        for col in columns_to_add:
            df[col] = 0  # 컬럼을 추가하고 값은 0으로 설정
            
        column_order = [
            'lat', 'lon', 'year', 'month', 'day', 'hour', 'week', 'no2', 'o3', 'co',
            'so2', 'pm10', 'pm25', 'wd', 'ws', 'ta', 'td', 'hm', 'rn', 'sd_tot',
            'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa', 'season', 'season_1',
            'season_2', 'season_3', 'season_4', 'weekend', 'weekday', 'week_0',
            'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6'
        ]

        # 데이터프레임을 주어진 순서대로 정렬
        df = df[column_order]
        print('개수체크')
        print(df.shape)
    
        # 5️⃣ 정규화 (StandardScaler)
        cols_to_scale = df.columns
        df[cols_to_scale] = scaler_sw.transform(df[cols_to_scale])
        print("✅ 정규화 완료.", df.columns.tolist())
        
        cols = ['lat', 'lon', 'year', 'month', 'day', 'hour', 'week_0', 'week_1',
       'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'season_1',
       'season_2', 'season_3', 'season_4', 'wd', 'ws', 'ta', 'td', 'hm', 'rn',
       'sd_tot', 'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa']
        
        df = df[cols]
        X = []
        
        for i in range(24):
            X.append(df.values)

        X = np.array(X)
             

        # 7️⃣ 모델 예측
        print("⏳ 모델 예측 중...")
        y_pred = model_sw.predict(X)  # 모델로 예측
        
        print(y_pred[0][0])
        
        if len(y_pred.shape) == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[2])  # (1, 24, 6) -> (24, 6)
        
        
        predicted_values = y_pred[0]  # 예시: 예측된 값, shape (6,)

        # 41개의 특성에 맞게 0으로 채운 배열 생성
        result = np.zeros((1, len(column_order)))  # 1행, 41열의 배열

        # 예측된 값들이 들어갈 위치를 설정
        output_columns = ['no2', 'o3', 'co', 'so2', 'pm10', 'pm25']
        for i, col in enumerate(output_columns):
            result[0, column_order.index(col)] = predicted_values[i]
            
        print(result)
        print(result[0][7:13])

        # scaler로 역변환
        y_origin = scaler_sw.inverse_transform(result)

        # 결과 확인
        print("✅ 역변환 완료. 예측값 예시:", y_origin[0][7:13])
        
        result = y_origin[0][7:13]
        arr = []
        for r in result:
            arr.append(format(r, '.6f'))  # 소수점 6자리까지 10진수로 출력
        print(arr)

        # 8️⃣ 결과 반환
        response = {
                    "no2": arr[0], "o3":  arr[1],
                    "co":  arr[2], "so2":  arr[3], 
                    "pm10": arr[4], "pm25": arr[5]}  # 예측값 반환
        return jsonify(response)
    
    
    
    except Exception as e:
        print("🔴 오류 발생:", e)
        return jsonify({"error": str(e)}), 500



model_load()

model_sw_load()

start_scheduler()  # 스케줄러 시작


