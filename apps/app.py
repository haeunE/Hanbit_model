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
        if df.columns.size<=8:
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


model_load()
print("🔹 스케줄러 시작 시도")
start_scheduler()



