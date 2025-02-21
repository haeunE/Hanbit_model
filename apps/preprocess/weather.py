import requests
import pandas as pd
from datetime import datetime, timedelta


# ✅ 연도별 데이터 가져오는 함수
def seoul_time_air_quality_data_yesterday(file_path,api_key):

    columns = [
        'YYMMDDHHMI', 'STN', 'WD', 'WS', 'GST_WD', 'GST_WS', 'GST_TM',
        'PA', 'PS', 'PT', 'PR', 'TA', 'TD', 'HM', 'PV','RN',
        'RN_DAY', 'RN_JUN', 'RN_INT', 'SD_HR3',
        'SD_DAY', 'SD_TOT', 'WC', 'WP', 'WW',
        'CA_TOT', 'CA_MID', 'CH_MIN', 'CT_TYPE',
        'CT_TOP', 'CT_MID', 'CT_LOW',
        'VS', 'SS', 'SI', 'ST',
        'TS', 'GD_5', 'GD_10', 'GD_20', 'GD_30',
        'SEA', 'WH', 'BF', 'IR', 'IX'
    ]
    data_list = []
    now = datetime.now()
    # 현재 시간에서 12시간 전을 계산
    twelve_hours_ago = now - timedelta(hours=8)
    
    # tm1은 12시간 전의 '시'로 설정
    tm1 = twelve_hours_ago.strftime('%Y%m%d%H') + "00"  # 12시간 전의 시각에서 분과 초는 '00'으로 설정
    tm2 = now.strftime('%Y%m%d%H') + "00"  # 같은 날 12시간 전 시간대의 마지막 분까지
    print(tm1,tm2)
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1={tm1}&tm2={tm2}&stn=108&authKey={api_key}"
    print(url)
    try:
        response = requests.get(url, timeout=30)  # 요청 타임아웃 60초 설정
        print(f"API 응답 코드: {response.status_code}")  # 응답 코드 확인

        if response.status_code != 200:
            print(f"❌ [API 요청 실패: {response.status_code}")
            return  # 오류 발생 시 함수 종료

        # ✅ 응답 데이터 줄 단위로 나누기
        lines = response.text.splitlines()

        # 3번째 줄부터 데이터 파싱 시작
        for line in lines[4:-1]:  # 4번째 줄부터 시작
            if line.strip() == '': continue
            parts = line.split()
            data_list.append(parts)
        print(data_list)
        if data_list:
            weather_df = pd.DataFrame(data_list, columns=columns)

            # Load weather data
            air_quality_df = pd.read_csv(file_path)

            # Convert time columns to datetime format for merging
            air_quality_df = air_quality_df.rename(columns={"MSRDT": "datetime"})
            weather_df = weather_df.rename(columns={"YYMMDDHHMI": "datetime"})
            # 'datetime' 컬럼을 datetime64로 변환
            air_quality_df['datetime'] = pd.to_datetime(air_quality_df['datetime'], format='%Y%m%d%H%M')
            weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], format='%Y%m%d%H%M')
            # Perform left join on datetime
            merged_df = pd.merge(air_quality_df, weather_df, on='datetime', how='inner')
            print(merged_df)
            merged_df = merged_df.head(7)
            merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"Merged data saved to {file_path}")
        else:
            print("No data fetched.")

        print(f"✅ [데이터 수집 완료! ")

    except requests.RequestException as e:
        print(f"❌ [API 요청 오류: {e}")


    print("🎉 모든 연도 원본 데이터 저장 완료!")