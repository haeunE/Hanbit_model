import requests
import pandas as pd
from datetime import datetime, timedelta

def seoul_time_air_quality_data_last_six_hours(api_key, file_path, city):
  base_url = "http://openAPI.seoul.go.kr:8088/{}/json/TimeAverageAirQuality/1/1000/{}/{}"
  data_list = []

  # 현재 시간 및 6시간 전 시간 계산
  now = datetime.now()
  
  # 시간별로 데이터를 가져오기 위한 반복문
  end_time = now
  count = 0

  while count < 8 :
    time_str = end_time.strftime('%Y%m%d%H') + "00"  # 분을 00으로 설정
    # URL을 6시간 전부터 현재까지의 데이터로 요청
    url = base_url.format(api_key, time_str, city)

    try:
      response = requests.get(url)
      response.raise_for_status()  # Raise an HTTPError for bad responses

      json_data = response.json()

      if "TimeAverageAirQuality" in json_data and "row" in json_data["TimeAverageAirQuality"]:
        rows = json_data["TimeAverageAirQuality"]["row"]
        data_list.extend(rows)
      else:
        print(f"No data for {time_str}. Response: {json_data}")

    except requests.exceptions.JSONDecodeError:
      print(f"Failed to decode JSON for {time_str}. Response content: {response.text}")
    except requests.exceptions.RequestException as e:
      print(f"Request failed for {time_str}. Error: {e}")

    # 1시간씩 감소
    end_time -= timedelta(hours=1)
    count += 1

  if data_list:
    df = pd.DataFrame(data_list)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Data saved to {file_path}")
  else:
    print("No data fetched.")