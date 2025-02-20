# 환경세팅
## 1. 가상환경설정
### 가상환경 생성
```
python -m venv venv
```
### 가상환경 활성화
```
venv\Scripts\activate
```
## 2. 의존성 설치
### 파일생성
```
pip install -r requirements.txt
```
### 설치
```
pip install flask
```
## 3. API 키 발급 및 사용 방법

### 1. 서울시 대기질 정보 API

서울시에서 제공하는 대기질 정보를 조회하기 위해 API 키를 발급받는 방법입니다.

API 키 발급 방법

서울 열린데이터 광장(https://data.seoul.go.kr/)에 접속합니다.

회원가입 또는 로그인 후, 서울시 대기환경 정보 페이지로 이동합니다.

"활용 신청" 버튼을 클릭하여 API 사용 신청을 합니다.

승인 후 "내 데이터"에서 발급된 API 키를 확인할 수 있습니다.
```.env
API_KEY_AIR=your_api_key
```

### 2. 기상청 실황 기상정보 API

기상청에서 제공하는 실황 기상정보를 조회하기 위한 API 키 발급 방법입니다.

API 키 발급 방법

기상자료개방포털에 접속합니다.

회원가입 또는 로그인 후, "마이페이지"에서 "API Key 신청"을 클릭합니다.

제공되는 서비스 목록에서 기상청 실황 기상정보 API를 선택하여 신청합니다.

승인 후 "마이페이지 > API Key 관리"에서 발급된 API 키를 확인할 수 있습니다.
```.env
API_KEY_WEATHER=your_api_key
```

## 4. 환경 변수 설정
### .env
```
FLASK_APP=apps.app
FLASK_DEBUG=False
API_KEY_AIR=your_api_key
API_KEY_WEATHER=your_api_key
