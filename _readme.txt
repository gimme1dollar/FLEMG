[엑셀 파일 형식으로 정리한 내용 파일 공유 URL]
https://1drv.ms/x/s!AiJTKSbHxk7Zih2mGCHbA9-fEwU0?e=wmitMx

[필요한 프로그램 및 모듈]
Python
Anaconda
OpenBci-GUI
VSPE

tensorflow
numpy
pyserial
datetime
matplotlib
drawnow

[프로그램 구동 방식]
(0) Flex Sensor는 ...
(0) EMG Sensor는 VSPE로 가상 시리얼 포트 만든다음에 OpenBCI GUI 통해서 EMG데이터 보내는 방식

(1) anaconda prompt에 들어가서 py36 환경 activate
(2) Desktop/tmp_JYLee/FLEMG 디렉토리 들어가서 필요한 프로그램 소스코드 실행
(3) 끌 때 Ctrl+C keyboard interrupt 걸면 안전하게 꺼지면서 저장할 것들 저장하게끔 함