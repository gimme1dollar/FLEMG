[엑셀 파일 형식으로 정리한 내용 파일 공유 URL]
https://1drv.ms/x/s!AiJTKSbHxk7Zih2mGCHbA9-fEwU0?e=EzsuXN

[프로그램 모듈]
+ 프로그램
- Python
- Anaconda : 가상 환경에서 tensorflow 등 돌리기 위함
- OpenBci-GUI
- VSPE

+ Python 모듈 
-tensorflow
-numpy
-pyserial
-datetime
-matplotlib
-drawnow

+ Custom 모듈
- Sensor : Serial Port를 열어서 데이터를 받아 Queue에 저장하는 모듈
- Encoder : Queue에 쌓인 데이터 중 의미 있는 데이터만 뽑아내는 모듈. 사용할 데이터들의 형식 (data dimension 등) 정보를 갖고 있음. 
- Processor.Preprocessor : 뉴럴네트워크를 사용하기 위해 적절한 형태의 데이터로 만들어주는 preprocessor. Encoder를 생성자 인자로 받음.
- Processor.Network : LSTM 뉴럴네트워크. Tensorflow를 활용하여, placeholder 구축 및 트레이닝/테스트 진행.
- Analysis : garph plotting (예정. 현재는 그냥 모듈 없이 custom 프로그램 상에서 graph plotting 하고 있음.)
- Actor : 의수를 제어할 때 사용할 모듈 (예정) 


++ Custom 프로그램 (P로 시작하는 것들은 실제 실험 과정에 쓰일 program, T로 시작하는 것들은 test용도)
+ Data Collecting 프로그램
- P_250Hz_collect.py : Encoding과정 없이 250Hz로 받아오는 센싱하는 데이터를 1초마다 (250 * 50, 250 * 33) Bytes씩 받아와서 txt 형태로 저장하는 프로그램
- P_collect_raw.py : Encoding과정 없이 250Hz로 받아오는 센싱하는 데이터를 while문에서 (50, 65) Bytes씩 받아와서 csv 형태로 저장하는 프로그램
- P_data_collect.py : 센서로 부터 데이터를 받아와 트레이닝 시킬 데이터를 모으는 프로그램. **VSPE로 가상 시리얼 포트 만든다음에 OpenBCI GUI 통해서 EMG데이터 보내는 방식으로 EMG 센서 데이터를 받아와야함.** Encoding 진행하여, preprocess하기 쉽게 csv형태로 저장.

+ Encoding 프로그램
- P_encode_250Hz.py : P_250Hz_collect.py를 통해 저장한 txt파일 내부의 데이터들을 encode하여, 각 라인에 한 데이터 유닛이 있도록 하는 프로그램 (미완).

+ LSTM Network training 프로그램
- P_infer_csv.py : csv파일 형식으로 된 데이터를 읽어서 LSTM 네트워크를 트레이닝 시키는 프로그램. 하이퍼파라미터는 프로그램 내부 코드를 수정하여 변경해야함.

+ Real-time Simulation 프로그램
- P_realtime_simulation.py : 이미 학습된 네트워크 모델을 불러와서, 실시간으로 받아오는 데이터를 feed하여 prediction 진행, 실시간으로 받아온 데이터와 예측 정보를 그래프로 그려줌.

[프로그램 구동 방식]
(0) 하드웨어 환경 세팅
- Flex Sensor는 Arduino를 이용해 시리얼 통신 방식으로 FLEX데이터 보내는 방식
- EMG Sensor는 (1) VSPE로 가상 시리얼 포트 만든다음에 OpenBCI GUI 통해서 EMG데이터 보내는 방식 혹은 (2) OpenBCI Cyton 보드 내부의 FW로 부터 Serial Communication으로 받아오는 방식 (default 프로그램에서는 'b'를 Serial로 보내면 streaming 시작, FW변경도 arduino IDE 이용하여 가능)
(1) 소프트웨어 환경 세팅
- anaconda prompt에 들어가서 py36 환경 activate
(2) 소프트웨어 구동
- Desktop/tmp_JYLee/FLEMG 디렉토리 들어가서 필요한 프로그램 소스코드 실행
(3) 소프트웨어 종료
- cmd창에서 Ctrl+C keyboard interrupt 걸면 안전하게 꺼지면서 저장할 것들 저장하게끔 함
