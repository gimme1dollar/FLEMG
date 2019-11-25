from modules import Encoder, Processor

# Instantiation
flag = True
MetaInfo = Encoder.encoder()
Reader = Processor.preprocessor(MetaInfo)

# Test :: Load
Reader.load('./data/full.csv')
print(f"./data/full.csv raw data length {len(Reader.data)}")
Reader.preprocess()
print(f"./data/full.csv processed data length {len(Reader.data)}")
print(f"processed data example\n {Reader.data[0]}")

# Test :: Save
Reader.save('./data/test-csv_save.csv')
print("test.csv saved")
## Check saved data is same with loaded data
Checker = Processor.preprocessor(MetaInfo)
Checker.load('./data/test-csv_save.csv')
Checker.preprocess()
if len(Checker.index) != len(Reader.index):
    flag = False
for i in range( len(Checker.data) ):
   for j in range( len(Checker.data[i])) :
       for k in range( len(Checker.data[i][j])):
           if(Checker.data[i][j][k] != Reader.data[i][j][k]):
                flag = False
print(f"Assert : {flag}")