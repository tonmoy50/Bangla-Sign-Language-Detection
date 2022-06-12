from t_to_s import *


tra=TranslatorModel()
tra.t_to_s('হাই আমি একজন মানুষ')

print(tra.TextBlob_translator("hi how are you doing my friend",to_voice=True))
print(tra.google_translator("hi how are you doing my friend",to_voice=True))