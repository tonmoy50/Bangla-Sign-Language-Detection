from gtts import gTTS
import os
from textblob import TextBlob
from googletrans import Translator
class TranslatorModel:
    # - *- coding: utf- 8 - *-
    #array = [' হাই', ' অমি' ,' মানুস' ,'ক্রুদ্ধ',' পায়খানা',' রঙ',' জরিমানা',' বন্ধু',' কিভাবে',' জীবন',' মত',' নাম',' কাজ']
    #text_bd=array[0]+array[7]+array[1]+array[9]
    #print(array[0][3])
    #print(text_bd)
    
    #print(os.getcwd)
    def TextBlob_translator( self, text,from_lang='en',to_lang='bn',to_voice=False):
        #print(text)
        translator_object=TextBlob(text)
        converted_text=translator_object.translate(from_lang=from_lang,to=to_lang)

        print(type(converted_text))
        x=str(converted_text)
        print(type(x))
        print(x)

        if(to_voice == True):
            tra=TranslatorModel()
            tra.t_to_s(text_t=x,lan=to_lang)   
        
        return(x)
    #print(translate_to_bangla("hi how are you"))
    def google_translator(self,text,from_lang='en',to_lang='bn',to_voice=False):
        T=Translator()
        translator_object= T.translate(text=text,src=from_lang,dest=to_lang)
        
        converted_text = translator_object.text
        if(to_voice == True):
            tra=TranslatorModel()
            tra.t_to_s(text_t=converted_text,lan=to_lang)  

        return converted_text
    #print(google_translator("hi how are you doing my friend",to_voice=True))
    #print(TextBlob_translator("hi how are you",to_voice=True))
    def t_to_s(self,text_t,lan='bn'):
            #print(text_bd)
            tts = gTTS(text=text_t,lang=lan,slow=False)
            tts.save('speech.mp3')
            os.system('start speech.mp3')


#obj = TranslatorModel()
#obj.TextBlob_translator(text="Hello")