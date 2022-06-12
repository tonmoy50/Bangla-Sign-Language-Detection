from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.loader import Loader
from kivy.uix.button import Button
import os
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, ListProperty
from kivy.uix.popup import Popup
import detect_sign
from detect_sign import SignDetection
#from kivy.clock import Clock
from t_to_s import *




class HomeWindow(Screen):
    pass
    

class SecondWindow(Screen):
    pass

class WindowManager(ScreenManager):
    image_path = StringProperty(defaultvalue='blank.jpg')
    logo_path = StringProperty(defaultvalue='logo.png')
    sign_text = StringProperty(defaultvalue="None")
    cr = StringProperty(defaultvalue="(c) Tonmoy")
    bangla_text = StringProperty(defaultvalue="None")
    

    def __init__(self, **kwargs):
        
        super(WindowManager, self).__init__(**kwargs)
        self.home_screen()
        
    

    def selected(self, filename):
        print(filename)
        

    def open(self, path, filename):
        #print(filename[0])
        try:

            self.image_path = filename[0]
            self.home_screen()
        except IndexError:
            popup = Popup(title='Hello Stupid', content=Label(text='Please Select an Image :)'), auto_dismiss=True, pos_hint={'center_x':.5, 'center_y':.5}, size_hint=(.5, .5))
            popup.open()
            popup.bind(on_press=popup.dismiss)

        

        #return path
    
    def set_sign_name(self):
        sd = SignDetection(image_path=self.image_path)
        self.sign_text = sd.class_name
        
        to_bangla = TranslatorModel()
        self.sign_text = str( to_bangla.TextBlob_translator(self.sign_text) )
        self.home_screen()



    def home_screen(self):
        self.current = "__home__"
        self.ids.home.ids.detect_image.bind(on_release=lambda x: self.set_sign_name() )

        self.ids.home.ids.add_image.bind(on_release=lambda x: self.second_screen() ) #Action for switching to second screen
        
        
        
        #print(self.image_path)
        


    def second_screen(self):
        filechooser = self.ids.second.ids.file
        self.current = "__second__"

        #filechooser.bind(on_selection=lambda x: self.selected(filechooser.path) )
        #print(filechooser.path)
        self.image_path = filechooser.path
        self.ids.second.ids.done.bind(on_release=lambda x:self.open(filechooser.path, filechooser.selection) )
        #self.ids.second.ids.done.bind(on_release=lambda x:self.home_screen() ) ##Action for switching to second screen
        #print(self.image_path)
        #self.image_path = self.ids.second.ids.file.path

        
        
        

    

class MainApp(App):
    def build(self):
        
        
        self.load_kv('test2.kv')
        
        return WindowManager()
    


if __name__ == "__main__":
    MainApp().run()

