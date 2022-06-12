from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.loader import Loader
from kivy.uix.button import Button
import os


class MainApp(App):

    def open(self, path, filename):
        print(filename)
        return path
        #if len(filename) > 0:
           # with open(os.path.join(path, filename[0])) as f:
                #print(f.read())
        

    def selected(self, filename):
        temp = filename[0]
        
        #print("selected: %s" % temp)
        return temp



    def choose_file(self, instances):
        filechooser = FileChooserIconView()
        filechooser.bind(on_selection=lambda x: self.selected(filechooser.selection))
        self.main_layout.add_widget(filechooser)

        btn2 = Button(text='Add', size_hint=(1, .2))
        x = btn2.bind(on_release=lambda x: self.open(filechooser.path, filechooser.selection))
        self.main_layout.add_widget(btn2)
        return x

    def build(self):
        self.main_layout = BoxLayout(orientation='vertical')
        title = Label( text='Bangla Hand Sign Detection',
                        size_hint=(1, 1),
                        pos_hint={'center_x': .5, 'center_y': .9},
                        font_size=24 )
        self.main_layout.add_widget(title)

        #img = Image(source='logo.png')
        #print(img)
        #main_layout.add_widget(img)

        #filechooser = FileChooserIconView()
        #filechooser.bind(on_selection=lambda x: self.selected(filechooser.selection))
        #main_layout.add_widget(filechooser)
        #print(filechooser.selection)

        #open_btn = Button(text='open', size_hint=(1, .2))
        #open_btn.bind(on_release=lambda x: self.open(filechooser.path, filechooser.selection))

        
        #main_layout.add_widget(open_btn)
        
        btn1 = Button(text='Add Image', size_hint=(1, .2))
        path = btn1.bind(on_press=self.choose_file)
        self.main_layout.add_widget(btn1)

        print(type(path) )


        return self.main_layout






if __name__ == "__main__":
    app1 = MainApp()
    app1.run()


