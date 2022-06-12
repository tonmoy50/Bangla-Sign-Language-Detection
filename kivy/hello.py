from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image


class MainApp(App):

    def build(self):
        label = Image(  source="logo.png",
                        size_hint=(1, .5),
                        pos_hint={'center_x': .5, 'center_y': .5} )

        
        return label


if __name__ == "__main__":
    app = MainApp()
    app.run()