from PyQt6.QtGui import QIcon, QFont, QPixmap
from PyQt6.QtCore import QDir, Qt, QUrl, QSize
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QColorDialog,
        QPushButton, QCheckBox, QProgressBar, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar, QMainWindow, QLineEdit)
from custom_color import custom_color
import os
import dlib 
from PIL import Image
from face import get_face
import warnings
# ignore warnings
import torch
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
# PATH = ROOT# + '/Artificial-Barber/' 
StyleSheet = '''
#RedProgressBar {
    text-align: center;
}
'''
class ColorDialog(QColorDialog, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOptions(self.options() | QColorDialog.ColorDialogOption.DontUseNativeDialog)

        for children in self.findChildren(QWidget):
            classname = children.metaObject().className()
            if classname not in ("QColorPicker", "QColorLuminancePicker"):
                children.hide()

class MainWindow(QMainWindow):
    
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Artificial Barber")
        # make a qhboxlayout to hold an image and an upload button for the image
        self.images_info = {'image_path': None, 'target_hair_color': None, 'target_hair_style': None, 'time_or_efficiency': None}
        self.main_layout = QVBoxLayout()
        
        self.image_layout = QHBoxLayout()
        
        self.raw_image = QLabel('Raw Image')
        self.raw_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_image.setStyleSheet("border: 1px solid black")
        self.raw_image.setFixedSize(400, 400)
        self.raw_image.setScaledContents(True)
        
        self.result_image = QLabel('Result Image')
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setStyleSheet("border: 1px solid black")
        self.result_image.setFixedSize(400, 400)
        self.result_image.setScaledContents(True)
        
        self.image_layout.addWidget(self.raw_image)
        self.image_layout.addWidget(self.result_image)
        self.main_layout.addLayout(self.image_layout)
        sample_image_path = ROOT + '/img/additional/guidelines.png'
        self.upload_button = QPushButton("Upload your picture")
        self.upload_button.setToolTip(f'<br><img src="{sample_image_path}">')
        self.upload_button.setStatusTip("Please upload only the image of 1 person")
        self.upload_button.clicked.connect(self.upload_image)
        # self.upload_button.setStyleSheet("background-color: #00bfff; color: white; font-size: 20px; font-weight: bold")
        self.upload_button.setFixedSize(400, 50)
        self.upload_button.setShortcut('Ctrl+U')
        self.main_layout.addWidget(self.upload_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # now let's add a status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Waiting for the image...')
        # self.status_bar.setStyleSheet("background-color: grey; color: white; font-size: 15px")
        self.status_bar.setFixedSize(400, 50)
        self.main_layout.addWidget(self.status_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # now we will add a label with 2 checkboxes for choosing the type of the model -> either time efficient or accuracy efficient
        # self.choose_model_type = QLabel('Choose the type of the model')
        # self.choose_model_type.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.choose_model_type.setFixedSize(200, 50)
        self.time_efficient = QCheckBox('Fast Inference')
        self.time_efficient.setStyleSheet("font-size: 15px")
        self.time_efficient.setFixedSize(150, 50)
        # self.time_efficient.setChecked(True)
        
        self.performance_efficient = QCheckBox('Best Quality')
        self.performance_efficient.setStyleSheet("font-size: 15px")
        self.performance_efficient.setFixedSize(200, 50)
        # self.performance_efficient.setChecked(True)

        self.time_efficient.stateChanged.connect(self.show_time_state)
        self.performance_efficient.stateChanged.connect(self.show_performance_state)
        
        self.model_layout = QHBoxLayout()
        # self.model_layout.addWidget(self.choose_model_type, alignment=Qt.AlignmentFlag.AlignRight)
        self.model_layout.addWidget(self.time_efficient, alignment=Qt.AlignmentFlag.AlignRight)
        self.model_layout.addWidget(self.performance_efficient, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addLayout(self.model_layout)
        
        self.colors_and_style_layout = QHBoxLayout()
        
        self.colors_and_title_layout = QVBoxLayout()
        self.colors_and_title_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.color_title = QLabel('Choose the color of your target hair')
        self.color_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.color_title.setStyleSheet("font-size: 15px; color: white; background-color: grey")
        self.color_title.setFixedSize(350, 30)
        # self.color_title.setWordWrap(True)
        
        self.colors_and_title_layout.addWidget(self.color_title)
        self.colors_and_custom_layout = QVBoxLayout()
        self.colors_layout = QHBoxLayout()
        
        self.colors1 = ['basic', 'red', 'brown', 'blue', 'blond']
        self.color_pushs = {}
        for color in self.colors1:
            if color == 'basic':
                self.color_label = QLabel('Basic')
                self.color_label.setFixedSize(50, 30)
                self.color_label.setStyleSheet("font-size: 16px; color: white; background-color: purple; font-family: Monaco")
                self.colors_layout.addWidget(self.color_label)
                continue
            self.color_button = QPushButton(color)
            self.color_pushs[f'{color}_push'] = self.color_button
            self.color_button.setCheckable(True)
            target_sample_color_path = ROOT + f'/img/appearance/{color}.png'
            self.color_button.setToolTip(f'<br><img src="{target_sample_color_path}" width="256" height="256">')
            self.color_button.clicked.connect(lambda state, color=color: self.select_color(state, color=color))
            
            # self.color_button.setStyleSheet("QPushButton:checked {background-color: %s; color: white; font-size: 15px; font-weight: bold}" % color)
            self.color_button.setStyleSheet(self.color_style_sheet(color))
            self.color_button.setFixedSize(70, 50)
            self.colors_layout.addWidget(self.color_button)
        
        self.colors_layout2 = QHBoxLayout()
        self.colors2 = ['fancy', '2 Colors', '3 Colors', 'Rainbow 1', 'Rainbow 2']
        self.colors2_colors = ['qlineargradient(x1:0, y1:0, x2:1, y2:1, stop: 0.5 #A0A0A0, stop: 1 #FF66B2)', 
                               'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop: 0.25 pink, stop: 0.5 purple, stop: 0.75 black)', 
                               'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop: 0.1 red, stop: 0.3 orange, stop: 0.5 green, stop: 0.9 blue)', 
                               'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop: 0.1 yellow, stop: 0.3 blue, stop: 0.5 green, stop: 0.9 red)']
        self.colors2_images = ['double.png', 'ppb.png', 'rainbow.png', 'rainbow2.png']
        self.colors2_and_images = {color: self.colors2_images[i-1] for i, color in enumerate(self.colors2) if i != 0}
        for i, color in enumerate(self.colors2):
            if color == 'fancy':
                self.color_label = QLabel('Combo')
                self.color_label.setFixedSize(50, 30)
                self.color_label.setStyleSheet("font-size: 16px; color: white; background-color: purple; font-family: Monaco")
                self.colors_layout2.addWidget(self.color_label)
                continue
            self.color_button = QPushButton(color)
            self.color_pushs[f'{color}_push'] = self.color_button
            self.color_button.setCheckable(True)
            self.color_button.setToolTip(f'<br><img src="{ROOT}/img/appearance/{self.colors2_and_images[color]}" width="256" height="256">')
            self.color_button.clicked.connect(lambda state, color=color: self.select_color(state, color=color))
            # self.color_button.setStyleSheet("QPushButton:checked {background-color: %s; color: white; font-size: 15px; font-weight: bold}" % color)
            self.color_button.setStyleSheet(self.color_style_sheet(self.colors2_colors[i-1]))
            # link the button to a function
            self.color_button.setFixedSize(70, 50)
            self.colors_layout2.addWidget(self.color_button)
        self.colors_and_custom_layout.addLayout(self.colors_layout)
        self.colors_and_custom_layout.addLayout(self.colors_layout2)
        self.use_custom_color = QCheckBox('Use custom color')
        self.use_custom_color.setStyleSheet("font-size: 15px")
        self.use_custom_color.stateChanged.connect(self.using_custom_color)
        self.colors_and_custom_layout.addWidget(self.use_custom_color)
        self.customs_layout = QHBoxLayout()
        self.custom_button = ColorDialog()
        self.custom_button.currentColorChanged.connect(self.onCurrentColorChanged)
        self.custom_push_and_label = QVBoxLayout()
        self.custom_color_label = QLabel('Custom Color Image')
        self.custom_color_label.setFixedSize(150, 30)
        self.custom_push_and_label.addWidget(self.custom_color_label)
        self.onCurrentColorChanged(self.custom_button.currentColor())
        # self.custom_color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.colors_and_custom_layout.addWidget(self.custom_text)
        # self.status_bar2 = QStatusBar()
        # self.status_bar2.showMessage('Custom or default color')
        # self.status_bar2.setStyleSheet("background-color: grey; color: white; font-size: 15px")
        # self.status_bar2.setFixedSize(400, 50)
        self.custom_upload_button = QPushButton('Custom Color Image')
        self.custom_upload_button.setFixedSize(150, 30)
        self.custom_push_and_label.addWidget(self.custom_upload_button)
        self.customs_layout.addLayout(self.custom_push_and_label)
        iscolor = True
        self.custom_upload_button.clicked.connect(lambda state, iscolor=iscolor: self.custom_upload(state, iscolor=iscolor))
        self.customs_layout.addWidget(self.custom_button)
        self.colors_and_custom_layout.addLayout(self.customs_layout)#, alignment=Qt.AlignmentFlag.AlignCenter)
        # self.colors_and_custom_layout.addWidget(self.custom_color_label, alignment=Qt.AlignmentFlag.AlignCenter)
        # self.colors_and_custom_layout.addWidget(self.status_bar2, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.colors_and_title_layout.addLayout(self.colors_and_custom_layout)
        
        self.style_and_title_layout = QVBoxLayout()
        # self.style_and_title_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.style_title = QLabel('Choose the style of your target hair')
        self.style_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.style_title.setStyleSheet("font-size: 15px; color: white; background-color: grey")
        self.style_title.setFixedSize(400, 30)
        self.style_and_title_layout.addWidget(self.style_title)
        self.style_layout = QHBoxLayout()
        self.style_pushs = {}
        self.style1_layout = QVBoxLayout()
        self.style2_layout = QVBoxLayout()
        self.styles1 = ['straight', 'curly', 'bob']
        self.styles2 = ['afro', 'fade', 'wavy']
        self.styles = self.styles1 + self.styles2
        for i, style in enumerate(self.styles):
            self.style_button = QPushButton(style)
            self.style_pushs[f'{style}_push'] = self.style_button
            sample_style_path = f'{ROOT}/img/structure/{style}.png'
            # the tooltip is the image of the style, but we have to resize the image to fit the tooltip
            self.style_button.setToolTip(f'<br><img src="{sample_style_path}" width="256" height="256">')
            # self.style_button.setStyleSheet(f"background-color: {self.colors1[i]}; color: white; font-size: 15px")
            self.style_button.clicked.connect(lambda state, style=style: self.select_style(state, style=style))
            self.style_button.setCheckable(True)
            self.style_button.setFixedSize(100, 50)
            self.style_button.setStyleSheet("QPushButton:checked {color: white; background-color: green;}")

            # self.style_button.setShortcut('Ctrl+{}'.format(self.styles.index(style)+1))
            if style in self.styles1:
                self.style1_layout.addWidget(self.style_button)
            elif style in self.styles2:
                self.style2_layout.addWidget(self.style_button)
            # self.style1_layout.addWidget(self.style_button)
        self.style_layout.addLayout(self.style1_layout)
        self.style_layout.addLayout(self.style2_layout)
        iscolor = False
        self.custom_style = QPushButton('Custom Style Image')
        self.custom_style.clicked.connect(lambda state, iscolor=iscolor: self.custom_upload(state, iscolor=iscolor))
        # self.custom_style.clicked.connect(lambda state: self.custom_upload(state, iscolor=iscolor))
        self.custom_style.setFixedSize(150, 30)
        self.progress_bar = QStatusBar()
        self.progress_bar.showMessage('Click the button below for analysis')
        self.progress_bar.setStyleSheet("font-size: 15px; color: white; background-color: purple")
        self.progress_bar.setFixedSize(250, 50)
        # self.progress_bar = QProgressBar(minimum=0, maximum=300, value=0, alignment=Qt.AlignmentFlag.AlignCenter, objectName='RedProgressBar')
        # self.progress_bar.setFixedSize(200, 50)
        # self.progress_bar.setStyleSheet("font-size: 15px; color: white; background-color: purple; text-align: center")
        self.generate_button = QPushButton('Generate')
        self.generate_button.clicked.connect(self.generate)
        self.generate_button.setFixedSize(100, 50)
        self.generate_button.setStyleSheet("background-color: green; color: white; font-size: 15px")
        
        self.style_and_title_layout.addLayout(self.style_layout)
        self.style_and_title_layout.addWidget(self.custom_style, alignment=Qt.AlignmentFlag.AlignCenter)
        self.style_and_title_layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.style_and_title_layout.addWidget(self.generate_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.colors_and_style_layout.addLayout(self.colors_and_title_layout)
        self.colors_and_style_layout.addLayout(self.style_and_title_layout)
        
        self.main_layout.addLayout(self.colors_and_style_layout)
        
        widget = QWidget()
        widget.setLayout(self.main_layout)
        # widget.setStyleSheet("background-color: grey")
        self.setCentralWidget(widget)

    def onCurrentColorChanged(self, color):
        self.custom_color_label.setText(f'Color ID - {color.name()}')
        self.custom_color_label.setStyleSheet(f"background-color: {color.name()}; color: white; font-size: 15px")
        if self.use_custom_color.isChecked():
            # Operations to do when the custom color is selected
            r, g, b = color.getRgb()[:3]
            custom_color(b, g, r)
            self.images_info['target_hair_color'] = f'img/custom/new_hair_{b}_{g}_{r}.png'
        
    def using_custom_color(self, state):
        for col, button in self.color_pushs.items():
            button.setCheckable(state == 0)
            if button.isCheckable():
                col = col[:-5]
                if col in self.colors2:
                    button.setStyleSheet(self.color_style_sheet(self.colors2_colors[self.colors2.index(col)-1]))
                else:
                    button.setStyleSheet(self.color_style_sheet(col))
            else:
                button.setStyleSheet("background-color: black")
    def generate(self):
        # print('image_info', self.images_info)
        if None in self.images_info.values():
            nons = [k for k, v in self.images_info.items() if v is None]
            nons_message = 'You didn\'t select ' + ', '.join(nons)
            self.progress_bar.showMessage(nons_message)
            self.progress_bar.setStyleSheet("font-size: 15px; color: white; background-color: red")
            self.progress_bar.setFixedSize(400, 50)
        else:
            self.progress_bar.showMessage('Generating in progress...')
            self.progress_bar.setStyleSheet("font-size: 15px; color: white; background-color: purple")
            self.progress_bar.setFixedSize(250, 50)
            # let's exectute command using subprocess
            import subprocess
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.images_info['time_or_efficiency'] == 'time':
                W_step_size, FS_step_size = 250, 200
            else:
                W_step_size, FS_step_size = 1100, 250
            command = f"python3 inference.py --im_path1 {self.images_info['image_path']} --im_path2 {self.images_info['target_hair_style']} --im_path3 {self.images_info['target_hair_color']} --W_steps {W_step_size} --FS_steps {FS_step_size} --device {device}"
            subprocess.call(command, shell=True)

            # let's show the generated image in the resul_image
            result_image_path = f"output/{self.images_info['image_path'].split('/')[-1].split('.')[0]}_{self.images_info['target_hair_style'].split('/')[-1].split('.')[0]}_{self.images_info['target_hair_color'].split('/')[-1].split('.')[0]}_realistic.png"
            self.result_image.setPixmap(QPixmap(result_image_path))
            
            
    def show_time_state(self, state):
        if state == 2:
            # self.performance_efficient.setCheckable(False)
            self.performance_efficient.hide()
            self.model_layout.addWidget(self.time_efficient, alignment=Qt.AlignmentFlag.AlignCenter)
            self.images_info['time_or_efficiency'] = 'time'
        else:
            # self.performance_efficient.setCheckable(True)
            self.performance_efficient.show()
            self.model_layout.addWidget(self.time_efficient, alignment=Qt.AlignmentFlag.AlignRight)
            self.model_layout.addWidget(self.performance_efficient, alignment=Qt.AlignmentFlag.AlignLeft)
            self.images_info['time_or_efficiency'] = None

    def show_performance_state(self, state):
        if state == 2:
            # self.time_efficient.setCheckable(False)
            self.time_efficient.hide()
            self.model_layout.addWidget(self.performance_efficient, alignment=Qt.AlignmentFlag.AlignCenter)
            
            self.images_info['time_or_efficiency'] = 'efficiency'
        else:
            # self.time_efficient.setCheckable(True)
            self.time_efficient.show()
            self.model_layout.addWidget(self.time_efficient, alignment=Qt.AlignmentFlag.AlignRight)
            self.model_layout.addWidget(self.performance_efficient, alignment=Qt.AlignmentFlag.AlignLeft)
            self.images_info['time_or_efficiency'] = None
    
    def color_style_sheet(self, color):
        uncheck_color = 'black' if color == 'blond' else 'white' 
        if color == 'blond':
            color = '#FAF0BE'
        return "\
                QPushButton {   \
                    color:%s;    \
                    background-color:%s;    \
                    border-radius: 15px;    \
                }   \
                QPushButton:checked{\
                    background-color: #CCFF99;\
                    border-radius: 15px;\
                    color: black;    \
                }\
                " % (uncheck_color, color)
    
    def select_color(self, state, color=None):
        # self.images_info['image_hair_color'] = color
        
        if not self.use_custom_color.isChecked():
            if state:
                # disable all other buttons
                for col, button in self.color_pushs.items():
                    if not col.startswith(color):
                        button.setCheckable(False)
                        button.setStyleSheet("background-color: black")
                if color in self.colors2:
                    color_path = ROOT + f'/img/appearance/{self.colors2_and_images[color]}'
                else:
                    color_path = ROOT + f'/img/appearance/{color}.png'
                self.images_info['target_hair_color'] = color_path
            elif not state and (color == self.images_info['target_hair_color'].split('/')[-1].split('.')[0] or (color in self.colors2 and self.colors2_and_images[color] == self.images_info['target_hair_color'].split('/')[-1])):
                    # enable all other buttons
                for col, button in self.color_pushs.items():
                    if not col.startswith(color):
                        button.setCheckable(True)
                        col = col[:-5]
                        if col in self.colors2:
                            button.setStyleSheet(self.color_style_sheet(self.colors2_colors[self.colors2.index(col)-1]))
                        else:
                            button.setStyleSheet(self.color_style_sheet(col))
                        
                self.images_info['target_hair_color'] = None
        
    def select_style(self, state, style):
        sample_style_path = ROOT + f'/img/structure/{style}.png'
        if state:
            self.images_info['target_hair_style'] = sample_style_path
            for s, button in self.style_pushs.items():
                if not s.startswith(style):
                    button.setCheckable(False)
                    button.setStyleSheet("background-color: black")
        elif not state and style == self.images_info['target_hair_style'].split('/')[-1].split('.')[0]:
            for s, button in self.style_pushs.items():
                if not s.startswith(style):
                    button.setCheckable(True)
                    button.setStyleSheet("QPushButton:checked {color: white; background-color: green;}")
            self.images_info['target_hair_style'] = None
                    
    
    def upload_image(self):
        # get the path of the image
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        # print(self.image_path)
        image_name = self.image_path.split('/')[-1]
        img_format = image_name.split('.')[-1]
        # set the image to the raw image label
        self.raw_image.setPixmap(QPixmap(self.image_path))
        # self.raw_image.setScaledContents(True)
        
        self.detector = dlib.get_frontal_face_detector()
        # we can use the model that is trained on the 68 points
        if self.image_path != '':
            img = dlib.load_rgb_image(self.image_path)
            detections = self.detector(img, 1)
            num_faces = len(detections) 
        if self.image_path == '' or num_faces == 0:
            self.status_bar.showMessage('No face detected. Please upload your image again')
            self.status_bar.setStyleSheet("background-color: red")
            
            # here we should clear the image
            self.raw_image.clear()
        
        elif num_faces != 1:
            self.status_bar.setFixedSize(500, 50)
            self.status_bar.showMessage('Multiple faces have been detected. Please use an image with only one face')
            self.status_bar.setStyleSheet("background-color: red")
            
            # here too, we should clear the image
            self.raw_image.clear()
        
        else:
            # get the coordinates of the face
            target_face = get_face(self.image_path, self.detector, output_size=1024)
            
            self.image_path = ROOT + '/img/input/' + image_name.split('.')[0] + '_face' + '.' + img_format
            # print(self.image_path)
            target_face.save(self.image_path)
            
            self.status_bar.showMessage('Image is valid. You can proceed to model analysis')
            self.status_bar.setStyleSheet("background-color: green")
            self.images_info['image_path'] = self.image_path
            self.raw_image.setPixmap(QPixmap(self.image_path))
    
    def custom_upload(self, state, iscolor=True):
        self.custom_image, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        # print(self.image_path)
        image_name = self.custom_image.split('/')[-1]
        img_format = image_name.split('.')[-1]
        # set the image to the raw image label
        # self.raw_image.setPixmap(QPixmap(self.custom_image))
        # self.raw_image.setScaledContents(True)
        
        self.detector = dlib.get_frontal_face_detector()
        # we can use the model that is trained on the 68 points
        if self.custom_image != '':
            img = dlib.load_rgb_image(self.custom_image)
            detections = self.detector(img, 1)
            num_faces = len(detections) 
        if self.custom_image == '' or num_faces == 0:
            self.status_bar.showMessage('No face detected. Please upload your image again')
            self.status_bar.setStyleSheet("background-color: red")
        
        elif num_faces != 1:
            self.status_bar.setFixedSize(500, 50)
            self.status_bar.showMessage('Multiple faces have been detected. Please use an image with only one face')
            self.status_bar.setStyleSheet("background-color: red")
        
        else:
            # get the coordinates of the face
            
            custom_face = get_face(self.custom_image, self.detector, output_size=1024)
            folder_place = 'appearance' if iscolor else 'structure'
            custom_face_path = ROOT + f'/img/{folder_place}/' + image_name.split('.')[0] + '.' + img_format
            # print(self.image_path)
            custom_face.save(custom_face_path)
            label = 'color' if iscolor else 'style'
            self.status_bar.showMessage(f'Custom {label} uploaded successfully')
            self.status_bar.setStyleSheet("background-color: green")
            # self.delete_custom_color.show()
            self.images_info[f'target_hair_{label}'] = custom_face_path
    
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    player = MainWindow()
    player.resize(600, 400)
    player.show()
    sys.exit(app.exec())