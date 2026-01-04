from PyQt5.QtWidgets import QComboBox, QSlider, QPushButton


# Ignore mouse wheel to prevent accidental selection changes
class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

# Ignore mouse wheel to prevent accidental value changes
class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

# Make the button non-interactive to mouse clicks (visual indicator only)
class NonClickableButton(QPushButton):
    def mousePressEvent(self, event):
        event.ignore()
    def mouseReleaseEvent(self, event):
        event.ignore()