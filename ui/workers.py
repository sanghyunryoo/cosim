from PyQt5.QtCore import QObject, pyqtSignal
from core.tester import Tester


class TesterWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, tester: Tester):
        super().__init__()
        self.tester = tester
    def run(self):
        # Execute tester in a separate thread context
        try:
            self.tester.init_user_command()
            self.tester.test()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()