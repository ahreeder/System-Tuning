import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from main_window import MainWindow


def _dark_palette():
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(22, 22, 35))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(210, 218, 244))
    p.setColor(QPalette.ColorRole.Base,            QColor(15, 15, 26))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(28, 28, 42))
    p.setColor(QPalette.ColorRole.Text,            QColor(210, 218, 244))
    p.setColor(QPalette.ColorRole.Button,          QColor(45, 46, 65))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(210, 218, 244))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(100, 160, 250))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(15, 15, 26))
    p.setColor(QPalette.ColorRole.Mid,             QColor(80, 82, 105))
    p.setColor(QPalette.ColorRole.Dark,            QColor(35, 36, 52))
    return p


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PA Tuning Tool")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
