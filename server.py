import http.server
import socketserver
import webbrowser
import os

PORT = 8080

# Переходим в папку со скриптом
os.chdir(os.path.dirname(os.path.abspath(__file__)))

handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"Сервер запущен: {url}")
    print(f"Папка: {os.getcwd()}")
    print(f"Открываю браузер...")
    print(f"Для остановки нажмите Ctrl+C")
    
    webbrowser.open(url)  # автоматически открывает браузер
    httpd.serve_forever()