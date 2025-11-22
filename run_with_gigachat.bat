@echo off


REM Установка переменных окружения для GigaChat
set GIGACHAT_AUTH_KEY=you_key
set GIGACHAT_CLIENT_ID=you_key
set GIGACHAT_CLIENT_SECRET=you_key

echo ✅ GigaChat API настроен!
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    pause
    exit /b 1
)

echo Запуск сервера с GigaChat API...
echo Приложение будет доступно по адресу: http://localhost:5000
echo.
echo Для остановки нажмите Ctrl+C
echo.

REM Временно изменяем app.py для использования API режима
python -c "import re; f=open('app.py','r',encoding='utf-8'); c=f.read(); f.close(); c=re.sub(r'neural_mode=\w+', 'neural_mode=\"api\"', c); f=open('app.py','w',encoding='utf-8'); f.write(c); f.close()" 2>nul

python app.py

REM Восстанавливаем обратно
python -c "import re; f=open('app.py','r',encoding='utf-8'); c=f.read(); f.close(); c=re.sub(r'neural_mode=\"api\"', 'neural_mode=None', c); f=open('app.py','w',encoding='utf-8'); f.write(c); f.close()" 2>nul

pause

