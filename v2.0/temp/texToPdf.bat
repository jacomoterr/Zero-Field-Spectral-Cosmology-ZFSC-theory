@echo off
REM === texToPdf.bat ===
REM Компиляция XeLaTeX → PDF (3 раза) в отдельную папку build/

if "%~1"=="" (
    echo Использование: texToPdf имя_файла.tex
    exit /b 1
)

set filename=%~1
set basename=%~dpn1

REM Создать папку build рядом с файлом, если её нет
if not exist "%~dp1build" mkdir "%~dp1build"

echo Компиляция %filename% ...

REM Полный путь к xelatex.exe
set xelatex="C:\Users\50228\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"

%xelatex% -interaction=nonstopmode -synctex=1 -halt-on-error -output-directory="%~dp1build" "%filename%"
%xelatex% -interaction=nonstopmode -synctex=1 -halt-on-error -output-directory="%~dp1build" "%filename%"
%xelatex% -interaction=nonstopmode -synctex=1 -halt-on-error -output-directory="%~dp1build" "%filename%"

if errorlevel 1 (
    echo Ошибка компиляции!
    exit /b %errorlevel%
)

REM Переносим PDF в ту же папку, где лежит .tex
move /Y "%~dp1build\%~n1.pdf" "%~dp1" >nul

echo Готово: %basename%.pdf
pause
