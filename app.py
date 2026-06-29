import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import openpyxl

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Создаем папки при старте
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Ленивая загрузка моделей для экономии памяти
models = {}
def get_model(size):
    valid_sizes = ['n', 's', 'm']
    if size not in valid_sizes:
        size = 'n'
    if size not in models:
        models[size] = YOLO(f'yolov8{size}.pt')
    return models[size]

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            orig_path TEXT,
            res_path TEXT,
            persons INTEGER,
            chairs INTEGER,
            occupancy REAL
        )
    ''')
    # Добавляем колонку confidence, если её нет (для обратной совместимости)
    try:
        c.execute('ALTER TABLE history ADD COLUMN confidence REAL')
    except sqlite3.OperationalError:
        pass # Колонка уже существует
    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Получаем параметры из запроса, устанавливаем значения по умолчанию
    confidence = float(request.form.get('confidence', 0.25))
    model_size = request.form.get('model_size', 'n')
    
    if file:
        filename = file.filename
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(orig_path)
        
        # Загружаем выбранную модель и запускаем инференс с заданным порогом
        model = get_model(model_size)
        results = model(orig_path, conf=confidence)
        
        persons = 0
        chairs = 0
        
        # Подсчет объектов нужных классов
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                if class_name == 'person':
                    persons += 1
                elif class_name == 'chair':
                    chairs += 1
        
        # Сохранение изображения с bounding boxes
        res_filename = 'res_' + filename
        res_path = os.path.join(app.config['RESULT_FOLDER'], res_filename)
        
        res_img = results[0].plot()
        cv2.imwrite(res_path, res_img)
        
        # Обработка деления на ноль и ограничение > 100%
        if chairs == 0:
            occupancy = 0.0
            status = "Стулья не обнаружены"
            occupied_seats = 0
        else:
            occupied_seats = min(persons, chairs)
            occupancy = (occupied_seats / chairs) * 100.0
            if persons > chairs:
                status = "Зал переполнен"
            else:
                status = "Норма"
            
        occupancy = round(occupancy, 2)
        
        # Сохранение в базу данных
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('INSERT INTO history (timestamp, orig_path, res_path, persons, chairs, occupancy, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)',
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), orig_path, res_path, persons, chairs, occupancy, confidence))
        conn.commit()
        conn.close()
        
        # Возвращаем JSON для обновления интерфейса без перезагрузки
        return jsonify({
            'people_count': persons,
            'chairs_count': chairs,
            'occupied_seats': occupied_seats,
            'occupancy': occupancy,
            'status': status,
            'res_path': res_path,
            'orig_path': orig_path
        })

@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

@app.route('/export')
def export():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM history')
    rows = c.fetchall()
    conn.close()
    
    # Генерация Excel отчета
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "History Report"
    ws.append(["ID", "Дата и Время", "Оригинал", "Результат", "Кол-во людей", "Кол-во стульев", "Заполненность (%)", "Порог уверенности"])
    
    for row in rows:
        row_list = list(row)
        # Дополняем N/A если запись старая и нет колонки confidence
        if len(row_list) < 8:
            row_list.append('N/A')
        ws.append(row_list)
        
    export_path = "report.xlsx"
    wb.save(export_path)
    
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
