from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from book_generator import BookGenerator
from pdf_epub_exporter import PDFEPUBExporter

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

book_generator = BookGenerator(use_models=False, neural_mode="api")
exporter = PDFEPUBExporter()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate_book():
    try:
        data = request.json
        query = data.get('query', '')
        num_pages = data.get('num_pages', 10)

        if not query:
            return jsonify({"error": "Запрос не может быть пустым"}), 400

        book_data = book_generator.generate_book(query, num_pages=num_pages)

        return jsonify({
            "success": True,
            "book": book_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/<format_type>', methods=['POST'])
def download_book(format_type):
    try:
        data = request.json
        book_data = data.get('book')

        if not book_data:
            return jsonify({"error": "Данные книги отсутствуют"}), 400

        if format_type == 'pdf':
            file_path = exporter.export_to_pdf(book_data)
            return send_file(file_path, as_attachment=True, download_name='book.pdf')
        elif format_type == 'epub':
            file_path = exporter.export_to_epub(book_data)
            return send_file(file_path, as_attachment=True, download_name='book.epub')
        else:
            return jsonify({"error": "Неподдерживаемый формат"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    os.makedirs('static/generated_books', exist_ok=True)
    app.run(debug=True, port=5000)