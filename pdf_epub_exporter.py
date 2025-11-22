from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import ebooklib
from ebooklib import epub
from PIL import Image
import base64
import io
import os
from datetime import datetime

class PDFEPUBExporter:
    def __init__(self):
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self._register_fonts()
    
    def _register_fonts(self):
        try:
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/times.ttf",
                "C:/Windows/Fonts/timesbd.ttf",
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        if "arial" in font_path.lower():
                            if "bd" in font_path.lower():
                                pdfmetrics.registerFont(TTFont('Arial-Bold', font_path))
                            else:
                                pdfmetrics.registerFont(TTFont('Arial', font_path))
                        elif "times" in font_path.lower():
                            if "bd" in font_path.lower():
                                pdfmetrics.registerFont(TTFont('Times-Bold', font_path))
                            else:
                                pdfmetrics.registerFont(TTFont('Times', font_path))
                    except Exception as e:
                        print(f"Не удалось зарегистрировать шрифт {font_path}: {e}")
            
            # Проверяем, зарегистрирован ли хотя бы один шрифт
            registered_fonts = pdfmetrics.getRegisteredFontNames()
            if 'Arial' not in registered_fonts:
                print("⚠️  Шрифты с кириллицей не найдены. PDF может отображать квадраты вместо букв.")
        except Exception as e:
            print(f"Ошибка регистрации шрифтов: {e}")
    
    def _decode_image(self, image_data):

        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))
    
    def export_to_pdf(self, book_data):
        filename = f"{self.temp_dir}/book_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        
        registered_fonts = pdfmetrics.getRegisteredFontNames()
        title_font = "Arial-Bold" if "Arial-Bold" in registered_fonts else ("Arial" if "Arial" in registered_fonts else "Helvetica-Bold")
        text_font = "Arial" if "Arial" in registered_fonts else "Helvetica"
        
        c.setFont(title_font, 24)
        title = book_data.get('title', 'Детская книга')
        c.drawString(50, height - 50, title)
        
        pages = book_data.get('pages', [])
        for i, page in enumerate(pages):
            if i > 0:
                c.showPage()
            

            try:
                img = self._decode_image(page.get('image', ''))
                img_width = width - 100
                img_height = img_width * (img.height / img.width)
            
                if img_height > height * 0.5:
                    img_height = height * 0.5
                    img_width = img_height * (img.width / img.height)
                
                img_reader = ImageReader(img)
                c.drawImage(img_reader, 50, height - img_height - 100, 
                           width=img_width, height=img_height)
            except Exception as e:
                print(f"Ошибка добавления изображения: {e}")
            

            c.setFont(text_font, 14)
            text = page.get('text', '')
            
            max_width = width - 100
            words = text.split(' ')
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                if c.stringWidth(test_line, text_font, 14) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            y_position = height - img_height - 120
            for line in lines:
                if y_position < 100:
                    break
                c.drawString(50, y_position, line)
                y_position -= 20
        
        c.save()
        return filename
    
    def export_to_epub(self, book_data):
        filename = f"{self.temp_dir}/book_{datetime.now().strftime('%Y%m%d_%H%M%S')}.epub"
        book = epub.EpubBook()
        
        title = book_data.get('title', 'Детская книга')
        book.set_identifier('book_' + datetime.now().strftime('%Y%m%d%H%M%S'))
        book.set_title(title)
        book.set_language('ru')
        book.add_author('Генератор детских книг')
        
        pages = book_data.get('pages', [])
        chapters = []
        
        for i, page in enumerate(pages):
            chapter = epub.EpubHtml(
                title=f'Страница {page["page_number"]}',
                file_name=f'page_{i+1}.xhtml',
                lang='ru'
            )
            
            try:
                img = self._decode_image(page.get('image', ''))
                img_path = f'images/page_{i+1}.png'
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                book.add_item(epub.EpubItem(
                    uid=f'img_{i+1}',
                    file_name=img_path,
                    media_type='image/png',
                    content=img_bytes.getvalue()
                ))
                
                img_tag = f'<img src="{img_path}" alt="Иллюстрация" style="max-width: 100%;"/>'
            except Exception as e:
                print(f"Ошибка обработки изображения: {e}")
                img_tag = ''
            
            text = page.get('text', '')
            chapter.content = f'''
            <html>
            <head>
                <title>Страница {page["page_number"]}</title>
            </head>
            <body>
                <h2>Страница {page["page_number"]}</h2>
                {img_tag}
                <p>{text}</p>
            </body>
            </html>
            '''
            
            book.add_item(chapter)
            chapters.append(chapter)
        

        book.toc = tuple(chapters)
        

        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
     
        book.spine = ['nav'] + chapters
        
   
        epub.write_epub(filename, book, {})
        return filename

