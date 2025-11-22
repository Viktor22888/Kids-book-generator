from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
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
        
        title = book_data.get('title', 'Детская книга')
        pages = book_data.get('pages', [])
        
        cover_image = pages[0].get('image', '') if pages else ''
        self._draw_cover_page(c, width, height, title, title_font, text_font, cover_image)
        c.showPage()
        
        for i, page in enumerate(pages):
            if i > 0:
                c.showPage()
            
            self._draw_content_page(c, width, height, page, text_font, i + 1, len(pages))
        
        c.save()
        return filename
    
    def _draw_cover_page(self, c, width, height, title, title_font, text_font, cover_image=None):
        bg_color = HexColor('#FFF8E1')
        accent_color = HexColor('#FFB74D')
        text_color = HexColor('#5D4037')
        border_color = HexColor('#E0E0E0')
        
        c.setFillColor(bg_color)
        c.rect(0, 0, width, height, fill=1, stroke=0)
        
        if cover_image:
            try:
                img = self._decode_image(cover_image)
                img_width = width * 0.6
                img_height = img_width * (img.height / img.width)
                
                if img_height > height * 0.5:
                    img_height = height * 0.5
                    img_width = img_height * (img.width / img.height)
                
                img_x = (width - img_width) / 2
                img_y = height - 200 - img_height
                
                c.setStrokeColor(border_color)
                c.setLineWidth(3)
                c.roundRect(img_x - 10, img_y - 10, img_width + 20, img_height + 20, 15, stroke=1, fill=0)
                
                img_reader = ImageReader(img)
                c.drawImage(img_reader, img_x, img_y, width=img_width, height=img_height, mask='auto')
                
                title_y = img_y - 60
            except Exception as e:
                print(f"Ошибка добавления изображения на обложку: {e}")
                title_y = height - 200
        else:
            c.setFillColor(accent_color)
            c.circle(width/2, height - 200, 80, fill=1, stroke=0)
            
            c.setFillColor(colors.white)
            c.circle(width/2, height - 200, 70, fill=1, stroke=0)
            title_y = height - 250
        
        c.setFillColor(text_color)
        c.setFont(title_font, 36)
        title_width = c.stringWidth(title, title_font, 36)
        c.drawString((width - title_width) / 2, title_y, title)
        
        c.setFont(text_font, 18)
        subtitle = "Детская книга"
        subtitle_width = c.stringWidth(subtitle, text_font, 18)
        c.drawString((width - subtitle_width) / 2, title_y - 40, subtitle)
        
        c.setFillColor(accent_color)
        c.rect(50, 120, width - 100, 4, fill=1, stroke=0)
        
        c.setFillColor(text_color)
        c.setFont(text_font, 12)
        footer_text = "Создано с помощью Генератора детских книг"
        footer_width = c.stringWidth(footer_text, text_font, 12)
        c.drawString((width - footer_width) / 2, 50, footer_text)
    
    def _draw_content_page(self, c, width, height, page, text_font, page_num, total_pages):
        margin = 60
        content_width = width - 2 * margin
        
        bg_color = HexColor('#FAFAFA')
        border_color = HexColor('#E0E0E0')
        text_color = HexColor('#212121')
        accent_color = HexColor('#FFB74D')
        
        c.setFillColor(bg_color)
        c.rect(0, 0, width, height, fill=1, stroke=0)
        
        c.setStrokeColor(border_color)
        c.setLineWidth(2)
        c.roundRect(margin - 5, margin - 5, content_width + 10, height - 2 * margin + 10, 10, stroke=1, fill=0)
        
        c.setFillColor(accent_color)
        c.rect(margin, height - margin - 30, content_width, 25, fill=1, stroke=0)
        
        c.setFillColor(colors.white)
        c.setFont(text_font, 14)
        page_text = f"Страница {page_num}"
        page_text_width = c.stringWidth(page_text, text_font, 14)
        c.drawString(margin + 15, height - margin - 20, page_text)
        
        img_height = 0
        try:
            img = self._decode_image(page.get('image', ''))
            max_img_width = content_width - 20
            max_img_height = height * 0.45
            
            img_ratio = img.width / img.height
            if img.width > img.height:
                img_width = min(max_img_width, max_img_height * img_ratio)
                img_height = img_width / img_ratio
            else:
                img_height = min(max_img_height, max_img_width / img_ratio)
                img_width = img_height * img_ratio
            
            img_x = margin + (content_width - img_width) / 2
            img_y = height - margin - 80 - img_height
            
            c.setStrokeColor(border_color)
            c.setLineWidth(1)
            c.roundRect(img_x - 5, img_y - 5, img_width + 10, img_height + 10, 8, stroke=1, fill=0)
            
            img_reader = ImageReader(img)
            c.drawImage(img_reader, img_x, img_y, width=img_width, height=img_height, mask='auto')
        except Exception as e:
            print(f"Ошибка добавления изображения: {e}")
        
        text = page.get('text', '')
        text_start_y = img_y - 40 if img_height > 0 else height - margin - 100
        
        c.setFillColor(text_color)
        c.setFont(text_font, 16)
        
        max_text_width = content_width - 20
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if c.stringWidth(test_line, text_font, 16) <= max_text_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        line_height = 22
        y_position = text_start_y
        
        for line in lines:
            if y_position < margin + 40:
                break
            line_width = c.stringWidth(line, text_font, 16)
            c.drawString(margin + (content_width - line_width) / 2, y_position, line)
            y_position -= line_height
        
        c.setFillColor(HexColor('#9E9E9E'))
        c.setFont(text_font, 10)
        footer_text = f"{page_num} / {total_pages}"
        footer_width = c.stringWidth(footer_text, text_font, 10)
        c.drawString((width - footer_width) / 2, margin - 20, footer_text)
    
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

