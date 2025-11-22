const API_URL = 'http://localhost:5000';


const bookForm = document.getElementById('bookForm');
const generateBtn = document.getElementById('generateBtn');
const bookContainer = document.getElementById('bookContainer');
const bookTitle = document.getElementById('bookTitle');
const bookPages = document.getElementById('bookPages');
const errorMessage = document.getElementById('errorMessage');
const downloadPdfBtn = document.getElementById('downloadPdf');

let currentBookData = null;

bookForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('query').value.trim();
    const numPages = parseInt(document.getElementById('numPages').value) || 10;

    if (!query) {
        showError('Пожалуйста, введите запрос для книги');
        return;
    }

    setLoading(true);
    hideError();
    hideBook();

    try {
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, num_pages: numPages })
        });
        const data = await response.json();

        if (data.success && data.book) {
            currentBookData = data.book;
            displayBook(data.book);
        } else {
            showError(data.error || 'Произошла ошибка при генерации книги');
        }
    } catch (error) {
        console.error('Ошибка:', error);
        showError('Не удалось подключиться к серверу. Убедитесь, что сервер запущен.');
    } finally {
        setLoading(false);
    }
});

downloadPdfBtn.addEventListener('click', downloadAsPdf);

function downloadAsPdf() {
    if (!currentBookData) {
        showError('Нет книги для скачивания');
        return;
    }

    setPdfLoading(true);

    fetch(`${API_URL}/api/download/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            book: currentBookData
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Ошибка при создании PDF');
            });
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'book.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error('Ошибка при создании PDF:', error);
        showError(error.message || 'Не удалось создать PDF файл');
    })
    .finally(() => {
        setPdfLoading(false);
    });
}

function displayBook(book) {
    bookTitle.textContent = book.title;
    bookPages.innerHTML = '';

    book.pages.forEach(page => {
        bookPages.appendChild(createPageElement(page));
    });

    bookContainer.style.display = 'block';
    bookContainer.scrollIntoView({ behavior: 'smooth' });
}

function createPageElement(page) {
    const pageDiv = document.createElement('div');
    pageDiv.className = 'book-page';

    const pageNumber = document.createElement('div');
    pageNumber.className = 'page-number';
    pageNumber.textContent = `Страница ${page.page_number}`;

    const image = document.createElement('img');
    image.className = 'page-image';
    image.src = page.image;
    image.alt = `Иллюстрация для страницы ${page.page_number}`;
    image.onerror = function () { this.style.display = 'none'; };

    const text = document.createElement('div');
    text.className = 'page-text';
    text.textContent = page.text;

    pageDiv.append(pageNumber, image, text);
    return pageDiv;
}

function setLoading(loading) {
    const btnText = generateBtn.querySelector('.btn-text');
    const btnLoader = generateBtn.querySelector('.btn-loader');

    if (loading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline';
        generateBtn.disabled = true;
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        generateBtn.disabled = false;
    }
}

function setPdfLoading(loading) {
    if (loading) {
        downloadPdfBtn.innerHTML = '⏳ Создаем PDF...';
        downloadPdfBtn.disabled = true;
    } else {
        downloadPdfBtn.innerHTML = '📥 Скачать PDF';
        downloadPdfBtn.disabled = false;
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function hideBook() {
    bookContainer.style.display = 'none';
}