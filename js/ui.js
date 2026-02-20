class UIManager {
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static createBookCard(book) {
        const bookCover = book.image_url
            ? `<img src="${book.image_url}" alt="${this.escapeHtml(book.title)}" class="book-cover-img" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 150%22%3E%3Crect fill=%22%23ddd%22 width=%22100%22 height=%22150%22/%3E%3Ctext x=%2250%22 y=%2275%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 font-size=%2224%22 fill=%22%23999%22%3E📖%3C/text%3E%3C/svg%3E'">`
            : `<div class="book-cover-placeholder">📖</div>`;

        return `
            <div class="book-card" onclick="UIManager.showBookDetails('${book.isbn}')">
                <div class="book-cover">
                    ${bookCover}
                </div>
                <div class="book-info">
                    <div class="book-title">${this.escapeHtml(book.title)}</div>
                    <div class="book-author">${this.escapeHtml(book.author || 'Unknown Author')}</div>
                    <div class="book-year">${book.year || 'N/A'}</div>
                </div>
            </div>
        `;
    }

    static createRecommendationCards(recommendations) {
        return recommendations.map(rec => {
            const recCover = rec.image_url
                ? `<img src="${rec.image_url}" alt="${this.escapeHtml(rec.title)}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 150%22%3E%3Crect fill=%22%23667eea%22 width=%22100%22 height=%22150%22/%3E%3Ctext x=%2250%22 y=%2775%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 font-size=%2224%22 fill=%22white%22%3E📖%3C/text%3E%3C/svg%3E'">`
                : `<div class="rec-book-cover-placeholder">📖</div>`;
            return `
                <div class="rec-book-card" onclick="UIManager.showBookDetails('${rec.isbn}')">
                    <div class="rec-book-cover">
                        ${recCover}
                    </div>
                    <div class="rec-book-info">
                        <div class="rec-book-title">${this.escapeHtml(rec.title)}</div>
                        <div class="rec-book-author">${this.escapeHtml(rec.author || 'Unknown')}</div>
                        <div class="book-year" style="margin-top: auto; font-size: 0.8em; color: #999;">${rec.year || 'N/A'}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    static async showBookDetails(isbn) {
        const modal = document.getElementById('bookModal');
        const content = document.getElementById('bookDetailsContent');

        content.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading book details...</p></div>';
        modal.classList.add('show');

        try {
            const data = await BookAPI.getBookDetails(isbn);

            if (data.success) {
                const book = data.book;
                const recommendations = data.recommendations || [];

                let recommendationsHtml = '';

                if (recommendations.length > 0) {
                    recommendationsHtml = `
                        <div class="recommendations-section">
                            <h3>Similar Books You Might Like</h3>
                            <div class="rec-type-info">
                                Recommendations based on book description similarity (TF-IDF)
                            </div>
                            <div class="recommendations-grid">
                                ${this.createRecommendationCards(recommendations)}
                            </div>
                        </div>
                    `;
                }

                content.innerHTML = `
                    <div class="book-details">
                        <div class="book-details-cover">
                            ${book.image_url
                        ? `<img src="${book.image_url}" alt="${this.escapeHtml(book.title)}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 150%22%3E%3Crect fill=%22%23667eea%22 width=%22100%22 height=%22150%22/%3E%3Ctext x=%2250%22 y=%2775%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 font-size=%2240%22 fill=%22white%22%3E📖%3C/text%3E%3C/svg%3E'">`
                        : `<div style="width: 100%; height: 300px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 3em; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.2);">📖</div>`
                    }
                        </div>
                        <div class="book-details-info">
                            <h2 class="book-detail-title">${this.escapeHtml(book.title)}</h2>
                            <div class="book-detail-author">by ${this.escapeHtml(book.author || 'Unknown Author')}</div>
                            <div class="book-detail-meta">
                                ${book.publisher ? `<strong>Publisher:</strong> ${this.escapeHtml(book.publisher)}<br>` : ''}
                                ${book.year ? `<strong>Year:</strong> ${book.year}<br>` : ''}
                                <strong>ISBN:</strong> ${book.isbn}
                            </div>
                            ${book.description ? `<div class="book-description">${this.escapeHtml(book.description)}</div>` : ''}
                        </div>
                    </div>
                    ${recommendationsHtml}
                `;
            } else {
                content.innerHTML = `<div class="error-message">Error loading book details: ${data.error}</div>`;
            }
        } catch (error) {
            console.error('Error showing book details:', error);
            content.innerHTML = `<div class="error-message">Error loading book details. Please try again.</div>`;
        }
    }

    static closeModal() {
        document.getElementById('bookModal').classList.remove('show');
    }

    static showLoading(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="loading"><div class="spinner"></div><p>${message}</p></div>`;
        }
    }

    static showError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="error-message" style="grid-column: 1/-1;">${message}</div>`;
        }
    }

    static showEmpty(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="empty-state" style="grid-column: 1/-1;"><div class="empty-state-icon">📭</div><p>${message}</p></div>`;
        }
    }
}

window.UIManager = UIManager;
