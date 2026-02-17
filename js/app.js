class BookRecommendationApp {
    constructor() {
        this.currentUserId = null;
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupUserIdListeners();
            this.loadTop5();
            this.setupModalListener();
        });
    }

    setupUserIdListeners() {
        const userInput = document.getElementById('userIdInput');
        const loadBtn = document.getElementById('loadUserRecsBtn');

        if (loadBtn) {
            loadBtn.addEventListener('click', () => this.loadUserRecommendations());
        }

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.loadUserRecommendations();
            }
        });

        userInput.focus();
    }

    setupModalListener() {
        const modal = document.getElementById('bookModal');
        window.onclick = (event) => {
            if (event.target === modal) {
                UIManager.closeModal();
            }
        };
    }

    setupSearchListeners() {
        const searchInput = document.getElementById('searchInput');

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            if (query.length >= 2) {
                this.showSearchSuggestions(query);
            } else {
                document.getElementById('searchSuggestions').classList.remove('show');
            }
        });

        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchBooks();
            }
            if (e.key === 'Escape') {
                this.clearSearch();
            }
        });

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.clearSearch();
            }
        });
    }

    async loadTop5() {
        try {
            UIManager.showLoading('top5Container', 'Loading top 5 books...');
            const data = await BookAPI.getTop5();

            if (data.success) {
                const container = document.getElementById('top5Container');
                container.innerHTML = data.books.map(book => UIManager.createBookCard(book)).join('');
            } else {
                UIManager.showError('top5Container', 'Error loading top 5 books');
            }
        } catch (error) {
            console.error('Error loading top 5:', error);
            UIManager.showError('top5Container', 'Error loading top 5 books. Make sure the backend is running on port 5000.');
        }
    }

    changeUser() {
        this.currentUserId = null;
        document.getElementById('mainContent').style.display = 'none';
        document.getElementById('loginSection').style.display = 'block';
        document.getElementById('top5Section').style.display = 'block';
        document.getElementById('userIdInput').value = '';
        document.getElementById('userIdInput').focus();
    }

    async loadUserRecommendations() {
        const userId = document.getElementById('userIdInput').value.trim();

        if (!userId) {
            alert('Please enter a User ID');
            return;
        }

        if (isNaN(userId) || parseInt(userId) <= 0) {
            alert('User ID must be a valid positive number');
            return;
        }

        this.currentUserId = userId;
        const btn = document.getElementById('loadUserRecsBtn');
        btn.disabled = true;
        btn.textContent = 'Loading...';

        const infoDiv = document.getElementById('userRecInfo');
        const loadingDiv = document.getElementById('loadingUserRecs');
        const container = document.getElementById('userRecommendationsContainer');

        infoDiv.innerHTML = '';
        loadingDiv.style.display = 'block';
        container.innerHTML = '';

        try {
            const data = await BookAPI.getUserRecommendations(userId);
            
            btn.disabled = false;
            btn.textContent = 'Start';
            loadingDiv.style.display = 'none';

            if (data.success) {
                document.getElementById('loginSection').style.display = 'none';
                document.getElementById('top5Section').style.display = 'none';
                document.getElementById('mainContent').style.display = 'block';

                this.setupSearchListeners();

                if (data.recommendations && data.recommendations.length > 0) {
                    infoDiv.innerHTML = `
                        ✅ <strong>User ${userId}</strong> - ${data.user_stats.total_ratings} ratings found. 
                        Showing ${data.recommendations.length} personalized recommendations.
                    `;
                    container.innerHTML = data.recommendations.map(book => UIManager.createBookCard(book)).join('');
                    document.getElementById('userRecommendationsSection').style.display = 'block';
                } else {
                    infoDiv.innerHTML = `
                        ℹ️ User ${userId} has ${data.user_stats.total_ratings} ratings, 
                        but no recommendations could be generated. Try another user.
                    `;
                    document.getElementById('userRecommendationsSection').style.display = 'none';
                }
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            btn.disabled = false;
            btn.textContent = 'Start';
            console.error('Error loading user recommendations:', error);
            alert('Error loading recommendations. Please try again.');
        }
    }

    async showSearchSuggestions(query) {
        try {
            const data = await BookAPI.searchBooks(query, 8);

            if (data.success && data.results && data.results.length > 0) {
                const suggestions = document.getElementById('searchSuggestions');
                suggestions.innerHTML = data.results.map(book => `
                    <div class="suggestion-item" onclick="app.selectSuggestion('${book.isbn}', '${UIManager.escapeHtml(book.title)}')">
                        <div class="suggestion-title">${UIManager.escapeHtml(book.title)}</div>
                        <div class="suggestion-author">by ${UIManager.escapeHtml(book.author || 'Unknown')}</div>
                    </div>
                `).join('');
                suggestions.classList.add('show');
            } else {
                document.getElementById('searchSuggestions').classList.remove('show');
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            document.getElementById('searchSuggestions').classList.remove('show');
        }
    }

    selectSuggestion(isbn, title) {
        UIManager.showBookDetails(isbn);
        document.getElementById('searchInput').value = '';
        document.getElementById('searchSuggestions').classList.remove('show');
    }

    clearSearch() {
        document.getElementById('searchInput').value = '';
        document.getElementById('searchSuggestions').classList.remove('show');
        document.getElementById('searchResultsSection').style.display = 'none';
    }

    async searchBooks() {
        const query = document.getElementById('searchInput').value.trim();
        const suggestionsDiv = document.getElementById('searchSuggestions');
        suggestionsDiv.classList.remove('show');

        if (query.length < 2) {
            alert('Please enter at least 2 characters');
            return;
        }

        const top5Container = document.getElementById('top5Container');
        const top5Section = top5Container.closest('.section');
        if (top5Section) {
            top5Section.style.display = 'none';
        }

        const resultsSection = document.getElementById('searchResultsSection');
        const resultsContainer = document.getElementById('searchResultsContainer');

        UIManager.showLoading('searchResultsContainer', 'Searching...');
        resultsSection.style.display = 'block';

        try {
            const data = await BookAPI.searchBooks(query);

            if (data.success) {
                if (data.results && data.results.length > 0) {
                    resultsContainer.innerHTML = data.results.map(book => UIManager.createBookCard(book)).join('');
                } else {
                    UIManager.showEmpty('searchResultsContainer', `No books found matching "${UIManager.escapeHtml(query)}"`);
                }
            } else {
                UIManager.showError('searchResultsContainer', `Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error searching:', error);
            UIManager.showError('searchResultsContainer', 'Error searching books. Please try again.');
        }
    }
}

const app = new BookRecommendationApp();

function changeUser() {
    app.changeUser();
}

function loadUserRecommendations() {
    app.loadUserRecommendations();
}

function searchBooks() {
    app.searchBooks();
}

function clearSearch() {
    app.clearSearch();
}
