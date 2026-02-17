const API_BASE = 'http://localhost:5000/api';

class BookAPI {
    static async getTop5() {
        try {
            const response = await fetch(`${API_BASE}/top5`);
            return await response.json();
        } catch (error) {
            console.error('Error loading top 5:', error);
            throw error;
        }
    }

    static async searchBooks(query, limit = 20) {
        try {
            const response = await fetch(
                `${API_BASE}/search?q=${encodeURIComponent(query)}&limit=${limit}`
            );
            return await response.json();
        } catch (error) {
            console.error('Error searching books:', error);
            throw error;
        }
    }

    static async getBookDetails(isbn) {
        try {
            const response = await fetch(`${API_BASE}/book/${isbn}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching book details:', error);
            throw error;
        }
    }

    static async getUserRecommendations(userId, topN = 10) {
        try {
            const response = await fetch(
                `${API_BASE}/user/${userId}/recommendations?top_n=${topN}`
            );
            return await response.json();
        } catch (error) {
            console.error('Error fetching user recommendations:', error);
            throw error;
        }
    }

    static async getHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            return await response.json();
        } catch (error) {
            console.error('Error checking health:', error);
            throw error;
        }
    }
}
