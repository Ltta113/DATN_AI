import pandas as pd
import numpy as np
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import warnings
import pickle
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
warnings.filterwarnings('ignore')


try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

class BookRecommendationSystem:
    def __init__(self, host='localhost', user='root', password='01012003', database='bookstore'):
        """
        Khởi tạo hệ thống gợi ý sách
        """
        self.db_config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
        
        
        self.books_df = None
        self.authors_df = None
        self.categories_df = None
        self.book_authors_df = None
        self.book_categories_df = None
        self.publishers_df = None
        
        
        self.content_similarity_matrix = None
        self.tfidf_vectorizer = None
        
        
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        
        self.load_data()
        
        
        self.preprocess_data()
        
    def connect_to_db(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Lỗi kết nối cơ sở dữ liệu: {e}")
            return None
    
    def load_data(self):
        conn = self.connect_to_db()
        if conn:
            try:
                self.books_df = pd.read_sql_query("""SELECT id, title, publisher_id, description, price FROM books WHERE status = 'active'""", conn)
                
                self.authors_df = pd.read_sql_query("""SELECT id, name FROM authors""", conn)
                
                self.book_authors_df = pd.read_sql_query("""SELECT book_id, author_id FROM book_authors""", conn)
                
                self.categories_df = pd.read_sql_query("""SELECT id, name FROM categories""", conn)
                
                self.book_categories_df = pd.read_sql_query("""SELECT book_id, category_id FROM book_categories""", conn)
                
                self.publishers_df = pd.read_sql_query("""SELECT id, name FROM publishers""", conn)
                
                print("Tải dữ liệu thành công!")
                
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu: {e}")
            finally:
                conn.close()
                
    def preprocess_data(self):

        if all(df is not None for df in [self.books_df, self.authors_df, self.categories_df, 
                                        self.book_authors_df, self.book_categories_df]):
            
            books_authors = pd.merge(self.book_authors_df, self.authors_df, left_on='author_id', right_on='id')
            books_authors_grouped = books_authors.groupby('book_id')['name'].apply(lambda x: ' '.join(x)).reset_index()
            self.books_df = pd.merge(self.books_df, books_authors_grouped, left_on='id', right_on='book_id', how='left')
            self.books_df.rename(columns={'name': 'authors'}, inplace=True)
            
            books_categories = pd.merge(self.book_categories_df, self.categories_df, left_on='category_id', right_on='id')
            books_categories_grouped = books_categories.groupby('book_id')['name'].apply(lambda x: ' '.join(x)).reset_index()
            self.books_df = pd.merge(self.books_df, books_categories_grouped, left_on='id', right_on='book_id', how='left')
            self.books_df.rename(columns={'name': 'categories'}, inplace=True)
            
            self.books_df = pd.merge(self.books_df, self.publishers_df[['id', 'name']], left_on='publisher_id', right_on='id', how='left')
            self.books_df.rename(columns={'name': 'publisher'}, inplace=True)
            
            self.books_df['authors'] = self.books_df['authors'].fillna('')
            self.books_df['categories'] = self.books_df['categories'].fillna('')
            self.books_df['publisher'] = self.books_df['publisher'].fillna('')
            
            self.books_df['content'] = (
                self.books_df['title'] + ' ' + 
                self.books_df['authors'] + ' ' + 
                self.books_df['categories'] + ' ' +
                self.books_df['publisher']
            )
            
            self.books_df['content'] = self.books_df['content'].apply(self._clean_text)
    
    def _clean_text(self, text):
        text = str(text).lower()
        
        text = re.sub(r'[^\w\s]', '', text)
        
        text = re.sub(r'\d+', '', text)
        
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        ps = PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        
        return text
    
    def build_content_based_model(self):

        if self.books_df is not None and 'content' in self.books_df.columns:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books_df['content'])
            
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            print("Đã xây dựng model Content-Based Filtering thành công!")
            return True
        else:
            print("Không thể xây dựng model Content-Based Filtering do thiếu dữ liệu!")
            return False

    def get_content_based_recommendations(self, book_id, top_n=10):

        if self.content_similarity_matrix is None:
            if not self.load_models():
                self.build_content_based_model()
            
        if book_id not in self.books_df['book_id_x'].values:
            print(f"Không tìm thấy sách với ID: {book_id}")
            return pd.DataFrame()
        
        book_idx = self.books_df[self.books_df['book_id_x'] == book_id].index[0]
        
        similarity_scores = list(enumerate(self.content_similarity_matrix[book_idx]))
        
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        top_books = similarity_scores[1:top_n+1]
        
        book_indices = [book[0] for book in top_books]
        recommended_books = self.books_df.iloc[book_indices][['id_x', 'title', 'authors', 'categories', 'description', 'price']]
        
        similarity_values = [book[1] for book in top_books]
        recommended_books['similarity_score'] = similarity_values
        
        return recommended_books
    
    def save_models(self):

        try:
            if self.content_similarity_matrix is not None:
                with open(os.path.join(self.models_dir, 'content_similarity_matrix.pkl'), 'wb') as f:
                    pickle.dump(self.content_similarity_matrix, f)
                
            if self.tfidf_vectorizer is not None:
                with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
            
            if self.books_df is not None:
                self.books_df.to_pickle(os.path.join(self.models_dir, 'books_df.pkl'))
                
            print("Đã lưu tất cả các mô hình thành công!")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")
            return False
    
    def load_models(self):
        try:
            if os.path.exists(os.path.join(self.models_dir, 'content_similarity_matrix.pkl')) and \
               os.path.exists(os.path.join(self.models_dir, 'books_df.pkl')) and \
               os.path.exists(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')):
                
                with open(os.path.join(self.models_dir, 'content_similarity_matrix.pkl'), 'rb') as f:
                    self.content_similarity_matrix = pickle.load(f)
                
                with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                self.books_df = pd.read_pickle(os.path.join(self.models_dir, 'books_df.pkl'))
                
                print("Đã tải tất cả các mô hình thành công!")
                return True
            else:
                print("Không tìm thấy các file mô hình đã lưu!")
                return False
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return False


class BookRecommendationAPI:
    def __init__(self, host='localhost', user='root', password='01012003', database='bookstore'):

        self.app = Flask(__name__)
        CORS(self.app)
        
        self.recommender = BookRecommendationSystem(host, user, password, database)
        
        if not self.recommender.load_models():
            self.recommender.build_content_based_model()
            self.recommender.save_models()
        
        self.define_routes()
    
    def define_routes(self):

        @self.app.route('/api/recommend/book/<int:book_id>', methods=['GET'])
        def recommend_by_book(book_id):
            try:
                
                top_n = request.args.get('top_n', default=10, type=int)
                
                recommendations = self.recommender.get_content_based_recommendations(book_id, top_n)
                
                if not recommendations.empty:
                    recommendations_json = recommendations.to_dict(orient='records')
                    return jsonify({
                        'success': True,
                        'recommendations': recommendations_json
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'Không tìm thấy sách với ID: {book_id} hoặc không có gợi ý phù hợp'
                    }), 404
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/retrain', methods=['POST'])
        def retrain_models():
            try:
                
                self.recommender.load_data()
                self.recommender.preprocess_data()
                
                self.recommender.build_content_based_model()
                
                self.recommender.save_models()
                
                return jsonify({
                    'success': True,
                    'message': 'Đã huấn luyện lại mô hình thành công!'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'message': 'Book Recommendation API is running'
            })
    
    def run(self, host='0.0.0.0', port=5000, debug=False):

        self.app.run(host=host, port=port, debug=debug)



if __name__ == "__main__":
    
    api_server = BookRecommendationAPI(
        host='localhost',
        user='root',
        password='01012003',
        database='bookstore'
    )
    api_server.run(host='0.0.0.0', port=5000, debug=True)