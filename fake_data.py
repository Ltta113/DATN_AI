import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import re
import unicodedata

# Hàm tạo slug từ title
def create_slug(title):
    # Chuyển về chữ thường
    title = title.lower()
    
    # Loại bỏ các ký tự đặc biệt và dấu cách
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8')  # Loại bỏ dấu tiếng Việt
    
    # Thay thế các ký tự không phải chữ cái và số thành dấu gạch ngang
    title = re.sub(r'[^a-z0-9\s-]', '', title)
    
    # Thay thế khoảng trắng thành dấu gạch ngang
    title = re.sub(r'[\s_-]+', '-', title)
    
    # Loại bỏ các dấu gạch ngang thừa ở đầu và cuối
    title = title.strip('-')
    
    return title

# Đọc file CSV vào DataFrame
df = pd.read_csv('E:\Workplace\Fake_Data\df_filtered.csv', encoding='utf-8-sig')

df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')  # Chuyển giá trị thành số, lỗi sẽ là NaN
df = df.dropna(subset=['original_price'])

df = df.drop_duplicates(subset=['title'], keep='first')

# Cấu hình kết nối MySQL
db_config = {
    'user': 'root',
    'password': '01012003',
    'host': 'localhost',
    'port': 3306,
    'database': 'bookstore'
}

# Kết nối MySQL
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

# Sử dụng SQLAlchemy để insert vào bảng chính và bảng trung gian
engine = create_engine(f'mysql+mysqlconnector://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}/{db_config["database"]}')

def check_duplicate_slug(slug):
    cursor.execute("SELECT COUNT(*) FROM books WHERE slug = %s", (slug,))
    result = cursor.fetchone()
    return result[0] > 0

# Hàm để kiểm tra và insert vào bảng category, author, publisher
def insert_unique_value(table, column, value):
    # Kiểm tra slug trước khi chèn
    slug = create_slug(value)
    cursor.execute(f"SELECT id FROM {table} WHERE {column} = %s OR slug = %s", (value, slug))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        trimmed_value = value.strip()
        cursor.execute(f"INSERT INTO {table} ({column}, slug) VALUES (%s, %s)", (trimmed_value, slug))
        connection.commit()
        return cursor.lastrowid

# Lặp qua từng dòng dữ liệu và insert vào các bảng
for index, row in df.iterrows():
    # Tạo slug từ title
    slug = create_slug(row['title'])
    
    if check_duplicate_slug(slug):
        # Nếu trùng lặp, thêm số vào cuối slug
        count = 1
        new_slug = f"{slug}-{count}"
        
        # Kiểm tra slug mới có trùng không, tăng số nếu có trùng
        while check_duplicate_slug(new_slug):
            count += 1
            new_slug = f"{slug}-{count}"
        
        slug = new_slug
    
    # Insert vào category
    category_id = insert_unique_value('categories', 'name', row['category'])
    publisher_id = insert_unique_value('publishers', 'name', row['manufacturer'])
    
    # Insert vào author (chú ý có thể có nhiều tác giả)
    authors = row['authors'].split("&") if isinstance(row['authors'], str) else []
    book_id = None
    if authors:
        # Insert vào bảng books
        cursor.execute(''' 
        INSERT INTO books (title, slug, price, stock, page_count, cover_image, publisher_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (row['title'], slug, row['original_price'], row['quantity'], row['pages'], row['cover_link'], publisher_id))
        connection.commit()
        book_id = cursor.lastrowid
        
        # Insert vào authors
        for author in authors:
            author_id = insert_unique_value('authors', 'name', author.strip())
            
            # Insert vào bảng book_authors
            cursor.execute('INSERT INTO book_authors (book_id, author_id) VALUES (%s, %s)', (book_id, author_id))
            connection.commit()
        
        # Insert vào bảng book_categories
        cursor.execute('INSERT INTO book_categories (book_id, category_id) VALUES (%s, %s)', (book_id, category_id))
        connection.commit()

# Đóng kết nối
connection.close()

print("Dữ liệu đã được import thành công vào các bảng MySQL!")
