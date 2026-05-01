# Book_price_prediction
ML project. Authors: Запоточний Богдан, Коцеловська Марія
# -------------------------------------------------------
import pandas as pd
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')
df.info()
df #display it
# -------------------------------------------------------

# ------------------------------------------
# ------------------------------------------
# Лінійна регресія !!!
# ------------------------------------------
# ------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load your data
#df = pd.read_csv('train.csv')
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')
df.info()

df #display it

# Вибираємо всі ознаки, які впливають на ціну
features = ['Title', 'Author', 'Edition', 'Reviews', 'Ratings', 'Synopsis', 'Genre', 'BookCategory']
X = df[features]

# Цільова змінна (те, що прогнозуємо)
y = df['Price']

# Розділяємо дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Очищення та попередня обробка даних ---

# Очищення стовпця 'Price'
# Видаляємо нечислові символи (наприклад, коми) та перетворюємо на float
# df['Price'] = df['Price'].astype(str).str.extract('(\d+\.?\d*)', expand=False).str.replace(',', '', regex=False).astype(float)
df['Price'] = df['Price'].astype(str).str.extract('(\\d+\\.?\\d*)', expand=False).str.replace(',', '', regex=False).astype(float)

# Очищення стовпця 'Reviews'
# Витягуємо числову частину та перетворюємо на float (наприклад, "10 Reviews" -> 10.0)
# df['Reviews'] = df['Reviews'].astype(str).str.extract('(\d[\d,.]*)', expand=False).str.replace(',', '', regex=False).astype(float)
df['Reviews'] = df['Reviews'].astype(str).str.extract('(\\d[\\d,.]*)', expand=False).str.replace(',', '', regex=False).astype(float)

# Очищення стовпця 'Ratings'
# Витягуємо числову частину та перетворюємо на float (наприклад, "4.5 out of 5 stars" або "1,234 Ratings" -> 4.5 або 1234.0)
# df['Ratings'] = df['Ratings'].astype(str).str.extract('(\d[\d,.]*)', expand=False).str.replace(',', '', regex=False).astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract('(\\d[\\d,.]*)', expand=False).str.replace(',', '', regex=False).astype(float)

# Видаляємо рядки з NaN значеннями, які могли з'явитися під час очищення
df.dropna(subset=['Price', 'Reviews', 'Ratings'], inplace=True)

# Вибираємо числові ознаки (X) та цільову змінну (y) для лінійної регресії
# Для простої лінійної регресії починаємо лише з очищених числових ознак.
X = df[['Reviews', 'Ratings']]
y = df['Price']

# -----------------------------------------


# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація моделі лінійної регресії
model = LinearRegression()

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = model.predict(X_test)

# Оцінка моделі
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) score: {r2:.2f}")

# Виведення коефіцієнтів моделі
print("\nКоефіцієнти лінійної регресії:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Перетин (intercept): {model.intercept_:.2f}")

# ---------------------------------------
# Графік порівняння реальних цін із прогнозованими моделлю лінійної регресії
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Linear Model)')
plt.title('Actual vs. Predicted Prices (Linear Regression Model)')
plt.grid(True)
plt.show()


# ------------------------------------------
# ------------------------------------------
# Лінійна регресія 1 !!!
# ------------------------------------------
# ------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# --- Очищення та попередня обробка даних ---

# Очищення стовпця 'Price'
df['Price'] = df['Price'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False).str.replace(',', '', regex=False).astype(float)

# Очищення стовпця 'Reviews'
df['Reviews'] = df['Reviews'].astype(str).str.extract(r'(\d[\d,.]*)', expand=False).str.replace(',', '', regex=False).astype(float)

# Очищення стовпця 'Ratings'
df['Ratings'] = df['Ratings'].astype(str).str.extract(r'(\d[\d,.]*)', expand=False).str.replace(',', '', regex=False).astype(float)

# Видаляємо рядки з NaN значеннями, які могли з'явитися під час очищення
df.dropna(subset=['Price', 'Reviews', 'Ratings'], inplace=True)

# --- Target Encoding для категоріальних ознак ---

# Приклад без регуляризації (для демонстрації концепції, але не рекомендується для використання "як є"):
# Обчислюємо середню ціну для кожного автора
mean_prices_by_author = df.groupby('Author')['Price'].mean()
df['Author_Encoded'] = df['Author'].map(mean_prices_by_author)

# Для нових авторів (не в навчальному наборі) можна використовувати глобальне середнє
global_mean_price = df['Price'].mean()
df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)

# Аналогічно для 'Genre'
mean_prices_by_genre = df.groupby('Genre')['Price'].mean()
df['Genre_Encoded'] = df['Genre'].map(mean_prices_by_genre)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Вибираємо числові ознаки (X) та цільову змінну (y) для лінійної регресії
# Тепер включаємо закодовані ознаки
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація моделі лінійної регресії
model = LinearRegression()

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = model.predict(X_test)

# Оцінка моделі
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nОцінка точності моделі:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) score: {r2:.2f}")

# Виведення коефіцієнтів моделі
print("\nКоефіцієнти лінійної регресії:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Перетин (intercept): {model.intercept_:.2f}")

# ---------------------------------------
# Графік порівняння реальних цін із прогнозованими моделлю лінійної регресії
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Linear Model)')
plt.title('Actual vs. Predicted Prices (Linear Regression Model)')
plt.grid(True)
plt.show()

# ---------------------------------------
# Поліноміальна регресія !!!
# ---------------------------------------
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Розділення даних на тренувальний та тестовий набори (як і раніше)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling: Масштабування числових ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Створення поліноміальних ознак (ступінь 2)
# Це додасть ознаки, такі як Reviews^2, Ratings^2 та Reviews * Ratings
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled) # Apply poly features to scaled data
X_test_poly = poly.transform(X_test_scaled) # Apply poly features to scaled data

# Ініціалізація моделі лінійної регресії для поліноміальних ознак
poly_model = LinearRegression()

# Навчання моделі
poly_model.fit(X_train_poly, y_train)

# Прогнозування на тестовому наборі
y_pred_poly = poly_model.predict(X_test_poly)

# Оцінка моделі
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Mean Absolute Error (MAE) для поліноміальної регресії: {mae_poly:.2f}")
print(f"R-squared (R2) score для поліноміальної регресії: {r2_poly:.2f}")

# Виведення коефіцієнтів моделі
print("\nКоефіцієнти поліноміальної регресії:")
# Отримуємо назви нових поліноміальних ознак
poly_feature_names = poly.get_feature_names_out(numerical_features)
for feature, coef in zip(poly_feature_names, poly_model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Перетин (intercept): {poly_model.intercept_:.2f}")

# ---------------------------------------

# Графік порівняння реальних цін із прогнозованими моделлю поліноміальної регресії
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Polynomial Model)')
plt.title('Actual vs. Predicted Prices (Polynomial Regression Model)')
plt.grid(True)
plt.show()

# ---------------------------------------
# Поліноміальна регресія 1 !!!
# ---------------------------------------

import pandas as pd # Ensure pandas is imported for get_dummies
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np # Ensure numpy is imported for hstack

# Create the full feature set for X from the original dataframe 'df'
# This DataFrame will be split into training and testing sets
X_full_features = df[numerical_features + categorical_features_to_add]

# Розділення даних на тренувальний та тестовий набори (як і раніше)
# Now X_train_df and X_test_df will contain both numerical and categorical features
X_train_df, X_test_df, y_train, y_test = train_test_split(X_full_features, y, test_size=0.2, random_state=42)

# Feature Scaling: Масштабування числових ознак (only for numerical parts of X_train_df/X_test_df)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df[numerical_features])
X_test_scaled = scaler.transform(X_test_df[numerical_features])

# Створення поліноміальних ознак (ступінь 2)
# Це додасть ознаки, такі як Reviews^2, Ratings^2 та Reviews * Ratings
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled) # Apply poly features to scaled numerical data
X_test_poly = poly.transform(X_test_scaled) # Apply poly features to scaled numerical data

# One-hot encode categorical features for training data
X_train_categorical = pd.get_dummies(X_train_df[categorical_features_to_add], drop_first=True, dtype=float)
# One-hot encode categorical features for test data
X_test_categorical = pd.get_dummies(X_test_df[categorical_features_to_add], drop_first=True, dtype=float)

# Align columns - this is crucial if some categories are present in train but not test, or vice-versa
# Use X_train_categorical columns to reindex X_test_categorical
X_test_categorical = X_test_categorical.reindex(columns=X_train_categorical.columns, fill_value=0.0)

# Combine polynomial features with one-hot encoded categorical features
X_train_combined = np.hstack((X_train_poly, X_train_categorical.values))
X_test_combined = np.hstack((X_test_poly, X_test_categorical.values))

# Ініціалізація моделі лінійної регресії для поліноміальних ознак
poly_model = LinearRegression()

# Навчання моделі з комбінованими ознаками
poly_model.fit(X_train_combined, y_train)

# Прогнозування на тестовому наборі
y_pred_poly = poly_model.predict(X_test_combined)

# Оцінка моделі
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Mean Absolute Error (MAE) для поліноміальної регресії з категоріальними ознаками: {mae_poly:.2f}")
print(f"R-squared (R2) score для поліноміальної регресії з категоріальними ознаками: {r2_poly:.2f}")

# Виведення коефіцієнтів моделі
print("\nКоефіцієнти поліноміальної регресії з категоріальними ознаками:")
# Отримуємо назви нових поліноміальних ознак
poly_feature_names = poly.get_feature_names_out(numerical_features)
# Отримуємо назви one-hot закодованих категоріальних ознак
categorical_feature_names = X_train_categorical.columns.tolist()
# Комбінуємо всі назви ознак
all_feature_names = list(poly_feature_names) + categorical_feature_names

for feature, coef in zip(all_feature_names, poly_model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Перетин (intercept): {poly_model.intercept_:.2f}")

# ---------------------------------------

# Графік порівняння реальних цін із прогнозованими моделлю поліноміальної регресії
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='purple') # Змінено колір для розрізнення
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Polynomial + Categorical Model)')
plt.title('Actual vs. Predicted Prices (Polynomial Regression with Categorical Features)')
plt.grid(True)
plt.show()

# ---------------------------------------
# Гребнева регресія
# ---------------------------------------
from sklearn.metrics import mean_absolute_error, r2_score

# Predict on the test data using the Elastic Net model
y_pred_elastic = elastic_net_model.predict(X_test_all)

# Evaluate Elastic Net model
mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)

print("Elastic Net Model Evaluation:")
print(f"  Mean Absolute Error (MAE): {mae_elastic:.2f}")
print(f"  R-squared (R2): {r2_elastic:.2f}")

# Predict on the test data using the Ridge model
y_pred_ridge = ridge_model.predict(X_test_all)

# Evaluate Ridge model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Model Evaluation:")
print(f"  Mean Absolute Error (MAE): {mae_ridge:.2f}")
print(f"  R-squared (R2): {r2_ridge:.2f}")

# або з більшою точністю

from sklearn.linear_model import Ridge

# Ініціалізація моделі Ridge-регресії (з параметром регуляризації alpha)
# Alpha можна налаштовувати, почнемо з 1.0
model_ridge = Ridge(alpha=1.0)

# Навчання моделі Ridge-регресії на розширеному наборі даних
model_ridge.fit(X_train_all, y_train)

# Прогнозування на тестовому наборі
y_pred_ridge = model_ridge.predict(X_test_all)

# Оцінка моделі Ridge-регресії
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"\n--- Модель Ridge-регресії з розширеним набором ознак ---")
print(f"Mean Absolute Error (MAE) для Ridge: {mae_ridge:.2f}")
print(f"R-squared (R2) score для Ridge: {r2_ridge:.2f} ({r2_ridge*100:.2f}%)")

# Виведення коефіцієнтів нової моделі
print("\nКоефіцієнти Ridge-регресії для розширеної моделі (перші 10 для прикладу):")
for feature, coef in zip(X_all_features.columns[:10], model_ridge.coef_[:10]):
    print(f"{feature}: {coef:.2f}")
print(f"Перетин (intercept): {model_ridge.intercept_:.2f}")

# --------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Ridge Model)')
plt.title('Actual vs. Predicted Prices (Ridge Model)')
plt.grid(True)
plt.show()

# ---------------------------------------
# RandomForestRegressor !!!
# ---------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import re

df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# Ensure 'Price' column is numeric
df['Price'] = df['Price'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Preprocess 'Reviews' column to extract numerical part
df['Reviews'] = df['Reviews'].astype(str).apply(lambda x: float(re.search(r'\d+\.?\d*', x).group()) if re.search(r'\d+\.?\d*', x) else 0.0)

# Preprocess 'Ratings' column to extract numerical part
df['Ratings'] = df['Ratings'].astype(str).apply(lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

# Define features and target
features = ['Title', 'Author', 'Edition', 'Reviews', 'Ratings', 'Synopsis', 'Genre', 'BookCategory']
X = df[features]
y = df['Price']

# Apply Label Encoding to remaining categorical features in X
for column in ['Title', 'Author', 'Edition', 'Synopsis', 'Genre', 'BookCategory']:
    if column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#---------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='yellow')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (RandomForestRegressor Model)')
plt.title('Actual vs. Predicted Prices (RandomForestRegressor Regression Model)')
plt.grid(True)
plt.show()
#---------------------------------------
# Завдання: порівняти Лінійну регресію, Поліноміальну регресію, Гребеневу регресію та RandomForestRegressor
# Можливо додати ще якісь регресії
# Типи регресій: 
# Лінійна регресія (Linear Regression) - є, 
# Поліноміальна регресія (Polynomial Regression) - є, 
# Гребенева регресія (Ridge Regression) - є,
# Ласо-регресія (Lasso Regression), 
# Еластична мережева регресія (Elastic Net Regression), 
# SVR (Support Vector Regression), 
# Дерева рішень та випадкові ліси (Decision Trees and Random Forests for Regression), RandomForestRegressor - є, 
# Градієнтний бустинг (Gradient Boosting Machines, XGBoost, LightGBM, CatBoost)
#---------------------------------------

# --------------------------------------
# Додаток для опису даних !!!
# --------------------------------------
# Перевірка пропущених значень
missing_values = df.isnull().sum()
display(missing_values)

# ---------------------------------------
# Вивести описову статистику
display(df.describe())

# Ось описова статистика фрейму даних:

# Count (Кількість): 6237 записів для всіх стовпців, що підтверджує відсутність пропущених значень.
# Unique (Унікальних значень): Для стовпця 'Title' є 5568 унікальних назв, для 'Author' — 3679 унікальних авторів, а для 'Reviews' — 36 унікальних відгуків. Це вказує на велике різноманіття даних у текстових стовпцях.
# Top (Найчастіше значення):
# Найпопулярніша назва книги: 'A Game of Thrones (A Song of Ice and Fire)' (зустрічається 4 рази).
# Найпопулярніший автор: 'Agatha Christie' (зустрічається 69 разів).
# Найпопулярніше видання: 'Paperback,– 5 Oct 2017' (зустрічається 48 разів).
# Найчастіший відгук: '5.0 out of 5 stars' (зустрічається 1375 разів).
# Найпопулярніший рейтинг: '1 customer review' (зустрічається 1040 разів).
# Найчастіший жанр і категорія: 'Action & Adventure (Books)' і 'Action & Adventure' відповідно.
# Найчастіша ціна: 299 (зустрічається 108 разів).
# Freq (Частота): Показує кількість появ найчастішого значення для кожного стовпця.
# Ці дані дають гарне уявлення про розподіл і характеристики даних у вашому наборі.

# ----------------------------------------

# Перетворення стовпця 'Price' на числовий тип
df['Price'] = df['Price'].str.replace(',', '.', regex=False).astype(float)

# Розрахунок середньої ціни для кожної категорії книги
average_price_by_category = df.groupby('BookCategory')['Price'].mean().sort_values(ascending=False).reset_index()

# Візуалізація середньої ціни за категоріями
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='Price', y='BookCategory', data=average_price_by_category, palette='viridis')
plt.title('Середня ціна книги за категоріями')
plt.xlabel('Середня ціна')
plt.ylabel('Категорія книги')
plt.show()

# -----------------------------------------

# Розрахунок топ-10 найпопулярніших авторів за кількістю книг
top_10_authors = df['Author'].value_counts().head(10).reset_index()
top_10_authors.columns = ['Author', 'BookCount']

# Візуалізація топ-10 авторів
plt.figure(figsize=(12, 7))
sns.barplot(x='BookCount', y='Author', hue='Author', data=top_10_authors, palette='magma', legend=False)
plt.title('Топ-10 найпопулярніших авторів за кількістю книг')
plt.xlabel('Кількість книг')
plt.ylabel('Автор')
plt.show()

# -------------------------------------------

# Аналіз залежності ціни від кількості відгуків 
# Щоб візуалізувати залежність ціни книги від кількості відгуків, ми використаємо діаграму розсіювання. Це допоможе нам визначити, чи є якась кореляція між цими двома показниками. 

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Reviews', y='Price', data=df, alpha=0.6)
plt.title('Залежність ціни від кількості відгуків')
plt.xlabel('Кількість відгуків')
plt.ylabel('Ціна')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# або 

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Reviews', y='Price', data=df, alpha=0.6)
plt.title('Залежність ціни від кількості відгуків')
plt.xlabel('Кількість відгуків')
plt.ylabel('Ціна')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45) # Rotate x-axis labels for readability
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()

# ---------------------------------------------
# Створення цінових діапазонів
# Переконайтеся, що стовпець 'Price' є числовим, перетворюючи нечислові значення на NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Ви можете налаштувати `bins` (кількість діапазонів) та `labels` (мітки для діапазонів)
num_bins = 5 # Кількість діапазонів
df['Price_Range'] = pd.cut(df['Price'], bins=num_bins, labels=[f'Range {i+1}' for i in range(num_bins)])

# Вивід перших кількох рядків з новим стовпцем
display(df[['Price', 'Price_Range']].head())

# ----------------------------------------------

# Підрахунок кількості книг у кожному ціновому діапазоні
price_range_counts = df['Price_Range'].value_counts().sort_index()

# Візуалізація розподілу книг за ціновими діапазонами
plt.figure(figsize=(10, 6))
sns.barplot(x=price_range_counts.index, y=price_range_counts.values, hue=price_range_counts.index, palette='coolwarm', legend=False)
plt.title('Розподіл книг за ціновими діапазонами')
plt.xlabel('Ціновий діапазон')
plt.ylabel('Кількість книг')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
