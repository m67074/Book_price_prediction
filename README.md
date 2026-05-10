# Book_price_prediction
ML project. Authors: Запоточний Богдан, Коцеловська Марія
# ==============
# Імпорт бібліотеки NumPy для числових операцій
import numpy as np
# Імпорт бібліотеки Pandas для роботи з табличними даними
import pandas as pd
# Імпорт метрик для оцінки моделі
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# Імпорт моделей LinearRegression, ElasticNet і Ridge
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge 
# Імпорт функції для розділення даних на тренувальні та тестові набори
from sklearn.model_selection import train_test_split
# Імпорт ансамблевих моделей регресії
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Імпорт інструментів для попередньої обробки даних
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
# Імпорт модуля для роботи з регулярними виразами
import re
# Імпорт бібліотеки Matplotlib для візуалізації даних
import matplotlib.pyplot as plt
# Імпорт модуля для роботи з потоками вводу/виводу
import io
# Імпорт модуля для кодування/декодування Base64
import base64
# Імпорт модуля для відображення об'єктів в IPython
import IPython.display as display

# Завантаження даних
# Завантаження даних з Excel
#df = pd.read_excel('train.xlsx')
# Завантаження даних з CSV-файлу
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# Виведення інформації про DataFrame (типи даних, кількість ненульових значень)
df.info()

import seaborn as sns

# Гістограма розподілу цін
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Розподіл цін на книги')
plt.xlabel('Ціна')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# Визначення мінімальної та максимальної ціни
min_price = df['Price'].min()
max_price = df['Price'].max()

print(f"Мінімальна ціна: {min_price:.2f}")
print(f"Максимальна ціна: {max_price:.2f}")

# Оцінки моделей
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # Додаємо невелике значення до y_true, щоб уникнути ділення на нуль при обчисленні MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    accuracy = 100 - mape
    print(f"\n{model_name} Model Evaluation:") # Adjusted print statement to be generic
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  R-squared (R2): {r2:.2f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return mae, mse, mape, r2, accuracy

# Графік порівняння реальних цін із прогнозованими моделлю 
def plot_predictions(y_true, y_pred, model_title, plot_color):
    """
    Generates a scatter plot comparing actual vs. predicted values for a given model.

    Args:
        y_true (pd.Series): Actual values.
        y_pred (np.array): Predicted values.
        model_title (str): Title for the plot and Y-axis label.
        plot_color (str): Color for the scatter points.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.7, color=plot_color)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r', linewidth=2)
    plt.xlabel('Actual Prices')
    plt.ylabel(f'Predicted Prices ({model_title})')
    plt.title(f'Actual vs. Predicted Prices ({model_title})')
    plt.grid(True)
    plt.show()

# Визначення точок даних з найбільшими помилками прогнозування
def display_top_errors(y_true, y_pred, model_name, top_n=10):
    """
    Identifies and displays the top N data points with the largest prediction errors.

    Args:
        y_true (pd.Series): Actual values.
        y_pred (np.array): Predicted values.
        model_name (str): Name of the model for display purposes.
        top_n (int): The number of top errors to display.
    """
    errors_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred, 'Absolute_Error': np.abs(y_true - y_pred)})
    errors_df = errors_df.sort_values(by='Absolute_Error', ascending=False)
    print(f"\nTop {top_n} data points with the largest prediction errors for {model_name}:")
    display.display(errors_df.head(top_n))

# --- Початок доданої попередньої обробки даних для самостійного виконання ---
# Перетворення стовпця 'Price' на числовий формат, NaN для некоректних значень
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Очищення стовпців 'Reviews' та 'Ratings' для вилучення числових значень
# df['Reviews'] = df['Reviews'].astype(str).str.extract('(\\d+\\.?\\d*)').astype(float)
# df['Ratings'] = df['Ratings'].astype(str).str.extract('(\\d+)').astype(float)
# df['Reviews'] = df['Reviews'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
# df['Ratings'] = df['Ratings'].astype(str).str.extract('(\d+)').astype(float)
df['Reviews'] = df['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Calculate mean prices by Author and Genre for target encoding
# Note: Applying target encoding on the full dataframe before splitting can lead to data leakage.
# For a more robust ML pipeline, target encoding should ideally be calculated on training data only.
# However, to replicate the kernel state's 'Author_Encoded' and 'Genre_Encoded' as floats,
# we apply it here for self-containment.

# Розрахунок глобального середнього значення ціни для заповнення пропущених значень
global_mean_price = df['Price'].mean() 

# Розрахунок середніх цін за автором та жанром для кодування
mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')

# Створення нових стовпців для закодованих автора та жанру на основі середніх цін
df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre

# Заповнення будь-яких пропущених значень у закодованих стовпцях глобальним середнім значенням ціни
df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Визначення ознак (X) та цільової змінної (y) за допомогою оброблених даних
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Видалення рядків, де y (Price) є NaN, оскільки вони не можуть бути використані для навчання
# Забезпечення однакових індексів X та y після видалення NaN
y.dropna(inplace=True)
X = X.loc[y.index]

# Розділення даних на тренувальний та тестовий набори 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначення числових та категоріальних ознак для подальшої обробки
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

# Масштабування числових ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Об'єднання масштабованих числових ознак з категоріальними
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))
# --- Кінець доданої попередньої обробки даних для самостійного виконання ---

# Ініціалізація моделі лінійної регресії
model = LinearRegression()

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = model.predict(X_test)

# Оцінка моделі
# Розрахунок середньої абсолютної помилки
mae = mean_absolute_error(y_test, y_pred)
# Розрахунок коефіцієнта детермінації R2
r2 = r2_score(y_test, y_pred)
# Розрахунок середньої квадратичної помилки
mse = mean_squared_error(y_test, y_pred)
# Розрахунок середньої абсолютної відсоткової помилки (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# Розрахунок точності у відсотках
accuracy_in_percent = 100 - mape

print("\nОцінка точності моделі:")
print(f"Mean Absolute Error (MAE): {mae:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"R-squared (R2) score: {r2:.2f}")
print(f"Точність: {accuracy_in_percent:.2f}%") # Те саме число у відсотках

# Графік порівняння реальних цін із прогнозованими моделлю лінійної регресії
# Встановлення розміру графіка
plt.figure(figsize=(10, 10))
# Побудова точкового графіка: реальні vs прогнозовані ціни
plt.scatter(y_test, y_pred, alpha=0.7, color='orange')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
# Підпис осі X
plt.xlabel('Actual Prices')
# Підпис осі Y
plt.ylabel('Predicted Prices (Linear Model)')
# Заголовок графіка
plt.title('Actual vs. Predicted Prices (Linear Regression Model)')
# Увімкнення сітки
plt.grid(True)
# Відображення графіка
plt.show()

# Identify data points with the largest prediction errors
# Створення DataFrame з помилками
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Absolute_Error': np.abs(y_test - y_pred)})
# Сортування за абсолютною помилкою
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# Ініціалізація моделі лінійної регресії для поліноміальних ознак
poly_model = LinearRegression()

# Навчання моделі з комбінованими ознаками
poly_model.fit(X_train_combined, y_train)

# Прогнозування на тестовому наборі
y_pred_poly = poly_model.predict(X_test_combined)

# Оцінка моделі
# Середня абсолютна помилка
mae_poly = mean_absolute_error(y_test, y_pred_poly)
# Коефіцієнт R2
r2_poly = r2_score(y_test, y_pred_poly)
# Середня квадратична помилка
mse_poly = mean_squared_error(y_test, y_pred_poly)
# Розрахунок MAPE для поліноміальної моделі
mape_poly = np.mean(np.abs((y_test - y_pred_poly) / (y_test + 1e-10))) * 100
# Точність поліноміальної моделі
accuracy_poly = 100 - mape_poly

print(f"Mean Absolute Error (MAE): {mae_poly:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_poly:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_poly:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"R-squared (R2) score: {r2_poly:.2f}")
print(f"Точність моделі: {accuracy_poly:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='purple') 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Polynomial Regression)')
plt.title('Actual vs. Predicted Prices (Polynomial Regression)')
plt.grid(True)
plt.show()

# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_poly, 'Absolute_Error': np.abs(y_test - y_pred_poly)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# Initialize and train Elastic Net model
elastic_net_model = ElasticNet(random_state=42)
elastic_net_model.fit(X_train_combined, y_train)

# Initialize and train Ridge model
ridge_net_model = Ridge(random_state=42)
ridge_net_model.fit(X_train_combined, y_train)

# Predict on the test data using the Elastic Net model
y_pred_elastic = elastic_net_model.predict(X_test_combined)

# Predict on the test data using the Ridge model
y_pred_ridge = ridge_net_model.predict(X_test_combined)

# Evaluate Elastic Net model
mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)

mape_elastic = np.mean(np.abs((y_test - y_pred_elastic) / (y_test + 1e-10))) * 100
accuracy_elastic = 100 - mape_elastic

print("Elastic Net Model Evaluation:")
print(f"  Mean Absolute Error (MAE): {mae_elastic:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_elastic:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_elastic:.2f}%") # Середня абсолютна відсоткова помилка
print(f"  R-squared (R2): {r2_elastic:.2f}")
print(f"Точність моделі: {accuracy_elastic:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_elastic, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Elastic Net Model)')
plt.title('Actual vs. Predicted Prices (Elastic Net Model)')
plt.grid(True)
plt.show()

# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_elastic, 'Absolute_Error': np.abs(y_test - y_pred_elastic)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))


# Evaluate Ridge model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

mape_ridge = np.mean(np.abs((y_test - y_pred_ridge) / (y_test + 1e-10))) * 100
accuracy_ridge = 100 - mape_ridge

print("\nRidge Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_ridge:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_ridge:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_ridge:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"  R-squared (R2): {r2_ridge:.2f}")
print(f"Точність моделі: {accuracy_ridge:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_ridge, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Ridge Model)')
plt.title('Actual vs. Predicted Prices (Ridge Model)')
plt.grid(True)
plt.show()

# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ridge, 'Absolute_Error': np.abs(y_test - y_pred_ridge)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_rf = model.predict(X_test)

# Оцінка моделі RandomForestRegressor
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

mape_rf = np.mean(np.abs((y_test - y_pred_rf) / (y_test + 1e-10))) * 100
accuracy_rf = 100 - mape_rf

print("\nRandomForestRegressor Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_rf:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_rf:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"  R-squared (R2): {r2_rf:.2f}")
print(f"Точність моделі: {accuracy_rf:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='yellow')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (RandomForestRegressor Model)')
plt.title('Actual vs. Predicted Prices (RandomForestRegressor Regression Model)')
plt.grid(True)
plt.show()

# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf, 'Absolute_Error': np.abs(y_test - y_pred_rf)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# Train the GradientBoostingRegressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_gbr = gbr_model.predict(X_test)

# Evaluate the GradientBoostingRegressor model
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)

mape_gbr = np.mean(np.abs((y_test - y_pred_gbr) / (y_test + 1e-10))) * 100
accuracy_gbr = 100 - mape_gbr

print("\nGradientBoostingRegressor Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_gbr:.2f}")
print(f"Mean Squared Error (MSE): {mse_gbr:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_gbr:.2f}%")
print(f"  R-squared (R2): {r2_gbr:.2f}")
print(f"Точність моделі: {accuracy_gbr:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_gbr, alpha=0.7, color='brown')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (GradientBoostingRegressor Model)')
plt.title('Actual vs. Predicted Prices (GradientBoostingRegressor Model)')
plt.grid(True)
plt.show()

# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gbr, 'Absolute_Error': np.abs(y_test - y_pred_gbr)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# Порівняння R2-оцінок моделей

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='R2 Score', hue='Model', data=results.sort_values(by='R2 Score', ascending=False), palette='viridis', legend=False)
plt.title('Порівняння R2-оцінок моделей')
plt.xlabel('Модель')
plt.ylabel('R2 Score')
plt.ylim(0, 1) # R2 score ranges from 0 to 1
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Порівняння R2 Score Accuracy MAE MSE MAPE 

import seaborn as sns
import matplotlib.pyplot as plt

# List of metrics to plot, along with their y-axis label, title, and sorting order
metrics_to_plot = [
    {'col': 'R2 Score', 'ylabel': 'R2 Score', 'title': 'Порівняння R2-оцінок моделей', 'ascending': False, 'ylim': (0, 1)},
    {'col': 'Accuracy (%)', 'ylabel': 'Точність (%)', 'title': 'Порівняння Точності Моделей', 'ascending': False, 'ylim': (0, 100)},
    {'col': 'MAE', 'ylabel': 'MAE', 'title': 'Порівняння MAE Моделей', 'ascending': True, 'ylim': (0, None)}, # MAE: lower is better
    {'col': 'MSE', 'ylabel': 'MSE', 'title': 'Порівняння MSE Моделей', 'ascending': True, 'ylim': (0, None)}, # MSE: lower is better
    {'col': 'MAPE (%)', 'ylabel': 'MAPE (%)', 'title': 'Порівняння MAPE Моделей', 'ascending': True, 'ylim': (0, None)} # MAPE: lower is better
]

for metric_info in metrics_to_plot:
    metric_col = metric_info['col']
    ylabel = metric_info['ylabel']
    title = metric_info['title']
    ascending_sort = metric_info['ascending']
    ylim_val = metric_info['ylim']

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y=metric_col, hue='Model', data=results.sort_values(by=metric_col, ascending=ascending_sort), palette='viridis', legend=False)
    plt.title(title)
    plt.xlabel('Модель')
    plt.ylabel(ylabel)
    if ylim_val[1] is not None: # Apply ylim only if a max value is provided
        plt.ylim(ylim_val)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Загальна таблиця для всіх моделей

results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Polynomial Regression',
        'Elastic Net',
        'Ridge',
        'RandomForestRegressor',
        'GradientBoostingRegressor'
    ],
    'MAE': [
        mae,
        mae_poly,
        mae_elastic,
        mae_ridge,
        mae_rf,
        mae_gbr
    ],
    'MSE': [
        mse,
        mse_poly,
        mse_elastic,
        mse_ridge,
        mse_rf,
        mse_gbr
    ],
    'MAPE (%)': [
        mape,
        mape_poly,
        mape_elastic,
        mape_ridge,
        mape_rf,
        mape_gbr
    ],
    'R2 Score': [
        r2,
        r2_poly,
        r2_elastic,
        r2_ridge,
        r2_rf,
        r2_gbr
    ],
    'Accuracy (%)': [
        accuracy_in_percent,
        accuracy_poly,
        accuracy_elastic,
        accuracy_ridge,
        accuracy_rf,
        accuracy_gbr
]
})

# Sort by R2 score for better comparison
display.display(results.sort_values(by='R2 Score', ascending=False))
# ==============

import numpy as np # Імпорт бібліотеки NumPy для числових операцій
import pandas as pd # Імпорт бібліотеки Pandas для роботи з табличними даними (DataFrame)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error # Імпорт метрик для оцінки моделі
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge # Імпорт моделей лінійної регресії, ElasticNet і Ridge
from sklearn.model_selection import train_test_split # Імпорт функції для розділення даних на тренувальні та тестові набори
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Імпорт ансамблевих моделей регресії
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler # Імпорт інструментів для попередньої обробки даних
import re # Імпорт модуля для роботи з регулярними виразами
import matplotlib.pyplot as plt # Імпорт бібліотеки Matplotlib для візуалізації даних
import io # Імпорт модуля для роботи з потоками вводу/виводу
import base64 # Імпорт модуля для кодування/декодування Base64
import IPython.display as display # Імпорт модуля для відображення об'єктів в IPython

# Завантаження даних
# df = pd.read_excel('train.xlsx') # Закоментований рядок для завантаження даних з Excel
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';') # Завантаження даних з CSV-файлу

df.info() # Виведення інформації про DataFrame (типи даних, кількість ненульових значень)

print(f"Оригінальний розмір DataFrame: {len(df)} рядків")
print(f"Розмір DataFrame після видалення викидів: {len(df_no_outliers)} рядків")
print(f"Кількість видалених викидів: {len(df) - len(df_no_outliers)}")

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # Додаємо невелике значення до y_true, щоб уникнути ділення на нуль при обчисленні MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100 
    r2 = r2_score(y_true, y_pred)
    accuracy = 100 - mape
    print(f"\n{model_name} Model Evaluation (Cleaned Data):")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  R-squared (R2): {r2:.2f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return mae, mse, mape, r2, accuracy

# --- Початок доданої попередньої обробки даних для самостійного виконання ---
# Перетворення стовпця 'Price' на числовий формат, NaN для некоректних значень
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Очищення стовпців 'Reviews' та 'Ratings' для вилучення числових значень
df['Reviews'] = df['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Розрахунок глобального середнього значення ціни для заповнення пропущених значень
global_mean_price = df['Price'].mean()

# Розрахунок середніх цін за автором та жанром для кодування
mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')

# Створення нових стовпців для закодованих автора та жанру на основі середніх цін
df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre

# Заповнення будь-яких пропущених значень у закодованих стовпцях глобальним середнім значенням ціни
df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Визначення ознак (X) та цільової змінної (y) за допомогою оброблених даних
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Видалення рядків, де y (Price) є NaN, оскільки вони не можуть бути використані для навчання
# Забезпечення однакових індексів X та y після видалення NaN
y.dropna(inplace=True)
X = X.loc[y.index]

# Розділення даних на тренувальний та тестовий набори (як і раніше)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначення числових та категоріальних ознак для подальшої обробки
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

# Масштабування числових ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features]) # Масштабування тренувальних даних
X_test_scaled = scaler.transform(X_test[numerical_features]) # Масштабування тестових даних

# Об'єднання масштабованих числових ознак з категоріальними
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))
# --- Кінець доданої попередньої обробки даних для самостійного виконання ---

# Ініціалізація моделі лінійної регресії
model = LinearRegression()

# Навчання моделі на тренувальних даних
model.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = model.predict(X_test)

# Оцінка моделі Linear Regression
evaluate_model(y_test, y_pred, "Linear Regression")

# Графік порівняння реальних цін із прогнозованими моделлю лінійної регресії
plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
plt.scatter(y_test, y_pred, alpha=0.7, color='orange') # Побудова точкового графіка: реальні vs прогнозовані ціни
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Побудова ідеальної лінії прогнозу
plt.xlabel('Actual Prices') # Підпис осі X
plt.ylabel('Predicted Prices (Linear Model)') # Підпис осі Y
plt.title('Actual vs. Predicted Prices (Linear Regression Model)') # Заголовок графіка
plt.grid(True) # Увімкнення сітки
plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Absolute_Error': np.abs(y_test - y_pred)}) # Створення DataFrame з помилками
errors = errors.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою

print("\nTop 10 data points with the largest prediction errors:") # Виведення заголовка
display.display(errors.head(10)) # Відображення 10 найбільших помилок

# Ініціалізація моделі лінійної регресії для поліноміальних ознак
poly_model = LinearRegression()

# Навчання моделі з комбінованими ознаками (включаючи масштабовані числові та категоріальні)
poly_model.fit(X_train_combined, y_train)

# Прогнозування на тестовому наборі за допомогою поліноміальної моделі
y_pred_poly = poly_model.predict(X_test_combined)

# Оцінка моделі Polynomial Regression
evaluate_model(y_test, y_pred_poly, "Polynomial Regression")

plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='purple') # Побудова точкового графіка
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ідеальна лінія прогнозу
plt.xlabel('Actual Prices') # Підпис осі X
plt.ylabel('Predicted Prices (Polynomial Regression)') # Підпис осі Y
plt.title('Actual vs. Predicted Prices (Polynomial Regression)') # Заголовок графіка
plt.grid(True) # Увімкнення сітки
plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування для поліноміальної моделі
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_poly, 'Absolute_Error': np.abs(y_test - y_pred_poly)}) # Створення DataFrame з помилками
errors = errors.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою

print("\nTop 10 data points with the largest prediction errors:") # Виведення заголовка
display.display(errors.head(10)) # Відображення 10 найбільших помилок

# Ініціалізація та навчання моделі Elastic Net
elastic_net_model = ElasticNet(random_state=42) # Створення екземпляра ElasticNet з фіксованим random_state
elastic_net_model.fit(X_train_combined, y_train) # Навчання моделі Elastic Net

# Ініціалізація та навчання моделі Ridge
ridge_net_model = Ridge(random_state=42) # Створення екземпляра Ridge з фіксованим random_state
ridge_net_model.fit(X_train_combined, y_train) # Навчання моделі Ridge

# Прогнозування на тестових даних за допомогою моделі Elastic Net
y_pred_elastic = elastic_net_model.predict(X_test_combined)

# Прогнозування на тестових даних за допомогою моделі Ridge
y_pred_ridge = ridge_net_model.predict(X_test_combined)

# Оцінка моделі Elastic Net
mae_elastic = mean_absolute_error(y_test, y_pred_elastic) # Середня абсолютна помилка
r2_elastic = r2_score(y_test, y_pred_elastic) # Коефіцієнт R2
mse_elastic = mean_squared_error(y_test, y_pred_elastic) # Середня квадратична помилка

evaluate_model(y_test, y_pred_elastic, "Elastic Net")

plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
plt.scatter(y_test, y_pred_elastic, alpha=0.7) # Побудова точкового графіка
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ідеальна лінія прогнозу
plt.xlabel('Actual Prices') # Підпис осі X
plt.ylabel('Predicted Prices (Elastic Net Model)') # Підпис осі Y
plt.title('Actual vs. Predicted Prices (Elastic Net Model)') # Заголовок графіка
plt.grid(True) # Увімкнення сітки
plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування для Elastic Net
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_elastic, 'Absolute_Error': np.abs(y_test - y_pred_elastic)}) # Створення DataFrame з помилками
errors = errors.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою

print("\nTop 10 data points with the largest prediction errors:") # Виведення заголовка
display.display(errors.head(10)) # Відображення 10 найбільших помилок


evaluate_model(y_test, y_pred_ridge, "Ridge Regression")

plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
plt.scatter(y_test, y_pred_ridge, alpha=0.7, color='green') # Побудова точкового графіка
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ідеальна лінія прогнозу
plt.xlabel('Actual Prices') # Підпис осі X
plt.ylabel('Predicted Prices (Ridge Model)') # Підпис осі Y
plt.title('Actual vs. Predicted Prices (Ridge Model)') # Заголовок графіка
plt.grid(True) # Увімкнення сітки
plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування для Ridge
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ridge, 'Absolute_Error': np.abs(y_test - y_pred_ridge)}) # Створення DataFrame з помилками
errors = errors.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою

print("\nTop 10 data points with the largest prediction errors:") # Виведення заголовка
display.display(errors.head(10)) # Відображення 10 найбільших помилок

# Ініціалізація та навчання моделі RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42) # Створення екземпляра RandomForestRegressor
model.fit(X_train, y_train) # Навчання моделі

# Здійснення прогнозів за допомогою навченої моделі
y_pred_rf = model.predict(X_test)

# Оцінка моделі RandomForestRegressor
evaluate_model(y_test, y_pred_rf, "RandomForestRegressor")

plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='yellow') # Побудова точкового графіка
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ідеальна лінія прогнозу
plt.xlabel('Actual Prices') # Підпис осі X
plt.ylabel('Predicted Prices (RandomForestRegressor Model)') # Підпис осі Y
plt.title('Actual vs. Predicted Prices (RandomForestRegressor Regression Model)') # Заголовок графіка
plt.grid(True) # Увімкнення сітки
plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування для RandomForestRegressor
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf, 'Absolute_Error': np.abs(y_test - y_pred_rf)}) # Створення DataFrame з помилками
errors = errors.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою

print("\nTop 10 data points with the largest prediction errors:") # Виведення заголовка
display.display(errors.head(10)) # Відображення 10 найбільших помилок

# Ініціалізація та навчання моделі GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Створення екземпляра GradientBoostingRegressor
gbr_model.fit(X_train, y_train) # Навчання моделі

# Здійснення прогнозів за допомогою навченої моделі
y_pred_gbr = gbr_model.predict(X_test)

# Оцінка моделі GradientBoostingRegressor
evaluate_model(y_test, y_pred_gbr, "GradientBoostingRegressor")

plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
plt.scatter(y_test, y_pred_gbr, alpha=0.7, color='brown') # Побудова точкового графіка
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ідеальна лінія прогнозу
plt.xlabel('Actual Prices') # Підпис осі X
plt.ylabel('Predicted Prices (GradientBoostingRegressor Model)') # Підпис осі Y
plt.title('Actual vs. Predicted Prices (GradientBoostingRegressor Model)') # Заголовок графіка
plt.grid(True) # Увімкнення сітки
plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування для GradientBoostingRegressor
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gbr, 'Absolute_Error': np.abs(y_test - y_pred_gbr)}) # Створення DataFrame з помилками
errors = errors.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою

print("\nTop 10 data points with the largest prediction errors:") # Виведення заголовка
display.display(errors.head(10)) # Відображення 10 найбільших помилок

# ============

import pandas as pd
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')
df.info()
df #display it
# -------------------------------------------------------

# ------------------------------------------
# ------------------------------------------
# Лінійна регресія 1.5 !!! Точність моделі = 81.63% (main)
# ------------------------------------------
# ------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error # оцінки точості моделі
import matplotlib.pyplot as plt

# Load your data
#df = pd.read_excel('train.xlsx')
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
mse = mean_squared_error(y_test, y_pred)

# Розрахунок середньої точності у відсотках
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy_in_percent = 100 - mape

print("\nОцінка точності моделі:")
print(f"Mean Absolute Error (MAE): {mae:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"R-squared (R2) score: {r2:.2f}")
print(f"Точність: {accuracy_in_percent:.2f}%") # Те саме число у відсотках

# Виведення коефіцієнтів моделі
print("\nКоефіцієнти лінійної регресії:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Перетин (intercept): {model.intercept_:.2f}")

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



# ---------------------------------------
# ---------------------------------------
# Поліноміальна регресія 1.5 !!! Точність моделі = 81.46% (main)
# ---------------------------------------
# ---------------------------------------
import numpy as np # Ensure numpy is imported for hstack
import pandas as pd # Ensure pandas is imported for get_dummies
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load your data
df = pd.read_excel('train.xlsx')
#df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')
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


numerical_features = ['Reviews', 'Ratings']
categorical_features_to_add = ['Author_Encoded', 'Genre_Encoded']
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
mse_poly = mean_squared_error(y_test, y_pred_poly)

mape_poly = np.mean(np.abs((y_test - y_pred_poly) / (y_test + 1e-10))) * 100
accuracy_poly = 100 - mape_poly

print(f"Mean Absolute Error (MAE): {mae_poly:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_poly:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_poly:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"R-squared (R2) score: {r2_poly:.2f}")
print(f"Точність моделі: {accuracy_poly:.2f}%")

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

# Графік порівняння реальних цін із прогнозованими моделлю поліноміальної регресії
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_poly, alpha=0.7, color='purple') 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Polynomial + Categorical Model)')
plt.title('Actual vs. Predicted Prices (Polynomial Regression with Categorical Features)')
plt.grid(True)
plt.show()

# ---------------------------------------
# ---------------------------------------

# ---------------------------------------
# ---------------------------------------
# Еластична мережева Додано точність моделі = 85.14% регресія та Гребнева регресія Додано точність моделі = 84.97% (main)
# ---------------------------------------
# ---------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, Ridge # Import ElasticNet and Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load your data
#df = pd.read_excel('train.xlsx')
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# --- Start of added preprocessing for self-containment ---
# Convert 'Price' to numeric, handling errors by coercing to NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Clean 'Reviews' and 'Ratings' columns to extract numerical values
df['Reviews'] = df['Reviews'].astype(str).str.extract('(\\d+\\.?\\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract('(\\d+)').astype(float)

# Calculate mean prices by Author and Genre for target encoding
# Note: Applying target encoding on the full dataframe before splitting can lead to data leakage.
# For a more robust ML pipeline, target encoding should ideally be calculated on training data only.
# However, to replicate the kernel state's 'Author_Encoded' and 'Genre_Encoded' as floats,
# we apply it here for self-containment.

global_mean_price = df['Price'].mean() # Calculate global mean price for filling NaNs

mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')

df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre

df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Define X and y using the processed features
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Drop rows where y (Price) is NaN, as these cannot be used for training
# Ensure X and y have the same index after dropping NaNs
y.dropna(inplace=True)
X = X.loc[y.index]

# Розділення даних на тренувальний та тестовий набори (як і раніше)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define features for polynomial expansion and categorical features
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

# Scale Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Combine scaled numerical features with categorical features
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))
# --- End of added preprocessing for self-containment ---

# Initialize and train Elastic Net model
elastic_net_model = ElasticNet(random_state=42)
elastic_net_model.fit(X_train_combined, y_train)

# Initialize and train Ridge model
ridge_net_model = Ridge(random_state=42)
ridge_net_model.fit(X_train_combined, y_train)

# Predict on the test data using the Elastic Net model
# y_pred_elastic = elastic_net_model.predict(X_test_all)
y_pred_elastic = elastic_net_model.predict(X_test_combined)

# Predict on the test data using the Ridge model
# y_pred_ridge = ridge_model.predict(X_test_all)
# y_pred_ridge = ridge_model.predict(X_test_combined)
y_pred_ridge = ridge_net_model.predict(X_test_combined)

# Evaluate Elastic Net model
mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)

mape_elastic = np.mean(np.abs((y_test - y_pred_elastic) / (y_test + 1e-10))) * 100
accuracy_elastic = 100 - mape_elastic

print("Elastic Net Model Evaluation:")
print(f"  Mean Absolute Error (MAE): {mae_elastic:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_elastic:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_elastic:.2f}%") # Середня абсолютна відсоткова помилка
print(f"  R-squared (R2): {r2_elastic:.2f}")
print(f"Точність моделі: {accuracy_elastic:.2f}%")


# Evaluate Ridge model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

mape_ridge = np.mean(np.abs((y_test - y_pred_ridge) / (y_test + 1e-10))) * 100
accuracy_ridge = 100 - mape_ridge

print("\nRidge Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_ridge:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_ridge:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_ridge:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"  R-squared (R2): {r2_ridge:.2f}")
print(f"Точність моделі: {accuracy_ridge:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_ridge, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Ridge Model)')
plt.title('Actual vs. Predicted Prices (Ridge Model)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_elastic, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Elastic Net Model)')
plt.title('Actual vs. Predicted Prices (Elastic Net Model)')
plt.grid(True)
plt.show()

# ---------------------------------------
# ---------------------------------------
# RandomForestRegressor 1 !!! Додано точність моделі = 83.48% (main)
# ---------------------------------------
# ---------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import matplotlib.pyplot as plt
import io
import base64
import IPython.display as display


df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# --- Start of added preprocessing for self-containment ---
# Convert 'Price' to numeric, handling errors by coercing to NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Clean 'Reviews' and 'Ratings' columns to extract numerical values
df['Reviews'] = df['Reviews'].astype(str).str.extract('(\\d+\\.?\\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract('(\\d+)').astype(float)

# Calculate mean prices by Author and Genre for target encoding
# Note: Applying target encoding on the full dataframe before splitting can lead to data leakage.
# For a more robust ML pipeline, target encoding should ideally be calculated on training data only.
# However, to replicate the kernel state's 'Author_Encoded' and 'Genre_Encoded' as floats,
# we apply it here for self-containment.

global_mean_price = df['Price'].mean() # Calculate global mean price for filling NaNs

mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')

df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre

df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Define X and y using the processed features
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Drop rows where y (Price) is NaN, as these cannot be used for training
# Ensure X and y have the same index after dropping NaNs
y.dropna(inplace=True)
X = X.loc[y.index]

# Розділення даних на тренувальний та тестовий набори (як і раніше)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define features for polynomial expansion and categorical features
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

# Scale Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Combine scaled numerical features with categorical features
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))
# --- End of added preprocessing for self-containment ---
# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_rf = model.predict(X_test)

# Оцінка моделі RandomForestRegressor
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

mape_rf = np.mean(np.abs((y_test - y_pred_rf) / (y_test + 1e-10))) * 100
accuracy_rf = 100 - mape_rf

print("\nRandomForestRegressor Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}") # Mean Absolute Error — Середня абсолютна помилка
print(f"Mean Squared Error (MSE): {mse_rf:.2f}") # Mean Squared Error — Середня квадратична помилка
print(f"Mean Absolute Percentage Error (MAPE): {mape_rf:.2f}%") # Mean Absolute Percentage Error — Середня абсолютна відсоткова помилка
print(f"  R-squared (R2): {r2_rf:.2f}")
print(f"Точність моделі: {accuracy_rf:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='yellow')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (RandomForestRegressor Model)')
plt.title('Actual vs. Predicted Prices (RandomForestRegressor Regression Model)')
plt.grid(True)
plt.show()
# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf, 'Absolute_Error': np.abs(y_test - y_pred_rf)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# ---------------------------------------
# ---------------------------------------

# ---------------------------------------
# ---------------------------------------
# GradientBoostingRegressor !!! Додано точність моделі = 84.61% (main)
# ---------------------------------------
# ---------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import matplotlib.pyplot as plt
import io
import base64
import IPython.display as display

df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# --- Start of added preprocessing for self-containment ---
# Convert 'Price' to numeric, handling errors by coercing to NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Clean 'Reviews' and 'Ratings' columns to extract numerical values
# df['Reviews'] = df['Reviews'].astype(str).str.extract('(\\d+\\.?\\d*)').astype(float)
# df['Ratings'] = df['Ratings'].astype(str).str.extract('(\\d+)').astype(float)
df['Reviews'] = df['Reviews'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract('(\d+)').astype(float)

# Calculate mean prices by Author and Genre for target encoding
# Note: Applying target encoding on the full dataframe before splitting can lead to data leakage.
# For a more robust ML pipeline, target encoding should ideally be calculated on training data only.
# However, to replicate the kernel state's 'Author_Encoded' and 'Genre_Encoded' as floats,
# we apply it here for self-containment.

global_mean_price = df['Price'].mean() # Calculate global mean price for filling NaNs

mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')

df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre

df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Define X and y using the processed features
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Drop rows where y (Price) is NaN, as these cannot be used for training
# Ensure X and y have the same index after dropping NaNs
y.dropna(inplace=True)
X = X.loc[y.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# Define features for polynomial expansion and categorical features
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

# Scale Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Combine scaled numerical features with categorical features
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))
# --- End of added preprocessing for self-containment ---

# Train the GradientBoostingRegressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_gbr = gbr_model.predict(X_test)

# Evaluate the GradientBoostingRegressor model
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)

mape_gbr = np.mean(np.abs((y_test - y_pred_gbr) / (y_test + 1e-10))) * 100
accuracy_gbr = 100 - mape_gbr

print("\nGradientBoostingRegressor Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_gbr:.2f}")
print(f"Mean Squared Error (MSE): {mse_gbr:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_gbr:.2f}%")
print(f"  R-squared (R2): {r2_gbr:.2f}")
print(f"Точність моделі: {accuracy_gbr:.2f}%")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_gbr, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2) # Ideal prediction line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (GradientBoostingRegressor Model)')
plt.title('Actual vs. Predicted Prices (GradientBoostingRegressor Model)')
plt.grid(True)
plt.show()

# Identify data points with the largest prediction errors
errors = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gbr, 'Absolute_Error': np.abs(y_test - y_pred_gbr)})
errors = errors.sort_values(by='Absolute_Error', ascending=False)

print("\nTop 10 data points with the largest prediction errors:")
display.display(errors.head(10))

# --------------------------------------
# --------------------------------------
# Порівняння абсолютної помилки для моделей : RandomForestRegressor та GradientBoostingRegressor 
# --------------------------------------
# --------------------------------------
import seaborn as sns

# Create a DataFrame to hold absolute errors for both models
all_errors = pd.DataFrame({
    'Model': ['RandomForestRegressor'] * len(errors['Absolute_Error']) + ['GradientBoostingRegressor'] * len(errors_gbr['Absolute_Error']),
    'Absolute_Error': pd.concat([errors['Absolute_Error'], errors_gbr['Absolute_Error']])
})

plt.figure(figsize=(10, 6))
sns.violinplot(x='Model', y='Absolute_Error', data=all_errors, hue='Model', palette='muted', legend=False)
plt.title('Comparison of Absolute Error Distributions Between Models')
plt.xlabel('Model')
plt.ylabel('Absolute Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, all_errors['Absolute_Error'].quantile(0.95)) # Limit y-axis for better visualization, excluding extreme outliers
plt.show()

# --------------------------------------
# --------------------------------------


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
# Градієнтний бустинг (Gradient Boosting Machines, XGBoost, LightGBM, CatBoost) GradientBoostingRegressor - є
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

#======================================================================
#====================================================================
# 1. МАГІЧНІ КОМАНДИ ТА ІМПОРТ
%matplotlib inline
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from IPython.display import display

# Налаштування стилю графіків
plt.style.use('seaborn-v0_8')

# 2. ЗАВАНТАЖЕННЯ ДАНИХ
try:
    train_raw = pd.read_excel('train.xlsx')
    test_raw = pd.read_excel('test.xlsx')
    print("✅ Файли успішно завантажено!")
except Exception as e:
    print(f"❌ Помилка завантаження: {e}. Перевірте вкладку Files зліва.")

# 3. ФУНКЦІЯ ОБРОБКИ (ADVANCED FEATURE ENGINEERING)
def preprocess_data(df, is_train=True, encoders=None):
    df = df.copy()

    # Очищення числових значень
    df['Reviews'] = df['Reviews'].apply(lambda x: float(re.search(r'(\d+\.\d+|\d+)', str(x)).group(1))
                                       if pd.notnull(x) and re.search(r'(\d+\.\d+|\d+)', str(x)) else 0.0)
    df['Ratings'] = df['Ratings'].apply(lambda x: int(re.sub(r'[^\d]', '', str(x)))
                                       if pd.notnull(x) and re.sub(r'[^\d]', '', str(x)) != '' else 0)

    # Створення нових ознак (Features)
    # Витягуємо рік
    df['Year'] = df['Edition'].apply(lambda x: int(re.search(r'(\d{4})', str(x)).group(1))
                                    if re.search(r'(\d{4})', str(x)) else 2010)
    # Тип обкладинки
    df['Is_Hardcover'] = df['Edition'].apply(lambda x: 1 if 'Hardcover' in str(x) else 0)
    # Довжина назви
    df['Title_Len'] = df['Title'].apply(lambda x: len(str(x).split()))

    # Кодування категоріальних колонок
    cat_cols = ['Author', 'Genre', 'BookCategory']
    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
    else:
        for col in cat_cols:
            le = encoders[col]
            # Обробка невідомих категорій у тесті
            df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        return df

# 4. ПІДГОТОВКА ТА НАВЧАННЯ
print("⏳ Обробка даних та запуск моделі...")
train_df, encoders = preprocess_data(train_raw, is_train=True)

# Вибираємо фінальні ознаки для навчання
features = ['Reviews', 'Ratings', 'Author', 'Genre', 'BookCategory', 'Year', 'Is_Hardcover', 'Title_Len']
X = train_df[features]
y = np.log1p(train_df['Price']) # Логарифмування ціни для стабільності

# Розділяємо на Train/Validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Налаштування моделі Random Forest
model = RandomForestRegressor(n_estimators=300, max_depth=22, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# 5. ОЦІНКА РЕЗУЛЬТАТІВ
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log) # Повертаємо з логарифма
y_val_real = np.expm1(y_val)

mae = mean_absolute_error(y_val_real, val_preds)
r2 = r2_score(y_val_real, val_preds)
mape = mean_absolute_percentage_error(y_val_real, val_preds)
accuracy = max(0, (1 - mape) * 100)

# 6. ВИВІД ТАБЛИЦЬ (КРАСИВИЙ ФОРМАТ)
print("\n" + "="*40)
print(f"📊 СТАТИСТИКА ТОЧНОСТІ:")
print(f"Середня помилка (MAE): {mae:.2f} одиниць")
print(f"ТОЧНІСТЬ ПРОГРАМИ: {accuracy:.2f}%")
print(f"Коефіцієнт R²: {r2:.4f}")
print("="*40)

print("\n📋 ТАБЛИЦЯ ПОРІВНЯННЯ (План vs Факт):")
comparison_df = pd.DataFrame({
    'Реальна ціна': y_val_real,
    'Прогноз': val_preds.round(2),
    'Похибка': np.abs(y_val_real - val_preds).round(2)
}).head(10)
display(comparison_df)

print("\n📈 ВАЖЛИВІСТЬ ОЗНАК:")
importance_df = pd.DataFrame({
    'Параметр': features,
    'Вплив (%)': (model.feature_importances_ * 100).round(2)
}).sort_values(by='Вплив (%)', ascending=False)
display(importance_df)

# 7. ГЕНЕРАЦІЯ ГРАФІКІВ (БЕЗ ПОПЕРЕДЖЕНЬ)
print("\n🎨 Візуалізація результатів...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Матриця розсіювання
ax1.scatter(y_val_real, val_preds, alpha=0.4, color='#3498db', edgecolor='white', s=50)
ax1.plot([y_val_real.min(), y_val_real.max()], [y_val_real.min(), y_val_real.max()], 'r--', lw=3)
ax1.set_title('Реальна ціна vs Прогнозована', fontsize=15)
ax1.set_xlabel('Справжня ціна книги', fontsize=12)
ax1.set_ylabel('Прогноз моделі', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)

# Barplot важливості
sns.barplot(x='Вплив (%)', y='Параметр', data=importance_df, ax=ax2,
            palette='magma', hue='Параметр', legend=False)
ax2.set_title('Які фактори керують ціною?', fontsize=15)
ax2.set_xlabel('Рівень впливу у %', fontsize=12)

plt.tight_layout()
plt.show()

# 8. ЗБЕРЕЖЕННЯ ФІНАЛЬНОГО ПРОГНОЗУ ДЛЯ ТЕСТУ
test_df = preprocess_data(test_raw, is_train=False, encoders=encoders)
final_preds = np.expm1(model.predict(test_df[features]))
pd.DataFrame({'Price': final_preds}).to_excel('final_predictions_model.xlsx', index=False)
print("\n✅ Файл 'final_predictions_model.xlsx' готовий до завантаження!")
