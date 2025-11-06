import pandas as pd
from sklearn.datasets import make_moons, load_iris
from pathlib import Path

# --- 1. Визначення шляху до папки зі скриптом ---
# __file__ — це змінна, що містить шлях до поточного файлу .py
# .parent — це команда, щоб отримати папку, в якій лежить цей файл
script_directory = Path(__file__).parent

print(f"Скрипт запущено. Файли будуть збережені у: {script_directory}")

# --- 2. Набір даних 'make_moons' ---
print("Обробка 'make_moons'...")
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
moons_df = pd.DataFrame(X_moons, columns=['feature_1', 'feature_2'])
moons_df['target'] = y_moons

# Складаємо повний шлях: Папка_скрипта / ім'я_файла.csv
moons_filepath = script_directory / "moons_dataset.csv"
moons_df.to_csv(moons_filepath, index=False)

print(f" -> Файл '{moons_filepath.name}' успішно створено.")

# --- 3. Набір даних 'load_iris' ---
print("Обробка 'load_iris'...")
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
iris_df = pd.DataFrame(X_iris, columns=iris.feature_names)
iris_df['target'] = y_iris

# Складаємо повний шлях: Папка_скрипта / ім'я_файла.csv
iris_filepath = script_directory / "iris_dataset.csv"
iris_df.to_csv(iris_filepath, index=False)

print(f" -> Файл '{iris_filepath.name}' успішно створено.")
print("\nГотово! Файли знаходяться у тій самій папці, що і ваш скрипт.")