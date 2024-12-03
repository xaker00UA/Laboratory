import os
import subprocess
import sys


def run_main_files(start_path):
    """Рекурсивно проходит по папкам и запускает main.py в текущем виртуальном окружении."""
    for root, _, files in os.walk(start_path):
        if "main.py" in files:
            main_file_path = os.path.join(root, "main.py")
            # Получаем абсолютный путь к файлу
            abs_main_file_path = os.path.abspath(main_file_path)
            # Формируем команду
            command = [sys.executable, abs_main_file_path]
            print(f"Запуск команды: {command}")
            # Запускаем команду в директории, где находится main.py
            subprocess.run(command, cwd=root, shell=True)


if __name__ == "__main__":
    # Укажите путь к папке, с которой нужно начать поиск
    start_directory = "."
    run_main_files(start_directory)
