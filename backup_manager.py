# backup_manager.py
import os
import shutil
from datetime import datetime


class BackupManager:
    def __init__(self, base_dir='backups'):
        """
        Manage backups of models and datasets
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def backup_model(self, model_path):
        """
        Create a backup of a trained model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(model_path)
        backup_filename = f'{timestamp}_{filename}'
        backup_path = os.path.join(self.base_dir, backup_filename)

        shutil.copy2(model_path, backup_path)
        return backup_path

    def cleanup_old_backups(self, days_to_keep=30):
        """
        Remove backups older than specified days
        """
        now = datetime.now()
        for filename in os.listdir(self.base_dir):
            filepath = os.path.join(self.base_dir, filename)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

            if (now - file_mtime).days > days_to_keep:
                os.remove(filepath)