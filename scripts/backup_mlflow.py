from sleep_staging.backup import backup_mlflow
from sleep_staging.config import project_root

if __name__ == "__main__":
    print(backup_mlflow(project_root()))
