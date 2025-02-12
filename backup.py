#!/usr/bin/env python3
import os
import sys
import tarfile
from datetime import datetime
import shutil
import argparse

BACKUP_LIST = "backup_list.txt"
SNAPSHOTS_DIR = "./.snapshots"

def revert_backup():
    latest_backup = os.path.join(SNAPSHOTS_DIR, "backup_latest.tar")
    if not os.path.exists(latest_backup):
        print("Error: No backup found to revert")
        sys.exit(1)
    
    with tarfile.open(latest_backup, "r") as tar:
        tar.extractall()
    print("Latest backup restored successfully")

def create_backup():
    if not os.path.exists(BACKUP_LIST):
        print(f"Error: {BACKUP_LIST} not found")
        sys.exit(1)
        
    with open(BACKUP_LIST) as f:
        files = [line.strip() for line in f if line.strip()]
        
    # Validate files exist
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print("Error: Following files not found:")
        print("\n".join(missing))
        sys.exit(1)

    # Create snapshots directory
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    
    # Create backup name with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}.tar"
    backup_path = os.path.join(SNAPSHOTS_DIR, backup_name)
    
    # Create tar archive
    with tarfile.open(backup_path, "w") as tar:
        for file in files:
            tar.add(file)
    
    # Create/update latest backup link
    latest_backup = os.path.join(SNAPSHOTS_DIR, "backup_latest.tar")
    if os.path.exists(latest_backup):
        os.remove(latest_backup)
    shutil.copy2(backup_path, latest_backup)
    
    print(f"Backup created: {backup_path}")
    print(f"Latest backup updated: {latest_backup}")

def main():
    parser = argparse.ArgumentParser(description='Backup and restore sensitive files')
    parser.add_argument('--revert', action='store_true', help='Revert to latest backup')
    args = parser.parse_args()

    if args.revert:
        revert_backup()
    else:
        create_backup()

if __name__ == "__main__":
    main() 