#!/usr/bin/env python3
"""
Setup and initialization module for models and database
"""

import os
import subprocess
from pathlib import Path
from config import MODEL_FILE_PATHS, SCRIPTS_CONFIG
from utils.logger import setup_logger
from utils.database import check_database_exists

logger = setup_logger(__name__)

def check_and_setup():
    """
    Check if models and database exist
    Models must be imported from Google Drive (pre-trained)
    """
    logger.info("=" * 70)
    logger.info("INITIALIZATION: Checking models and database...")
    logger.info("=" * 70)
    
    models_exist = check_models_exist()
    database_exists = check_database_exists()
    
    if not models_exist:
        logger.error("Models not found!")
        logger.error("Please download pre-trained models from Google Drive")
        logger.error("Expected .pkl files to be in project root:")
        for key, path in MODEL_FILE_PATHS.items():
            logger.error(f"   - {key}: {path}")
        raise FileNotFoundError("Model files not found. Please download from Google Drive.")
    else:
        logger.info("All required model files exist")
    
    if not database_exists:
        logger.warning("Database not found. Running create_database.py...")
        run_create_database()
    else:
        logger.info("Database exists and is valid")
    
    logger.info("=" * 70)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("=" * 70)

def check_models_exist():
    """
    Check if all required model files exist
    
    Returns:
        bool: True if all models exist
    """
    missing = []
    
    for key, path in MODEL_FILE_PATHS.items():
        if not os.path.exists(path):
            missing.append(f"{key} ({path})")
    
    if missing:
        logger.warning(f"Missing models: {', '.join(missing)}")
        return False
    
    logger.info(f"All {len(MODEL_FILE_PATHS)} model files found")
    return True

def run_create_database():
    """
    Run create_database.py script
    """
    try:
        script_path = Path(__file__).parent.parent / 'scripts' / 'create_database.py'
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        logger.info(f"Running {script_path.name}...")
        result = subprocess.run(
            ['python3', str(script_path)],
            capture_output=True,
            text=True,
            timeout=SCRIPTS_CONFIG['create_database_timeout']
        )
        
        if result.returncode == 0:
            logger.info("Database created successfully!")
        else:
            logger.error(f"Database creation failed:\n{result.stderr}")
            raise RuntimeError("Failed to create database")
            
    except Exception as e:
        logger.error(f"Error running create_database: {e}")
        raise
