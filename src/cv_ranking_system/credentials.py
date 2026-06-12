"""Credentials Management for Kaggle and HuggingFace APIs

This module provides a centralized way to manage credentials for external services
without relying on system-level configurations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class KaggleCredentials:
    """Kaggle API credentials."""
    username: str
    api_key: str
    
    def validate(self) -> bool:
        """Validate that both username and API key are provided."""
        return bool(self.username and self.api_key)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HuggingFaceCredentials:
    """HuggingFace API credentials."""
    api_token: str
    
    def validate(self) -> bool:
        """Validate that API token is provided."""
        return bool(self.api_token)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class CredentialsManager:
    """Manages credentials for Kaggle and HuggingFace APIs."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize credentials manager.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
        """
        self.env_file = env_file or self._find_env_file()
        self.kaggle_creds: Optional[KaggleCredentials] = None
        self.huggingface_creds: Optional[HuggingFaceCredentials] = None
        
        self._load_credentials()
    
    @staticmethod
    def _find_env_file() -> Optional[str]:
        """Find .env file starting from current directory up to project root."""
        current = Path.cwd()
        while current != current.parent:
            env_path = current / ".env"
            if env_path.exists():
                logger.info(f"Found .env file at: {env_path}")
                return str(env_path)
            current = current.parent
        
        # Also check in project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            logger.info(f"Found .env file at: {env_path}")
            return str(env_path)
        
        return None
    
    def _load_credentials(self) -> None:
        """Load credentials from .env file and environment variables."""
        # Load .env file if it exists
        if self.env_file and Path(self.env_file).exists():
            logger.info(f"Loading credentials from: {self.env_file}")
            load_dotenv(self.env_file)
        else:
            logger.warning(
                "No .env file found. Checking environment variables. "
                "Create a .env file or set environment variables for credentials."
            )
        
        # Load Kaggle credentials
        self._load_kaggle_credentials()
        
        # Load HuggingFace credentials
        self._load_huggingface_credentials()
    
    def _load_kaggle_credentials(self) -> None:
        """Load Kaggle credentials from environment."""
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_api_key = os.getenv("KAGGLE_API_KEY")
        
        if kaggle_username and kaggle_api_key:
            self.kaggle_creds = KaggleCredentials(
                username=kaggle_username,
                api_key=kaggle_api_key
            )
            logger.info(f"Kaggle credentials loaded for user: {kaggle_username}")
        else:
            logger.warning(
                "Kaggle credentials not found. "
                "Set KAGGLE_USERNAME and KAGGLE_API_KEY in .env or environment."
            )
    
    def _load_huggingface_credentials(self) -> None:
        """Load HuggingFace credentials from environment."""
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        
        if hf_token:
            self.huggingface_creds = HuggingFaceCredentials(api_token=hf_token)
            logger.info("HuggingFace credentials loaded")
        else:
            logger.warning(
                "HuggingFace credentials not found. "
                "Set HUGGINGFACE_API_TOKEN or HF_TOKEN in .env or environment."
            )
    
    def get_kaggle_credentials(self) -> Optional[KaggleCredentials]:
        """Get Kaggle credentials.
        
        Returns:
            KaggleCredentials if available, None otherwise.
        """
        return self.kaggle_creds
    
    def get_huggingface_credentials(self) -> Optional[HuggingFaceCredentials]:
        """Get HuggingFace credentials.
        
        Returns:
            HuggingFaceCredentials if available, None otherwise.
        """
        return self.huggingface_creds
    
    def validate_kaggle(self) -> bool:
        """Validate Kaggle credentials are available and valid."""
        if not self.kaggle_creds:
            logger.error("Kaggle credentials not loaded")
            return False
        return self.kaggle_creds.validate()
    
    def validate_huggingface(self) -> bool:
        """Validate HuggingFace credentials are available and valid."""
        if not self.huggingface_creds:
            logger.error("HuggingFace credentials not loaded")
            return False
        return self.huggingface_creds.validate()
    
    def setup_kaggle_auth(self) -> bool:
        """Configure Kaggle API authentication.
        
        Sets up Kaggle authentication by creating kaggle.json file
        in the appropriate location.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.validate_kaggle():
            logger.error("Cannot setup Kaggle auth: credentials invalid")
            return False
        
        try:
            # Create kaggle config directory
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            
            # Create kaggle.json
            kaggle_config_file = kaggle_dir / "kaggle.json"
            config_data = {
                "username": self.kaggle_creds.username,
                "key": self.kaggle_creds.api_key
            }
            
            with open(kaggle_config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Set proper permissions (Unix-like systems)
            try:
                kaggle_config_file.chmod(0o600)
            except (OSError, NotImplementedError):
                # Windows doesn't support chmod, skip
                pass
            
            logger.info(f"Kaggle authentication configured at: {kaggle_config_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to setup Kaggle authentication: {e}")
            return False
    
    def setup_huggingface_auth(self) -> bool:
        """Configure HuggingFace API authentication.
        
        Sets up HuggingFace authentication by setting environment variable
        or creating the auth token file.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.validate_huggingface():
            logger.error("Cannot setup HuggingFace auth: credentials invalid")
            return False
        
        try:
            # HuggingFace primarily uses environment variable
            os.environ["HF_TOKEN"] = self.huggingface_creds.api_token
            
            # Also try to setup via huggingface_hub if available
            try:
                from huggingface_hub import login
                login(token=self.huggingface_creds.api_token, add_to_git_credential=False)
                logger.info("HuggingFace authentication configured via huggingface_hub")
            except ImportError:
                logger.warning("huggingface_hub not installed, using environment variable only")
            except Exception as e:
                logger.warning(f"Could not configure via huggingface_hub: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to setup HuggingFace authentication: {e}")
            return False
    
    def setup_all_auth(self) -> dict:
        """Setup authentication for all configured credentials.
        
        Returns:
            Dictionary with status for each service.
        """
        results = {
            "kaggle": self.setup_kaggle_auth(),
            "huggingface": self.setup_huggingface_auth()
        }
        
        logger.info(f"Authentication setup results: {results}")
        return results
    
    def get_status(self) -> dict:
        """Get current credentials status.
        
        Returns:
            Dictionary with status information.
        """
        return {
            "kaggle": {
                "configured": self.kaggle_creds is not None,
                "valid": self.validate_kaggle() if self.kaggle_creds else False,
                "username": self.kaggle_creds.username if self.kaggle_creds else None
            },
            "huggingface": {
                "configured": self.huggingface_creds is not None,
                "valid": self.validate_huggingface() if self.huggingface_creds else False
            },
            "env_file": self.env_file
        }


# Global credentials manager instance
_credentials_manager: Optional[CredentialsManager] = None


def get_credentials_manager() -> CredentialsManager:
    """Get or create the global credentials manager instance."""
    global _credentials_manager
    if _credentials_manager is None:
        _credentials_manager = CredentialsManager()
    return _credentials_manager


def initialize_credentials(env_file: Optional[str] = None) -> CredentialsManager:
    """Initialize credentials manager with optional custom .env file.
    
    Args:
        env_file: Path to custom .env file.
        
    Returns:
        Initialized CredentialsManager instance.
    """
    global _credentials_manager
    _credentials_manager = CredentialsManager(env_file)
    return _credentials_manager
