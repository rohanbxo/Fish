"""
Configuration management
"""

import os
from typing import Dict, Any
import yaml
import json


class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'model': {
                'email_model_path': None,
                'url_model_path': None,
                'max_length': 512,
                'batch_size': 16
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'reload': True
            },
            'web': {
                'port': 8501
            },
            'paths': {
                'data_dir': './data',
                'model_dir': './models',
                'logs_dir': './logs'
            },
            'detection': {
                'email_weight': 0.6,
                'url_weight': 0.4,
                'risk_thresholds': {
                    'high': 0.8,
                    'medium': 0.5,
                    'low': 0.3
                }
            }
        }
    
    def load(self, config_path: str):
        """Load configuration from file"""
        ext = os.path.splitext(config_path)[1].lower()
        
        with open(config_path, 'r') as f:
            if ext in ['.yaml', '.yml']:
                loaded_config = yaml.safe_load(f)
            elif ext == '.json':
                loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {ext}")
        
        # Update config with loaded values
        self._update_config(self.config, loaded_config)
    
    def _update_config(self, base: Dict, update: Dict):
        """Recursively update configuration"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        ext = os.path.splitext(save_path)[1].lower()
        
        with open(save_path, 'w') as f:
            if ext in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False)
            elif ext == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {ext}")
