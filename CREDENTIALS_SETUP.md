# Credentials Setup Guide

## Overview

This guide explains how to configure Kaggle and HuggingFace credentials for the CV Ranking System without using system-level configurations.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Get your credentials:**
   - [Kaggle API Key](#kaggle-setup)
   - [HuggingFace API Token](#huggingface-setup)

3. **Update `.env` file with your credentials**

4. **Done!** The system will automatically use these credentials

## Kaggle Setup

### Step 1: Go to Kaggle API Settings
1. Log in to [Kaggle.com](https://www.kaggle.com)
2. Click on your profile icon (top right)
3. Select **Account**

### Step 2: Create/Download API Token
1. Scroll down to the **API** section
2. Click **Create New API Token**
   - This downloads a file called `kaggle.json`
3. Open `kaggle.json` in a text editor

### Step 3: Add to `.env`
Copy the values from `kaggle.json` to your `.env` file:

```bash
# .env file
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_KEY=your_kaggle_api_key
```

**Example:**
```bash
KAGGLE_USERNAME=john_doe
KAGGLE_API_KEY=abc123def456ghi789jkl012
```

## HuggingFace Setup

### Step 1: Go to HuggingFace Settings
1. Log in to [HuggingFace.co](https://huggingface.co)
2. Click on your profile icon (top right)
3. Select **Settings**

### Step 2: Create API Token
1. Click **Access Tokens** in the left menu
2. Click **New token**
3. Give it a name (e.g., "CV Ranking System")
4. Select **read** permission (for downloading models/datasets)
5. Click **Create token**
6. Copy the token value

### Step 3: Add to `.env`
```bash
# .env file
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Alternative variable name (also supported):**
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Usage in Code

### Automatic Initialization

The credentials are automatically loaded when you use the dataset manager:

```python
from cv_ranking_system.dataset_manager import DatasetManager

# This automatically loads credentials from .env
manager = DatasetManager()

# Download datasets
datasets = manager.download_extraction_datasets()
```

### Manual Initialization

For more control, you can explicitly initialize credentials:

```python
from cv_ranking_system.credentials import initialize_credentials

# Initialize with default .env location
creds = initialize_credentials()

# Or specify custom .env path
creds = initialize_credentials(env_file="/path/to/custom/.env")

# Check status
status = creds.get_status()
print(status)
```

### Check Credentials Status

```python
from cv_ranking_system.credentials import get_credentials_manager

creds = get_credentials_manager()

# Check if credentials are configured
status = creds.get_status()
print(f"Kaggle configured: {status['kaggle']['configured']}")
print(f"Kaggle valid: {status['kaggle']['valid']}")
print(f"HuggingFace configured: {status['huggingface']['configured']}")
```

### Get Specific Credentials

```python
from cv_ranking_system.credentials import get_credentials_manager

creds = get_credentials_manager()

# Get Kaggle credentials
kaggle_creds = creds.get_kaggle_credentials()
if kaggle_creds:
    print(f"Kaggle user: {kaggle_creds.username}")

# Get HuggingFace credentials
hf_creds = creds.get_huggingface_credentials()
if hf_creds:
    print("HuggingFace token is configured")
```

## Environment Variable Priority

The system checks environment variables in this order:

### For Kaggle:
1. `KAGGLE_USERNAME` environment variable
2. `KAGGLE_USERNAME` from `.env` file

### For HuggingFace:
1. `HF_TOKEN` environment variable
2. `HUGGINGFACE_API_TOKEN` environment variable
3. `HF_TOKEN` from `.env` file
4. `HUGGINGFACE_API_TOKEN` from `.env` file

## Troubleshooting

### "Kaggle credentials not found" Error

```python
from cv_ranking_system.credentials import get_credentials_manager

creds = get_credentials_manager()
status = creds.get_status()

if not status['kaggle']['configured']:
    print("Kaggle credentials not set in .env or environment")
    print("Please add KAGGLE_USERNAME and KAGGLE_API_KEY to .env")
```

**Solutions:**
1. Create `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. Add your credentials to `.env`:
   ```bash
   KAGGLE_USERNAME=your_username
   KAGGLE_API_KEY=your_api_key
   ```

3. Make sure `.env` is in the project root directory

### "HuggingFace authentication failed" Warning

This warning appears if you try to download private HuggingFace datasets without a token.

**Solutions:**
1. Add your HuggingFace token to `.env`:
   ```bash
   HUGGINGFACE_API_TOKEN=hf_xxxxx
   ```

2. Or set as environment variable:
   ```bash
   export HUGGINGFACE_API_TOKEN=hf_xxxxx
   ```

3. Public datasets will still download without authentication

### ".env file not found" Warning

The system looks for `.env` in these locations (in order):
1. Current working directory
2. Parent directories up to project root
3. Project root directory

**Solutions:**
1. Create `.env` in the project root:
   ```bash
   cp .env.example .env
   ```

2. Or specify custom path:
   ```python
   from cv_ranking_system.credentials import initialize_credentials
   creds = initialize_credentials(env_file="/path/to/.env")
   ```

## Security Best Practices

1. **Never commit `.env` to Git:**
   - `.env` is typically in `.gitignore`
   - Check your `.gitignore` file

2. **Keep credentials private:**
   - Don't share your `.env` file
   - Don't commit API keys/tokens

3. **Use strong passwords:**
   - Generate new tokens regularly
   - Revoke old tokens

4. **Limit token scope:**
   - HuggingFace: Use "read" permission if possible
   - Only grant necessary permissions

5. **Monitor API usage:**
   - Kaggle: Check [Kaggle API status](https://www.kaggle.com/settings/account)
   - HuggingFace: Monitor your [token usage](https://huggingface.co/settings/tokens)

## Advanced Configuration

### Using Different Credentials Per Run

```python
from cv_ranking_system.credentials import initialize_credentials
from cv_ranking_system.dataset_manager import DatasetManager

# Initialize with custom credentials file
creds = initialize_credentials(env_file="/path/to/prod/.env")

# Create manager with this credentials instance
manager = DatasetManager(credentials_manager=creds)

# Use manager
datasets = manager.download_extraction_datasets()
```

### Environment-Specific Configuration

```bash
# .env.dev
KAGGLE_USERNAME=dev_user
KAGGLE_API_KEY=dev_key
HUGGINGFACE_API_TOKEN=hf_dev_token

# .env.prod
KAGGLE_USERNAME=prod_user
KAGGLE_API_KEY=prod_key
HUGGINGFACE_API_TOKEN=hf_prod_token
```

Then load based on environment:
```python
import os
env = os.getenv("ENV", "dev")
creds = initialize_credentials(env_file=f".env.{env}")
```

## Integration with Fine-Tuning Pipeline

The credentials system is automatically integrated into the fine-tuning pipeline:

```python
from cv_ranking_system.finetune_pipeline import FineTuningOrchestrator
from cv_ranking_system.credentials import initialize_credentials

# Initialize credentials
creds = initialize_credentials()

# Create and run orchestrator (credentials are used automatically)
orchestrator = FineTuningOrchestrator(
    config=config,
    credentials_manager=creds
)

orchestrator.setup_environment()
orchestrator.prepare_data()  # Uses credentials for dataset downloads
```

## See Also

- [Fine-Tuning Guide](./FINETUNE_GUIDE.md)
- [Data Normalization Guide](./DATA_NORMALIZATION.md)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
