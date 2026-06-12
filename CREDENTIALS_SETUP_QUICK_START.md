# Credentials Setup for CV Ranking System

## Overview

This document explains how to configure Kaggle and HuggingFace credentials for dataset downloads in the CV Ranking System's fine-tuning pipeline.

**Important:** The system does NOT use machine-level credentials. You explicitly configure credentials for this project via `.env` file.

## Quick Setup (30 seconds)

```bash
# 1. Copy example file
cp .env.example .env

# 2. Get credentials
# - Kaggle: https://www.kaggle.com/settings/account → API → Create New Token
# - HuggingFace: https://huggingface.co/settings/tokens → New Token

# 3. Edit .env with your credentials
# KAGGLE_USERNAME=your_username
# KAGGLE_API_KEY=your_key
# HUGGINGFACE_API_TOKEN=hf_xxxxx

# 4. Done! The pipeline will use these credentials automatically
```

## Interactive Setup (Recommended)

For an interactive guided setup:

```bash
python setup_credentials.py
```

This script will:
1. Prompt you for Kaggle credentials
2. Prompt you for HuggingFace token
3. Create/update `.env` file
4. Verify credentials work
5. Show setup status

## Manual Setup

### Step 1: Create `.env` file

Copy the template:
```bash
cp .env.example .env
```

### Step 2: Get Kaggle Credentials

1. Visit [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to **API** section
3. Click **Create New API Token**
4. Opens `kaggle.json` file - open it in text editor
5. Copy `username` and `key` values to `.env`:

```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_KEY=your_kaggle_api_key
```

### Step 3: Get HuggingFace Token (Optional)

1. Visit [HuggingFace Access Tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Select **read** permission
4. Click **Create token**
5. Copy token to `.env`:

```bash
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 4: Verify

Test your setup:
```python
from cv_ranking_system.credentials import get_credentials_manager

creds = get_credentials_manager()
print(creds.get_status())
```

Expected output:
```
{
  'kaggle': {'configured': True, 'valid': True, 'username': 'your_username'},
  'huggingface': {'configured': True, 'valid': True},
  'env_file': '/path/to/.env'
}
```

## Using in Code

### Automatic (Recommended)

The pipeline automatically loads credentials:

```python
from cv_ranking_system.finetune_pipeline import FineTuningOrchestrator

# Credentials are loaded from .env automatically
orchestrator = FineTuningOrchestrator(config)
orchestrator.prepare_data()  # Uses credentials for downloads
```

### Manual

```python
from cv_ranking_system.credentials import initialize_credentials

# Initialize with default .env
creds = initialize_credentials()

# Or custom path
creds = initialize_credentials(env_file="/custom/path/.env")

# Use in dataset manager
from cv_ranking_system.dataset_manager import DatasetManager

manager = DatasetManager(credentials_manager=creds)
```

## File Structure

```
project/
├── .env                           # Your credentials (do NOT commit)
├── .env.example                   # Template (commit this)
├── setup_credentials.py           # Interactive setup script
├── CREDENTIALS_SETUP.md           # Detailed guide (this file)
└── src/cv_ranking_system/
    └── credentials.py             # Credentials management module
```

## Troubleshooting

### "Credentials not found" Error

**Problem:** Get error when downloading datasets

**Solution:**
1. Check `.env` exists in project root:
   ```bash
   ls -la .env
   ```

2. Check credentials are set:
   ```bash
   cat .env
   ```

3. If missing, run setup:
   ```bash
   python setup_credentials.py
   ```

### "Kaggle API error" During Download

**Problem:** Dataset download fails with Kaggle error

**Causes & Solutions:**

1. **Wrong API key format**
   - Get new token from Kaggle (regenerates the key)
   - Update `.env`

2. **Dataset doesn't exist**
   - Check dataset name in catalog
   - Visit https://www.kaggle.com/datasets

3. **API rate limit**
   - Wait a few minutes and retry
   - Check [Kaggle API status](https://www.kaggle.com/settings/account)

### "HuggingFace authentication failed" Warning

**Problem:** Get warning about HuggingFace credentials

**Causes & Solutions:**

1. **Token not set** (for private datasets)
   - Get token from https://huggingface.co/settings/tokens
   - Add to `.env`

2. **Invalid token**
   - Regenerate token at HuggingFace
   - Update `.env`

3. **Public datasets** (no token needed)
   - Warning is informational only
   - Downloads still work for public datasets

## Security Best Practices

✅ **Do:**
- Keep `.env` in `.gitignore`
- Use different credentials for dev/prod
- Rotate tokens periodically
- Limit token permissions

❌ **Don't:**
- Commit `.env` file
- Share credentials
- Use personal account tokens in production
- Set weak credentials

## File Locations

The system looks for `.env` in this order:
1. Current working directory
2. Parent directories (up to project root)
3. Project root directory
4. Specify custom: `initialize_credentials(env_file="/path/.env")`

## Integration Points

### Dataset Manager
```python
from cv_ranking_system.dataset_manager import DatasetManager

manager = DatasetManager()  # Credentials auto-loaded
datasets = manager.download_extraction_datasets()
```

### Fine-Tuning Pipeline
```python
from cv_ranking_system.finetune_pipeline import FineTuningOrchestrator

orchestrator = FineTuningOrchestrator(config)
orchestrator.prepare_data()  # Uses credentials for downloads
```

### Direct Access
```python
from cv_ranking_system.credentials import get_credentials_manager

creds = get_credentials_manager()
kaggle_creds = creds.get_kaggle_credentials()
hf_creds = creds.get_huggingface_credentials()
```

## Advanced: Multiple Credentials Profiles

For different environments (dev/prod):

```bash
# Development
cp .env.example .env.dev
# (edit with dev credentials)

# Production  
cp .env.example .env.prod
# (edit with prod credentials)
```

Then load based on environment:

```python
import os
from cv_ranking_system.credentials import initialize_credentials

env = os.getenv("ENVIRONMENT", "dev")
creds = initialize_credentials(env_file=f".env.{env}")
```

Run with:
```bash
ENVIRONMENT=prod python train.py
```

## FAQ

**Q: Do I need both Kaggle and HuggingFace credentials?**
A: No, only if you're downloading from those sources. Extraction datasets are on Kaggle/GitHub. Ranking datasets are on HuggingFace.

**Q: Can I use system-level credentials?**
A: Not with this setup. We explicitly use `.env` to avoid conflicts. This is intentional.

**Q: How do I know if credentials are working?**
A: Run `python setup_credentials.py` or check with:
```python
from cv_ranking_system.credentials import get_credentials_manager
print(get_credentials_manager().get_status())
```

**Q: Can I set credentials as environment variables instead?**
A: Yes! Both work:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_API_KEY=your_key
export HUGGINGFACE_API_TOKEN=hf_xxxxx

# Python will use these if .env is not found
```

**Q: What if I forget my API key?**
A: Generate a new one from the platform:
- Kaggle: https://www.kaggle.com/settings/account (regenerate)
- HuggingFace: https://huggingface.co/settings/tokens (create new)

**Q: Is the `.env` file safe?**
A: Use standard security practices:
- Add to `.gitignore` (don't commit)
- Restrict file permissions: `chmod 600 .env`
- Use different credentials for different environments

## Related Documentation

- [Fine-Tuning Guide](./FINETUNE_GUIDE.md) - How to train models
- [Data Normalization Guide](./DATA_NORMALIZATION.md) - Data preprocessing
- [Kaggle API Docs](https://github.com/Kaggle/kaggle-api)
- [HuggingFace Hub Docs](https://huggingface.co/docs/hub/index)

## Support

For issues:
1. Run `python setup_credentials.py` for status check
2. Check `.env` file format matches `.env.example`
3. Verify credentials at their source platforms
4. Check documentation links above
5. Review error messages carefully for specific issues
