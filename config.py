# config.py

# --- Models ---

# Smallish Marian models for EN->ZH and EN->JA
# (Helsinki-NLP Opus-MT models) :contentReference[oaicite:3]{index=3}
MARIAN_MODELS = {
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "en-ja": "Helsinki-NLP/opus-mt-en-jap",
}

# --- Dataset ---

# We’ll use OPUS-100, which has configs `en-zh` and `en-ja`
# (1M sentence pairs each, English-centric). :contentReference[oaicite:4]{index=4}
DATASET_NAME = "Helsinki-NLP/opus-100"
DATASET_CONFIGS = {
    "en-zh": "en-zh",
    "en-ja": "en-ja",
}

# For OPUS-100, each row is: {"translation": {"en": "...", "zh": "..."}} etc.
LANG_KEYS = {
    "en-zh": ("en", "zh"),
    "en-ja": ("en", "ja"),
}

# --- Training hyperparameters (shared across language pairs) ---

MAX_LENGTH = 128          # max tokens for src/tgt
BATCH_SIZE = 16
LR = 3e-5
NUM_EPOCHS = 3            # start small; can increase later
WEIGHT_DECAY = 0.01

EVAL_STEPS = 10000         # eval every N steps
SAVE_STEPS = 10000
LOGGING_STEPS = 200

# To keep disk usage low:
SAVE_TOTAL_LIMIT = 1      # only keep the most recent checkpoint
