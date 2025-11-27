# config.py

BASE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"

# OPUS-100 column names for targets
LANG_COLUMNS = {
    "en-zh": ("en", "zh"),
    "en-ja": ("en", "ja"),
}

# Marian en-mul requires a target language token at the start of the source
# Model card: "a sentence initial language token is required in the form `>>id<<`" :contentReference[oaicite:0]{index=0}
LANG_TAGS = {
    "en-zh": ">>zho<<",   # Chinese
    "en-ja": ">>jpn<<",   # Japanese
}

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

BATCH_SIZE = 16
LR = 1e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
SAVE_STEPS = 10000
EVAL_STEPS = 10000
LOGGING_STEPS = 500
SAVE_TOTAL_LIMIT = 1
WARMUP_STEPS = 8000
