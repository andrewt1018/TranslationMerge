# config.py

BASE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"

# OPUS-100 column names for targets
LANG_COLUMNS = {
    "en-zh": ("en", "zh"),
    "en-ja": ("en", "ja"),
}

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

### Extra configs for QWEN
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# We still use OPUS-100 language columns
LANG_COLUMNS = {
    "en-ja": ("en", "ja"),
    "en-zh": ("en", "zh"),
}

# Special tokens for the Qwen tokenizer
QWEN_LANG_TAGS = {
    "en-ja": {"src": "<en>", "tgt": "<ja>"},
    "en-zh": {"src": "<en>", "tgt": "<zh>"},
}

QWEN_MAX_SOURCE_LENGTH = 192
QWEN_MAX_TARGET_LENGTH = 192
QWEN_MAX_SEQ_LEN = 400

QWEN_BATCH_SIZE = 4
QWEN_LR = 1e-5
QWEN_NUM_EPOCHS = 3
QWEN_WEIGHT_DECAY = 0.01
QWEN_SAVE_STEPS = 10_000
QWEN_EVAL_STEPS = 10_000
QWEN_LOGGING_STEPS = 500
QWEN_SAVE_TOTAL_LIMIT = 1
QWEN_WARMUP_STEPS = 8_000

