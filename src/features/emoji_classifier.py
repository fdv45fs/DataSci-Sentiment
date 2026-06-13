import re
from typing import Callable

EMOJI_UNICODE_RANGES = [
    ("\U0001F600", "\U0001F64F"),
    ("\U0001F300", "\U0001F5FF"),
    ("\U0001F680", "\U0001F9FF"),
    ("\U00002600", "\U000027BF"),
    ("\U0001FA00", "\U0001FA6F"),
    ("\U0001FA70", "\U0001FAFF"),
    ("\U00002702", "\U000027B0"),
    ("\U0000FE00", "\U0000FE0F"),
    ("\U0001F1E0", "\U0001F1FF"),
]

EMOJI_REGEX = re.compile(
    "[" + "".join(f"{s}-{e}" for s, e in EMOJI_UNICODE_RANGES) + "]",
    flags=re.UNICODE,
)

FACE_POSITIVE_SET = frozenset([
    "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😊", "🙂",
    "🥰", "😍", "😘", "😗", "😙", "😚", "🤩", "😇", "🥳",
    "😌", "😋", "😛", "😜", "🤪", "😎", "🤗",
    "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎",
    "💕", "💞", "💓", "💗", "💖", "💘", "💝", "❣️", "💟",
])

FACE_NEGATIVE_SET = frozenset([
    "😡", "🤬", "😠", "😤", "😒", "🙄", "😑", "😔",
    "😢", "😭", "😓", "😥", "🥺", "😿", "😞", "😟",
    "🤮", "🤢", "😫", "😩", "😖", "😣",
    "💀", "☠️", "🖕",
])

FACE_NEUTRAL_SET = frozenset([
    "😐", "😶", "🤔", "🤨", "🧐", "😏",
    "😕", "😲", "😯", "😳", "😬", "🥴",
    "🤷", "🤦",
])

SYMBOL_POSITIVE_SET = frozenset([
    "👍", "🙏", "💪", "🔥", "✨", "⭐", "🌟", "💯",
    "🎉", "🎊", "🏆", "🥇", "🎶", "🎵", "🌈", "🌺",
    "🌸", "🌻", "🌼", "💐", "✅", "☑️", "👏", "🤝",
    "👌", "🤌", "🫶",
])

SYMBOL_NEGATIVE_SET = frozenset([
    "👎", "💔", "🚫", "❌", "⛔", "🛑", "😱", "💣",
    "⚠️", "❗", "❕", "‼️", "🙅",
])

EMOJI_TYPE_MAP: dict[str, str] = {}

for e in FACE_POSITIVE_SET:
    EMOJI_TYPE_MAP[e] = "face_positive"
for e in FACE_NEGATIVE_SET:
    EMOJI_TYPE_MAP[e] = "face_negative"
for e in FACE_NEUTRAL_SET:
    EMOJI_TYPE_MAP[e] = "face_neutral"
for e in SYMBOL_POSITIVE_SET:
    EMOJI_TYPE_MAP[e] = "symbol_positive"
for e in SYMBOL_NEGATIVE_SET:
    EMOJI_TYPE_MAP[e] = "symbol_negative"

EMOJI_TYPES = ["face_positive", "face_negative", "face_neutral", "symbol_positive", "symbol_negative", "other"]


def classify_emoji(emoji: str) -> str:
    return EMOJI_TYPE_MAP.get(emoji, "other")


def extract_emojis_from_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return EMOJI_REGEX.findall(text)


def count_emojis_by_type(text: str) -> dict[str, int]:
    counts = {t: 0 for t in EMOJI_TYPES}
    for emoji in extract_emojis_from_text(text):
        counts[classify_emoji(emoji)] += 1
    return counts


def has_emoji_type(text: str, emoji_type: str) -> bool:
    for emoji in extract_emojis_from_text(text):
        if classify_emoji(emoji) == emoji_type:
            return True
    return False


def is_emoji_only(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = EMOJI_REGEX.sub("", text).strip()
    return len(stripped) == 0 and len(extract_emojis_from_text(text)) > 0


def build_type_regex_pattern(emoji_set: frozenset) -> str:
    escaped = [re.escape(e) for e in sorted(emoji_set)]
    return "|".join(escaped) if escaped else "$^"
