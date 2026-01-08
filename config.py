LABEL_MAP = {"fake": 0, "real": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
PROMPT_TEMPLATE = (
"You are a news authenticity analysis expert.\n"
"Please provide a detailed explanation of the reasons why the following news content might be true or false.\n\n"
"News Content:\n{content}\n\n"
"Analysis Reasons:\n"
)
