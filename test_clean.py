# test_clean.py  -  quick check that clean_text handles empty and missing input
import pandas as pd

from clean import clean_text

test_cases = [None, "", "  ", "Hello World!", pd.NA, "WIN a FREE prize!!! Visit www.win.com or call 0800123456"]
for t in test_cases:
    print(f"Input: {t!r} -> Output: {clean_text(t)!r}")
