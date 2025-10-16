from collections import defaultdict
from urllib.parse import urlparse
import pandas as pd

def translate_risk_score(risk_score):
    if risk_score >= 5.0: # HIGH: Score is 5.0 or greater.
        return "HIGH"
    elif risk_score >= 3.0: # MEDIUM: Score is 3.0 to 4.99.
        return "MEDIUM"
    elif risk_score >= 1.0: # LOW: Score is 1.0 to 2.99.
        return "LOW"
    else: # NONE: Score is less than 1.0.
        return "NONE"

def detect_tiered_keywords(case_insensitive_content, tier_3_wordlist):
    results = defaultdict(int)
    # Occurences:
    for keyword in tier_3_wordlist["keywords"]:
        results[keyword] += case_insensitive_content.count(keyword)
    # Replacement:
    for detected_keyword in results.keys():
        case_insensitive_content = case_insensitive_content.replace(detected_keyword, tier_3_wordlist["replace"]*len(detected_keyword))
    
    return case_insensitive_content, sum(results.values())

def unify_capitalization(content, case_insensitive_content):
    '''
    There probably is a better way but this works for now.
    '''
    content_split, case_insensitive_content_split = content.split(), case_insensitive_content.split()
    for word in case_insensitive_content_split:
        for i, original_word in enumerate(content_split):
            if word == original_word.lower():
                case_insensitive_content = case_insensitive_content.replace(word, original_word, 1)
                content_split.pop(i) # Important to remove as same word might appear multiple times and it only selects first found.
                break
    return case_insensitive_content

def detect_urls(content):
    '''
    Good URL catching practice from:
    https://www.geeksforgeeks.org/python/python-check-url-string/
    '''
    words, urls = content.split(), []
    for word in words:
        parsed = urlparse(word)
        if parsed.scheme and parsed.netloc:
            urls.append(word)

    for detected_url in urls:
        content = content.replace(detected_url, "[link removed]")

    return content, len(urls)


def get_smallest_time_diff(time_df):
    smallest_diff = None
    time_df["created_at"] = pd.to_datetime(time_df["created_at"]).sort_values(ascending=True)
    for index, _ in time_df.iterrows(): # In order so we only need to compare to previous
        if index == 0:
            continue
        time_diff = (time_df.at[index, "created_at"] - time_df.at[index-1, "created_at"]).total_seconds()
        if smallest_diff is None or time_diff < smallest_diff:
            smallest_diff = time_diff

    return smallest_diff