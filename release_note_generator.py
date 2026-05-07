import os
import re
import json
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from github import Github
import requests
from dotenv import load_dotenv

load_dotenv()

TICKET_PATTERN = re.compile(r'([A-Z]+-\d+)')
BOILERPLATE_PATTERN = re.compile(r'(Closes|#\d+|Fixes|Resolved)', re.IGNORECASE)

CATEGORIES = {
    'feat': 'New Features',
    'fix': 'Bug Fixes',
    'enh': 'Improvements',
}

BRANCH_CATEGORY_MAP = {
    'feat': 'New Features',
    'feature': 'New Features',
    'fix': 'Bug Fixes',
    'bugfix': 'Bug Fixes',
    'enh': 'Improvements',
    'enhancement': 'Improvements',
}


def get_timezone_date():
    return datetime.now()


def get_week_range():
    now = get_timezone_date()
    start = now - timedelta(days=now.weekday())
    end = start + timedelta(days=6)
    return start, end


def extract_ticket_ids(text: str) -> list[str]:
    return TICKET_PATTERN.findall(text)


def clean_description(body: str) -> str:
    if not body:
        return ""
    lines = body.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line or BOILERPLATE_PATTERN.search(line):
            continue
        cleaned_lines.append(line)
    text = ' '.join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:200] if text else ""


def summarize_description(body: str) -> str:
    cleaned = clean_description(body)
    if not cleaned:
        return ""
    sentences = re.split(r'[.!?]+', cleaned)
    first_sentence = sentences[0].strip() if sentences else ""

    first_sentence = re.sub(r'^(we have|we added|users can|now you can|should now)', '', first_sentence, flags=re.IGNORECASE).strip()
    first_sentence = first_sentence[0].upper() + first_sentence[1:] if first_sentence else first_sentence

    # Let LLM handle truncation naturally - don't artificially truncate
    return first_sentence


def call_llm_summarize(text: str, llm_api_url: str, llm_api_key: str) -> Optional[str]:
    if not text or not llm_api_url or not llm_api_key:
        return None

    try:
        headers = {"api-key": llm_api_key, "Content-Type": "application/json"}
        payload = {"model": "gpt-5.4", "messages": [{"role": "user", "content": f"""You are writing release notes for NON-TECHNICAL USERS (customers).

CRITICAL RULES:
- NEVER use these words: migration, script, API, backend, frontend, endpoint, PR, commit, branch, repo, database, ticket, convert, fix/, feat/, chore/
- Write ONLY about what the USER can SEE or DO after this change
- Keep it to 1 sentence, maximum 15 words
- If no user-facing change exists, respond with exactly: INTERNAL

Input: {text}

Output (user-friendly, no technical jargon):"""}], "max_completion_tokens": 60}
        resp = requests.post(llm_api_url, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()

        # Try several common response shapes
        if isinstance(data, dict):
            for key in ("summary", "result", "output", "text"):
                if key in data and isinstance(data[key], str):
                    result = data[key].strip()
                    # Validate output - if it contains technical jargon, reject it
                    if any(word in result.lower() for word in ['migration', 'script', 'api', 'backend', 'repo', 'ticket', 'fix/', 'feat/']):
                        return None
                    return result

            # OpenAI-like
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                c0 = data["choices"][0]
                if isinstance(c0, dict):
                    if "text" in c0 and isinstance(c0["text"], str):
                        result = c0["text"].strip()
                        if any(word in result.lower() for word in ['migration', 'script', 'api', 'backend', 'repo', 'ticket']):
                            return None
                        return result
                    if "message" in c0 and isinstance(c0["message"], dict):
                        msg = c0["message"].get("content")
                        if isinstance(msg, str):
                            if any(word in msg.lower() for word in ['migration', 'script', 'api', 'backend', 'repo', 'ticket']):
                                return None
                            return msg.strip()

            # Some hosts return results array
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                r0 = data["results"][0]
                if isinstance(r0, dict):
                    for key in ("output", "content", "text"):
                        if key in r0 and isinstance(r0[key], str):
                            result = r0[key].strip()
                            if any(word in result.lower() for word in ['migration', 'script', 'api', 'backend', 'repo', 'ticket']):
                                return None
                            return result

        # Last resort: try plain text body
        text_body = resp.text
        if text_body:
            text_body = text_body.strip()
            if any(word in text_body.lower() for word in ['migration', 'script', 'api', 'backend', 'repo', 'ticket']):
                return None
            return text_body[:200]

    except Exception:
        return None

    return None


def call_llm_make_title(text: str, llm_api_url: str, llm_api_key: str) -> Optional[str]:
    if not text or not llm_api_url or not llm_api_key:
        return None

    try:
        headers = {"api-key": llm_api_key, "Content-Type": "application/json"}
        payload = {"model": "gpt-5.4", "messages": [{"role": "user", "content": f"""Rewrite this as a SHORT (5-8 words), user-friendly title.
CRITICAL: Remove ALL technical terms: migration, script, ticket, fix/, feat/, chore/, API, backend, endpoint, PR, commit, branch, repo, refactor, convert.
Focus on WHAT THE USER SEES OR CAN DO.
Write like you're telling a regular customer what's new.

Examples:
- "Add migration script for subscriptions" → "Cancel expired subscriptions automatically"
- "Fix/ticket convert copy attachments" → "Copy attachments when converting"

Text: {text}"""}], "max_completion_tokens": 40}
        resp = requests.post(llm_api_url, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()

        # Try several common response shapes
        if isinstance(data, dict):
            for key in ("title", "summary", "result", "output", "text"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()

            # OpenAI-like
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                c0 = data["choices"][0]
                if isinstance(c0, dict):
                    if "text" in c0 and isinstance(c0["text"], str):
                        return c0["text"].strip()
                    if "message" in c0 and isinstance(c0["message"], dict):
                        msg = c0["message"].get("content")
                        if isinstance(msg, str):
                            return msg.strip()

            # Some hosts return results array
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                r0 = data["results"][0]
                if isinstance(r0, dict):
                    for key in ("output", "content", "text"):
                        if key in r0 and isinstance(r0[key], str):
                            return r0[key].strip()

        # Last resort: try plain text body
        text_body = resp.text
        if text_body:
            return text_body.strip()[:100]

    except Exception:
        return None

    return None


SECURITY_PATTERNS = [
    r'password', r'passwd', r'secret', r'token', r'api[_-]?key', r'access[_-]?key',
    r'private[_-]?key', r'credential', r'.env', r'aws[_-]?key', r'aws[_-]?secret',
    r'bearer', r'authorization', r'jwt', r'oauth', r'crypt', r'encrypt', r'decrypt',
    r'signed[_-]?secret', r'client[_-]?secret', r'api[_-]?secret',
]


def is_security_sensitive(text: str) -> bool:
    text_lower = text.lower()
    for pattern in SECURITY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def make_user_friendly_title(title: str, llm_api_url: Optional[str] = None, llm_api_key: Optional[str] = None) -> str:
    # Try LLM first if available
    if llm_api_url and llm_api_key:
        llm_title = call_llm_make_title(title, llm_api_url, llm_api_key)
        if llm_title:
            # Capitalize first letter if LLM didn't
            if llm_title:
                llm_title = llm_title[0].upper() + llm_title[1:] if llm_title else llm_title
            return llm_title

    # Fallback: Aggressively clean the title
    # Remove conventional commit prefixes (including common types)
    title = re.sub(r'^(feat|fix|refactor|enh|chore|docs|style|test)(\([^)]*\))?:\s*', '', title, flags=re.IGNORECASE).strip()

    # Remove technical jargon that is not user-facing
    tech_patterns = [
        r'\bapi\b', r'\bendpoint\b', r'\bbackend\b', r'\bfrontend\b', r'\bpr\b',
        r'\bcommit\b', r'\bdb\b', r'\bdatabase\b', r'\brepo\b', r'\brepository\b',
        r'\bmerge\b', r'\bbranch\b', r'\bpull request\b', r'\bslack\b', r'\bwebhook\b',
        r'\bmigration\b', r'\bscript\b', r'\bticket\b', r'\bconvert\b', r'\bcopy\b',
        r'\bFix\b', r'\bfix/\b', r'\bfeat/\b', r'\bchore/\b'
    ]
    for pattern in tech_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)

    # Remove "Add migration script for..." type patterns
    title = re.sub(r'^(add|update|fix|remove|delete|modify)\s+', '', title, flags=re.IGNORECASE)

    # Clean up extra whitespace and punctuation left behind
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'^\W+|\W+$', '', title).strip()

    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]

    # Fallback if title is empty after processing
    if not title:
        title = "App improvements and updates"

    return title


def get_combined_text(title: str, body: str, commits: str) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if body:
        parts.append(f"Description: {body}")
    if commits:
        parts.append(f"Commits: {commits}")
    return " | ".join(parts)


def generate_user_friendly_internal_summary(title: str) -> str:
    return make_user_friendly_title(title)


def generate_summary(title: str, body: str, commits: str, llm_api_url: Optional[str], llm_api_key: Optional[str]) -> str:
    # Get the user-friendly title first
    friendly_title = make_user_friendly_title(title, llm_api_url, llm_api_key)

    if not friendly_title or friendly_title == "App improvements and updates":
        return ""

    if is_security_sensitive(friendly_title):
        return ""

    # Try LLM to generate a DIFFERENT summary sentence (not the same as title)
    if llm_api_key and llm_api_url:
        llm_input = f"Write a ONE-SENTENCE summary (max 12 words) that is DIFFERENT from the title. Focus on the BENEFIT to users, not just repeating the title. Title: '{friendly_title}'"

        remote = call_llm_summarize(llm_input, llm_api_url, llm_api_key)
        if remote and remote.strip().upper() != "INTERNAL":
            remote_clean = remote.strip()
            # Reject if it's the same as title or generic
            if remote_clean.lower() != friendly_title.lower():
                generic_phrases = ['now available for all users', 'now available', 'improvements and updates',
                                'backend improvements', 'no description', 'internally updated']
                if not is_security_sensitive(remote_clean):
                    if not any(phrase in remote_clean.lower() for phrase in generic_phrases):
                        return remote_clean

    # Fallback: Return empty - just show the title without summary
    return ""


def _make_benefit_from_title(title: str) -> str:
    """Generate a benefit statement that's DIFFERENT from the title."""
    title_lower = title.lower()

    # Pattern: "Cancel X automatically" -> "No manual work needed for X"
    if 'automatically' in title_lower:
        # Extract what's being done automatically
        words = title.split()
        if 'automatically' in words:
            idx = words.index('automatically')
            if idx > 0:
                action = ' '.join(words[:idx])
                return f"No manual action needed to {action}"

    # Pattern: "Copy X when Y" -> "X are now copied automatically when Y"
    if title_lower.startswith('copy '):
        return f"{title} - no manual work needed"

    # Pattern: "Improve X" -> "X now improved"
    if title_lower.startswith('improve '):
        feature = title[8:].strip()
        return f"{feature[0].upper() + feature[1:]} now improved"

    # Pattern: "Fix X" -> "X now fixed"
    if title_lower.startswith('fix '):
        issue = title[4:].strip()
        return f"{issue[0].upper() + issue[1:]} now fixed"

    # Pattern: "Add X" -> "X now available"
    if title_lower.startswith('add '):
        feature = title[4:].strip()
        return f"{feature[0].upper() + feature[1:]} now available"

    # Default: Return a generic benefit that's different from title
    return "No manual action needed"


def _make_benefit_statement(title: str) -> str:
    """Convert a user-friendly title into a benefit statement."""
    title_lower = title.lower()

    # Handle "automatically" patterns - the title itself is the benefit
    if 'automatically' in title_lower:
        return title

    # For specific patterns, create meaningful benefit statements
    if 'cancel' in title_lower:
        return "Expired subscriptions are now cancelled automatically"

    if 'copy' in title_lower and 'attachment' in title_lower:
        return "Attachments are now copied when converting"

    if 'copy' in title_lower:
        return "Items are now copied automatically"

    # If title starts with a verb, create a benefit statement
    verbs = ['cancel', 'copy', 'add', 'update', 'fix', 'delete', 'remove', 'create', 'edit', 'manage', 'view']
    for verb in verbs:
        if title_lower.startswith(verb):
            # Convert verb to past tense and make it a benefit
            past_tense = {
                'cancel': 'cancelled', 'copy': 'copied', 'add': 'added', 'update': 'updated',
                'fix': 'fixed', 'delete': 'deleted', 'remove': 'removed', 'create': 'created',
                'edit': 'edited', 'manage': 'managed', 'view': 'now visible'
            }
            rest_of_title = title[len(verb):].strip()
            if verb in past_tense:
                return f"{rest_of_title[0].upper() + rest_of_title[1:] if rest_of_title else rest_of_title} {past_tense[verb]}"

    # Default: Use the title itself (it's already user-friendly)
    return title


def categorize_pr(pr_branch: str, pr_title: str, pr_labels: list) -> str:
    if pr_branch:
        prefix = pr_branch.split('/')[0].lower()
        if prefix in BRANCH_CATEGORY_MAP:
            return BRANCH_CATEGORY_MAP[prefix]
        if prefix.startswith('feat'):
            return 'New Features'
        if prefix.startswith('fix') or prefix.startswith('bug'):
            return 'Bug Fixes'

    for label in pr_labels:
        label_name = label.name.lower()
        if label_name in ['feature', 'feat', 'new', 'feature-request']:
            return 'New Features'
        elif label_name in ['bug', 'fix', 'bugfix', 'bug-fix', 'hotfix']:
            return 'Bug Fixes'
        elif label_name in ['enhancement', 'enh', 'improve', 'improvement']:
            return 'Improvements'

    title_lower = pr_title.lower()
    if any(x in title_lower for x in ['feature', 'add ', 'new ', 'introduc']) or title_lower.startswith('feat'):
        return 'New Features'
    elif any(x in title_lower for x in ['fix', 'bug', 'hotfix', 'resolve']):
        return 'Bug Fixes'
    elif any(x in title_lower for x in ['enhanc', 'improv', 'update', 'refactor', 'optimiz', 'better']):
        return 'Improvements'

    return 'Other'


def get_branch_name_from_refs(refs):
    try:
        for ref in refs:
            if ref.ref:
                return ref.ref
    except:
        pass
    return ""


def fetch_pr_commits(pr) -> dict:
    commits_text = ""
    try:
        commits = pr.get_commits()
        for commit in commits:
            if commit.commit and commit.commit.message:
                commits_text += commit.commit.message + "\n"
    except:
        pass
    return {
        'number': pr.number,
        'title': pr.title,
        'body': pr.body or "",
        'commits': commits_text.strip(),
        'branch': pr.head.ref if pr.head else "",
        'labels': [label for label in pr.labels],
        'url': pr.html_url,
    }


def fetch_merged_prs(repo, start_date: datetime, end_date: datetime) -> list[dict]:
    print(f"Fetching merged PRs from {start_date.date()} to {end_date.date()}...")
    prs = []
    try:
        pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')
        matching_prs = []
        for pr in pulls:
            if pr.merged_at and start_date <= pr.merged_at.replace(tzinfo=None) <= end_date:
                matching_prs.append(pr)
        print(f"Found {len(matching_prs)} merged PRs. Fetching commits in parallel...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_pr_commits, pr): pr for pr in matching_prs}
            for i, future in enumerate(as_completed(futures)):
                prs.append(future.result())
                print(f"  Processed {i+1}/{len(matching_prs)} PRs")
    except Exception as e:
        print(f"Error fetching PRs: {e}")
    return prs


def group_by_ticket(prs: list[dict]) -> dict:
    grouped = {}
    for pr in prs:
        ticket_ids = extract_ticket_ids(pr['branch'] + pr['title'] + pr['body'])
        if ticket_ids:
            ticket_id = ticket_ids[0]
            if ticket_id not in grouped:
                grouped[ticket_id] = []
            grouped[ticket_id].append(pr)
        else:
            key = f"no-ticket-{pr['number']}"
            grouped[key] = [pr]
    return grouped


def extract_key_terms(text: str) -> set[str]:
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
                 'those', 'i', 'we', 'you', 'they', 'he', 'she', 'it', 'my', 'our',
                 'your', 'their', 'its', 'add', 'added', 'support', 'feature', 'new',
                 'using', 'also', 'now', 'updated', 'update', 'fix', 'fixed', 'bug'}
    if not text:
        return set()
    words = re.findall(r'[a-z]{3,}', text.lower())
    return {w for w in words if w not in stopwords}


def group_by_title_similarity(entries: list[dict]) -> list[list[dict]]:
    if not entries:
        return []

    groups = []
    used = set()

    def is_duplicate(entry1: dict, entry2: dict) -> bool:
        title1 = re.sub(r'[^a-z0-9]', '', entry1['title'].lower().strip())
        title2 = re.sub(r'[^a-z0-9]', '', entry2['title'].lower().strip())
        
        if title1 == title2 or title1 in title2 or title2 in title1:
            return True
        
        terms1 = extract_key_terms(entry1.get('summary', '') + ' ' + entry1.get('title', ''))
        terms2 = extract_key_terms(entry2.get('summary', '') + ' ' + entry2.get('title', ''))
        
        if len(terms1) >= 3 and len(terms2) >= 3:
            common = terms1 & terms2
            if len(common) >= 3:
                return True
        
        return False

    for i, entry in enumerate(entries):
        if i in used:
            continue

        group = [entry]
        used.add(i)

        for j, other in enumerate(entries):
            if j in used:
                continue

            if is_duplicate(entry, other):
                group.append(other)
                used.add(j)

        groups.append(group)

    return groups


def process_single_pr(pr: dict, llm_api_url: Optional[str], llm_api_key: Optional[str]) -> dict:
    summary = generate_summary(pr['title'], pr['body'], pr.get('commits', ''), llm_api_url, llm_api_key)
    category = categorize_pr(pr['branch'], pr['title'], pr['labels'])
    friendly_title = make_user_friendly_title(pr['title'], llm_api_url, llm_api_key)
    return {
        'title': pr['title'],
        'friendly_title': friendly_title,
        'summary': summary,
        'category': category,
        'prs': [pr],
    }


def process_prs(fe_prs: list[dict], be_prs: list[dict], llm_api_url: Optional[str] = None, llm_api_key: Optional[str] = None) -> dict:
    print("Processing PRs...")
    all_prs = []
    for pr in fe_prs + be_prs:
        pr['repo'] = 'FE' if pr.get('url', '').find('/ah-client/') >= 0 else 'BE'
    all_prs.extend(fe_prs)
    all_prs.extend(be_prs)

    result = {cat: [] for cat in CATEGORIES.values()}

    ticket_grouped = group_by_ticket(all_prs)
    entries_to_process = []
    seen_titles = set()

    for ticket_id, pr_group in ticket_grouped.items():
        combined_title = pr_group[0]['title']
        title_key = re.sub(r'[^a-z0-9]', '', combined_title.lower().strip())
        
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        
        merged_commits = "\n".join([p.get('commits', '') for p in pr_group if p.get('commits')])
        merged_body = " | ".join([p.get('body', '') for p in pr_group if p.get('body')])
        
        entries_to_process.append({
            'title': combined_title,
            'body': merged_body,
            'commits': merged_commits,
            'branch': pr_group[0]['branch'],
            'labels': pr_group[0]['labels'],
            'prs': pr_group,
        })

    print(f"Generating summaries for {len(entries_to_process)} entries using LLM...")
    if llm_api_key and llm_api_url:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_single_pr, pr, llm_api_url, llm_api_key): pr for pr in entries_to_process}
            for i, future in enumerate(as_completed(futures)):
                entry = future.result()
                result[entry['category']].append(entry)
                print(f"  Generated {i+1}/{len(entries_to_process)} summaries")
    else:
        for pr in entries_to_process:
            entry = process_single_pr(pr, llm_api_url, llm_api_key)
            result[entry['category']].append(entry)

    for category in result:
        result[category] = group_by_title_similarity(result[category])

    filtered_result = {k: v for k, v in result.items() if k != 'Other'}
    
    return filtered_result


def format_release_message(grouped_prs: dict, week_start: datetime, week_end: datetime) -> str:
    week_str = f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
    message = f"## 📢 Weekly Release Notes ({week_str})\n\n"

    category_order = ['New Features', 'Bug Fixes', 'Improvements']
    category_labels = {
        'New Features': '✨ What\'s New',
        'Bug Fixes': '🐛 Fixed',
        'Improvements': '⚡ Improvements'
    }

    for category in category_order:
        items = grouped_prs.get(category, [])
        if not items:
            continue

        message += f"### {category_labels.get(category, category)}\n"

        for group in items:
            entry = group[0]
            friendly_title = entry.get('friendly_title', make_user_friendly_title(entry['title']))
            summary = entry.get('summary', '').strip()

            if summary and summary != friendly_title:
                message += f"• **{friendly_title}**: {summary}\n\n"
            else:
                message += f"• **{friendly_title}**\n\n"

    return message


def run_generator(
    github_token: str,
    fe_repo_name: str,
    be_repo_name: str,
    llm_api_key: Optional[str] = None,
    llm_api_url: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> str:
    def progress(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(f"[PROGRESS] {msg}")
    
    print("=" * 50)
    print("Starting release note generation...")
    print("=" * 50)
    
    progress("Connecting to GitHub...")
    g = Github(github_token)

    progress("Fetching frontend repository...")
    fe_repo = g.get_repo(fe_repo_name)

    progress("Fetching backend repository...")
    be_repo = g.get_repo(be_repo_name)

    week_start, week_end = get_week_range()

    progress("Fetching frontend PRs...")
    fe_prs = fetch_merged_prs(fe_repo, week_start, week_end)
    print(f"Found {len(fe_prs)} FE PRs")

    progress("Fetching backend PRs...")
    be_prs = fetch_merged_prs(be_repo, week_start, week_end)
    print(f"Found {len(be_prs)} BE PRs")

    if llm_api_key and llm_api_url:
        progress("Generating AI summaries...")
    else:
        progress("Generating summaries...")

    grouped = process_prs(fe_prs, be_prs, llm_api_url=llm_api_url, llm_api_key=llm_api_key)

    progress("Formatting release note...")
    message = format_release_message(grouped, week_start, week_end)

    print("\n" + "=" * 50)
    print("Release note generation complete!")
    print("=" * 50)
    
    progress("Done!")
    
    return message


if __name__ == "__main__":
    import sys

    token = os.environ.get("GITHUB_TOKEN", "")
    fe_repo = os.environ.get("FE_REPO", "")
    be_repo = os.environ.get("BE_REPO", "")
    llm_key = os.environ.get("LLM_API_KEY", "")
    llm_url = os.environ.get("LLM_API_URL", "")

    if not token or not fe_repo or not be_repo:
        print("Missing required environment variables")
        sys.exit(1)

    message = run_generator(token, fe_repo, be_repo, llm_key, llm_url)
    print(message)