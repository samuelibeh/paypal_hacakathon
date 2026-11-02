"""Look up one contact for real and show every step: what Tavily returned (what the model
reads), the model's raw answer, and what Scout saved. If several people match, you're asked
which ones are your contact, as on the contact's card.

Uses Claude on Bedrock and Tavily (real calls), and a throwaway in-memory database, so nothing
is saved to MongoDB.

    python -m scripts.try_contact_lookup "Maurice Anigbogu" --phone 08123456789
    python -m scripts.try_contact_lookup "Morris Adeyemi" --company "Adeyemi Appliances"
    python -m scripts.try_contact_lookup "Jess" --notes "wedding photographer"
    python -m scripts.try_contact_lookup "Maurice Anigbogu" --your-email you@stu.cu.edu.ng --your-area "Oregun, Ikeja"
"""

import argparse
import sys

import mongomock

from app import config
from app.agents import contact_agent
from app.agents.contact_agent import ContactAgent
from app.embeddings import get_embedder
from app.llm import LLM
from app.store import Store
from app.tools.web_search import web_search

PROMPT_NAMES = {contact_agent.IDENTIFY: "IDENTIFY", contact_agent.DESCRIBE: "DESCRIBE", contact_agent.NOTES: "NOTES"}


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def tavily(query: str, max_results: int = 5) -> list[dict]:
    heading(f"TAVILY SEARCH: {query}")
    results = web_search(query, max_results=max_results)
    if not results:
        print("(no results)")
    for n, r in enumerate(results, 1):
        print(f"\n[Result {n}] {r.get('title', '')}\n  {r.get('url', '')}")
        print("  " + (r.get("content") or "")[:800].replace("\n", " "))  # the model sees up to 800 characters
    return results


class ShowingLLM:
    """Passes calls to Claude and prints what goes in and what comes back."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        name = PROMPT_NAMES.get(system, "prompt")
        heading(f"MODEL INPUT ({name} prompt)")
        print(prompt)
        answer = self.llm.complete(system, prompt, max_tokens=max_tokens)
        heading(f"MODEL ANSWER ({name})")
        print(answer)
        return answer


def show_contact(store: Store, user_id: str) -> dict:
    contact = store.list_contacts(user_id)[0]
    heading("WHAT SCOUT SAVED")
    print(f"status:      {contact.get('status')}")
    print(f"description: {contact.get('profile') or '(none)'}")
    print(f"written by:  {contact.get('profile_source') or '-'}")
    for link in contact.get("links") or []:
        print(f"link:        {link}")
    for p in contact.get("people") or []:
        print(f"\n  person {p['id']} [{p['state']}]: {p['title']}\n    {p['summary']}")
        for link in p["links"]:
            print(f"    {link}")
    return contact


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ₦ and friends in Windows terminals
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--phone")
    parser.add_argument("--email")
    parser.add_argument("--company")
    parser.add_argument("--title")
    parser.add_argument("--notes")
    parser.add_argument("--your-email", help="your own email, a clue about you (school or workplace)")
    parser.add_argument("--your-area", help="where you shop, a clue about you")
    args = parser.parse_args()

    store = Store(client=mongomock.MongoClient())
    # A gmail placeholder gives no clue; a school or work address does
    user_id = store.create_account(args.your_email or "you@gmail.com", "You", None, args.your_area)["user_id"]
    print("Loading the embedding model…")
    agent = ContactAgent(ShowingLLM(LLM(model_id=config.CONTACT_MODEL_ID)), store, get_embedder(), search=tavily)

    contact = {k: getattr(args, k) for k in ("name", "phone", "email", "company", "title", "notes")}
    summary = agent.import_contacts(user_id, [contact])
    status = store.list_contacts(user_id)[0]["status"]
    print(f"\nImported: {summary}. Status: {status}")
    if status == "skipped":
        print("A single name with no notes isn't searched; there's nothing to look up.")
    while agent.look_up_next():
        pass
    contact = show_contact(store, user_id)

    open_people = [p for p in contact.get("people") or [] if p["state"] == "open"]
    if open_people and sys.stdin.isatty():
        picked = input("\nWhich of these is your contact? Numbers separated by commas, or Enter for none of them: ")
        yes = [n.strip() for n in picked.split(",") if n.strip() in {p["id"] for p in open_people}]
        no = [] if yes else [p["id"] for p in open_people]  # unticked ones stay as "Is this also…?"
        agent.confirm_people(user_id, contact["contact_id"], yes, no)
        show_contact(store, user_id)


if __name__ == "__main__":
    main()
