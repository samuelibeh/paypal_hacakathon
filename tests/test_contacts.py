import json

import httpx

from app import contact_sources, main
from app import gmail as gmail_module
from app.agents import contact_agent
from app.agents.contact_agent import ContactAgent
from app.gmail import Gmail
from app.llm import LLM
from tests.fakes import FakeBedrock, FakeEmbedder, fake_store, signed_in, text_reply

IPHONE_VCF = """BEGIN:VCARD
VERSION:3.0
N:Adeyemi;Morris;;;
FN:Morris Adeyemi
ORG:Adeyemi Appliances;
TITLE:Owner
item1.TEL;type=pref:+234 800 000
 0001
EMAIL;type=INTERNET:Morris@AdeyemiAppliances.ng
NOTE:sells fridges\\, ACs
END:VCARD
BEGIN:VCARD
VERSION:3.0
N:;Mum;;;
TEL;type=CELL:0803 111 2222
END:VCARD
"""

GOOGLE_CSV = """First Name,Last Name,E-mail 1 - Value,Phone 1 - Value,Organization Name,Organization Title,Notes
Tunde,Okoro,tunde@slot.ng,+234 811 222 3333 ::: +234 700 000 0000,Slot Systems,Sales manager,
Jessica,Eze,,0800 000 0003,,,wedding photographer
"""


def test_reads_iphone_vcards_and_google_csv():
    morris, mum = contact_sources.parse_file("contacts.vcf", IPHONE_VCF)
    assert morris == {"name": "Morris Adeyemi", "phone": "+234 800 0000001", "email": "morris@adeyemiappliances.ng",
                      "company": "Adeyemi Appliances", "title": "Owner", "notes": "sells fridges, ACs"}
    assert mum["name"] == "Mum" and mum["phone"] == "0803 111 2222"

    tunde, jessica = contact_sources.parse_file("contacts.csv", GOOGLE_CSV)
    assert (tunde["name"], tunde["phone"], tunde["company"], tunde["title"]) == \
        ("Tunde Okoro", "+234 811 222 3333", "Slot Systems", "Sales manager")
    assert jessica["notes"] == "wedding photographer"

    person = {"names": [{"displayName": "Ada Obi"}], "phoneNumbers": [{"value": "0812 000 0000"}],
              "organizations": [{"name": "Jumia", "title": "Buyer"}]}
    assert contact_sources.from_google(person)["company"] == "Jumia"


def test_who_counts_as_the_same_person_and_who_gets_searched():
    assert contact_sources.key({"name": "A", "phone": "+234 803 111 2222"}) == contact_sources.key({"name": "B", "phone": "0803 111 2222"})
    assert contact_sources.full_name("Maurice Anigbogu") and not contact_sources.full_name("Mum")
    assert contact_sources.country_from_phone("08123456789") == "Nigeria"
    assert contact_sources.work_domain("a@gmail.com") is None and contact_sources.work_domain("a@slot.ng") == "slot.ng"


def _helpful(*numbers):
    """The contact model's answer to which shortlisted contacts can help."""
    return json.dumps({"helpful": [{"n": n, "why": f"Reason {n}."} for n in numbers]})


def _agent(store, *replies, search=None):
    client = FakeBedrock(*[text_reply(r) for r in replies])
    return ContactAgent(LLM(client, "m"), store, FakeEmbedder(), search=search or (lambda q, max_results: [])), client


def test_import_saves_everyone_queues_full_names_and_caps_lookups(monkeypatch):
    monkeypatch.setattr(contact_agent, "LOOKUP_CAP", 2)
    store = fake_store()
    agent, client = _agent(store, "Takes wedding photos.")

    summary = agent.import_contacts("u1", [
        {"name": "Maurice Anigbogu"}, {"name": "Tunde Okoro"}, {"name": "Ada Obi"},
        {"name": "Jess", "notes": "wedding photographer"}, {"name": "Mum"},
    ])

    status = {c["name"]: c["status"] for c in store.list_contacts("u1")}
    assert summary == {"saved": 5, "queued": 2, "waiting": 1}
    assert status == {"Maurice Anigbogu": "queued", "Tunde Okoro": "queued", "Ada Obi": "waiting",
                      "Jess": "done", "Mum": "skipped"}  # Jess was described from the note
    assert agent.look_up_more("u1") == 1
    # when each joined the lookup queue, so the app's progress counts only this run
    assert {c["name"] for c in store.list_contacts("u1") if c.get("queued_at")} == {"Maurice Anigbogu", "Tunde Okoro", "Ada Obi"}

    agent.import_contacts("u1", [{"name": "Mum", "phone": "0803 111 2222"}])
    agent.import_contacts("u1", [{"name": "Mum", "phone": "+234 803 111 2222"}])
    assert len(store.list_contacts("u1")) == 6  # the same Mum, imported twice


def test_one_clear_match_is_described_with_its_links_and_can_be_suggested():
    store = fake_store()
    searched = []

    def search(query, max_results):
        searched.append(query)
        return [{"title": "Morris Adeyemi - Owner", "url": "https://adeyemi.ng/about", "content": "Fridges and ACs in Ikeja"}]

    reply = json.dumps({"decision": "match", "description": "Runs Adeyemi Appliances in Ikeja, selling fridges.", "results": [1]})
    agent, _ = _agent(store, reply, _helpful(1), search=search)
    agent.import_contacts("u1", [{"name": "Morris Adeyemi", "company": "Adeyemi Appliances", "phone": "0803 111 2222"}])

    assert agent.look_up_next() and not agent.look_up_next()
    assert searched == ['"Morris Adeyemi" Adeyemi Appliances Nigeria']
    [morris] = store.list_contacts("u1")
    assert (morris["status"], morris["profile_source"], morris["links"]) == ("done", "web", ["https://adeyemi.ng/about"])
    assert [h["name"] for h in agent.find_contacts("u1", "Adeyemi Appliances selling fridges")] == ["Morris Adeyemi"]


def test_several_people_are_asked_about_and_ticked_ones_become_one_description():
    store = fake_store()
    results = [{"title": f"Page {n}", "url": f"https://site{n}.com", "content": "..."} for n in range(1, 5)]
    choose = json.dumps({"decision": "choose", "people": [
        {"title": "Cloud engineer at TradeMindsAI", "summary": "Studied at Covenant University.", "results": [1, 2]},
        {"title": "Speaker at a cloud meetup", "summary": "Talks on cloud costs.", "results": [3]},
        {"title": "Founder of LightEd", "summary": "Runs a solar project in Abuja.", "results": [4]},
    ]})
    agent, client = _agent(store, choose, "Cloud engineer at TradeMindsAI who speaks on cloud costs.", _helpful(1),
                           search=lambda q, max_results: results)
    agent.import_contacts("u1", [{"name": "Maurice Anigbogu"}])
    agent.look_up_next()

    [maurice] = store.list_contacts("u1")
    assert maurice["status"] == "check" and len(maurice["people"]) == 3
    assert agent.find_contacts("u1", "cloud engineer") == []  # not suggested until confirmed

    after = agent.confirm_people("u1", maurice["contact_id"], yes=["1", "2"], no=[])
    assert after["status"] == "done" and after["profile_source"] == "confirmed"
    assert after["links"] == ["https://site1.com", "https://site2.com", "https://site3.com"]
    assert [p["state"] for p in after["people"]] == ["yes", "yes", "no"]  # unticked: not them, nothing left to ask
    assert "Cloud engineer at TradeMindsAI" in client.requests[1]["messages"][0]["content"][0]["text"]
    assert [h["name"] for h in agent.find_contacts("u1", "cloud engineer TradeMindsAI")] == ["Maurice Anigbogu"]


def test_none_of_these_leaves_the_contact_undescribed():
    store = fake_store()
    choose = json.dumps({"decision": "choose", "people": [{"title": "A", "summary": "a", "results": []},
                                                         {"title": "B", "summary": "b", "results": []}]})
    agent, _ = _agent(store, choose)
    agent.import_contacts("u1", [{"name": "Maurice Anigbogu"}])
    agent.look_up_next()
    contact_id = store.list_contacts("u1")[0]["contact_id"]

    after = agent.confirm_people("u1", contact_id, yes=[], no=["1", "2"])
    assert after["status"] == "nothing" and after["profile"] == ""


def test_a_failed_lookup_is_retried_then_given_up():
    store = fake_store()

    def broken(query, max_results):
        raise RuntimeError("search is down")

    agent, _ = _agent(store, search=broken)
    agent.import_contacts("u1", [{"name": "Maurice Anigbogu"}])
    for _ in range(contact_agent.MAX_ATTEMPTS):
        agent.look_up_next()
    assert store.list_contacts("u1")[0]["status"] == "nothing"


def test_google_contacts_are_read_page_by_page():
    pages = {None: {"connections": [{"names": [{"displayName": "Ada Obi"}]}], "nextPageToken": "p2"},
             "p2": {"connections": [{"names": [{"displayName": "Bola Ade"}]}]}}

    def handler(request):
        if str(request.url) == gmail_module.TOKEN_URL:
            return httpx.Response(200, json={"access_token": "tok"})
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, json=pages[request.url.params.get("pageToken")])

    google = Gmail(fake_store(), "cid", "secret", "http://localhost/cb", http=httpx.Client(transport=httpx.MockTransport(handler)))
    state = httpx.URL(google.contacts_url("u1")).params["state"]
    assert google.contacts_user(state) == "u1" and not google.is_sign_in(state)
    user_id, people = google.fetch_contacts("code", state)
    assert user_id == "u1" and [p["names"][0]["displayName"] for p in people] == ["Ada Obi", "Bola Ade"]


def test_contact_routes_are_for_accounts_only(monkeypatch):
    store = fake_store()
    monkeypatch.setattr(main, "get_store", lambda: store)
    client = signed_in(main.app, "u1")

    parsed = client.post("/contacts/u1/parse-file", json={"filename": "c.vcf", "text": IPHONE_VCF}).json()["contacts"]
    assert [c["name"] for c in parsed] == ["Morris Adeyemi", "Mum"]
    assert client.post("/contacts/u1/parse-file", json={"filename": "c.csv", "text": "nothing here"}).status_code == 400

    store.save_pending_import("u1", parsed)
    assert len(client.get("/contacts/u1/pending").json()["contacts"]) == 2
    assert client.get("/contacts/u1/pending").json()["contacts"] == []  # handed over once

    guest = signed_in(main.app, "guest-abc")
    assert guest.post("/contacts/guest-abc/parse-file", json={"filename": "c.vcf", "text": IPHONE_VCF}).status_code == 403


def test_a_number_already_saved_is_asked_about_before_anything_changes():
    store = fake_store()
    agent, _ = _agent(store)
    agent.add_contact("u1", {"name": "Ayo Atunrase", "phone": "0816 543 8884"})

    clash = agent.add_contact("u1", {"name": "Victor Odoh", "phone": "+234 816 543 8884"})
    assert clash["conflict"]["name"] == "Ayo Atunrase" and clash["conflict"]["same"] == "number"
    assert [c["name"] for c in store.list_contacts("u1")] == ["Ayo Atunrase"]  # nothing swapped yet

    agent.add_contact("u1", {"name": "Victor Odoh", "phone": "08165438884"}, on_conflict="keep_both")
    assert [c["name"] for c in store.list_contacts("u1")] == ["Ayo Atunrase", "Victor Odoh"]
    agent.add_contact("u1", {"name": "Tolu Ade", "phone": "08165438884"}, on_conflict="replace")
    assert [c["name"] for c in store.list_contacts("u1")] == ["Tolu Ade", "Victor Odoh"]  # Ayo replaced


def test_the_whole_card_can_be_edited_and_a_clashing_number_is_asked_about():
    store = fake_store()
    agent, _ = _agent(store, _helpful(1))
    agent.add_contact("u1", {"name": "Ayo Atunrase", "phone": "08165438884"})
    agent.add_contact("u1", {"name": "Victor Odoh", "phone": "09087654321"})
    victor = next(c for c in store.list_contacts("u1") if c["name"] == "Victor Odoh")

    edited = agent.edit_contact("u1", victor["contact_id"], {"name": "Victor Odoh Jr", "phone": "09087654321",
                                                              "company": "Slot", "profile": "Sells phones at Slot Ikeja."})
    assert (edited["name"], edited["company"], edited["profile"], edited["profile_source"]) == \
        ("Victor Odoh Jr", "Slot", "Sells phones at Slot Ikeja.", "you")
    assert [h["name"] for h in agent.find_contacts("u1", "Sells phones at Slot Ikeja")] == ["Victor Odoh Jr"]

    clash = agent.edit_contact("u1", victor["contact_id"], {"name": "Victor Odoh Jr", "phone": "08165438884"})
    assert clash["conflict"]["name"] == "Ayo Atunrase"
    assert store.get_contact("u1", victor["contact_id"])["phone"] == "09087654321"  # unchanged until they choose


def _described(store, agent, *people):
    for name, profile in people:
        phone = f"080{len(store.list_contacts('u1')):08d}"
        agent.add_contact("u1", {"name": name, "phone": phone})
        contact = next(c for c in store.list_contacts("u1") if c["name"] == name)
        agent.edit_contact("u1", contact["contact_id"], {"name": name, "phone": phone, "profile": profile})


def test_only_the_contacts_the_model_says_can_help_are_suggested_with_why():
    store = fake_store()
    vet = json.dumps({"helpful": [{"n": 2, "why": "Bakes birthday cakes in Ikeja."}, {"n": 9, "why": "No such contact."},
                                  {"n": 2, "why": "Listed twice."}, {"n": 1, "why": ""}]})
    agent, client = _agent(store, vet)
    _described(store, agent, ("Ayo Ashiru", "Software engineer in Lagos who builds cake ordering apps."),
               ("Tola Ade", "Bakes custom cakes in Ikeja."))

    [tola] = agent.find_contacts("u1", "cake in Ikeja")
    assert tola == {"name": "Tola Ade", "phone": "08000000001", "email": None, "why": "Bakes birthday cakes in Ikeja."}
    asked = client.requests[-1]["messages"][0]["content"][0]["text"]
    assert "Need: cake in Ikeja" in asked and "Ayo Ashiru: Software engineer" in asked  # both shortlisted, one kept


def test_no_contacts_are_suggested_when_the_check_fails():
    store = fake_store()
    agent, _ = _agent(store, "not json at all")
    _described(store, agent, ("Tola Ade", "Bakes custom cakes in Ikeja."))
    assert agent.find_contacts("u1", "cakes in Ikeja") == []

    agent.llm.client.responses = []  # the model is unreachable: nothing, rather than an unchecked guess
    assert agent.find_contacts("u1", "cakes in Ikeja") == []


def test_searching_again_keeps_the_description_unless_something_new_is_found():
    store = fake_store()
    found = json.dumps({"decision": "match", "description": "Runs a phone shop in Ikeja.", "results": []})
    agent, _ = _agent(store, json.dumps({"decision": "none"}), found)
    agent.add_contact("u1", {"name": "Victor Odoh", "notes": "phones"})
    victor = store.list_contacts("u1")[0]
    agent.edit_contact("u1", victor["contact_id"], {"name": "Victor Odoh", "profile": "Sells phones."})

    assert agent.search_again("u1", victor["contact_id"])["status"] == "queued"
    agent.look_up_next()  # nothing new: the description stays
    after = store.get_contact("u1", victor["contact_id"])
    assert (after["status"], after["profile"]) == ("done", "Sells phones.")

    agent.search_again("u1", victor["contact_id"])
    agent.look_up_next()  # a clear match replaces it
    assert store.get_contact("u1", victor["contact_id"])["profile"] == "Runs a phone shop in Ikeja."


def test_contacts_can_be_added_edited_searched_again_and_deleted_from_the_app(monkeypatch):
    store = fake_store()
    agent, _ = _agent(store)
    monkeypatch.setattr(main, "get_store", lambda: store)
    main.app.dependency_overrides[main.get_contact_agent] = lambda: agent
    try:
        client = signed_in(main.app, "u1")
        added = client.post("/contacts/u1/add", json={"contact": {"name": "Ayo Atunrase", "phone": "08165438884"}}).json()
        clash = client.post("/contacts/u1/add", json={"contact": {"name": "Victor Odoh", "phone": "08165438884"}}).json()
        assert clash["conflict"]["contact_id"] == added["contact_id"]

        contact_id = added["contact_id"]
        assert client.put(f"/contacts/u1/{contact_id}", json={"name": "Ayo Bello", "phone": "08165438884"}).json()["name"] == "Ayo Bello"
        assert client.post(f"/contacts/u1/{contact_id}/search-again").status_code == 200
        assert client.delete(f"/contacts/u1/{contact_id}").json() == {"deleted": contact_id}
        assert client.delete(f"/contacts/u1/{contact_id}").status_code == 404
        assert store.list_contacts("u1") == []
    finally:
        main.app.dependency_overrides.pop(main.get_contact_agent, None)
