import httpx

from app.tools.product_image import find_image, image_from_html
from tests.fakes import fake_store

ADDRESSES = {"shop.com": "93.184.216.34", "cdn.shop.com": "93.184.216.35", "intranet": "10.0.0.5", "localhost": "127.0.0.1"}


def resolve(host, port):
    return [(None, None, None, "", (ADDRESSES[host], 0))]


def client(pages: dict):
    """pages: url -> httpx.Response; records the urls fetched."""
    fetched = []

    def handler(request):
        fetched.append(str(request.url))
        return pages.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler)), fetched


def html_page(head: str):
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=f"<html><head>{head}</head></html>")


def test_reads_the_preview_image_in_any_attribute_order():
    page = '<meta content="/img/fridge.jpg?w=500&amp;h=500" property="og:image">'
    assert image_from_html(page, "https://shop.com/p/1") == "https://shop.com/img/fridge.jpg?w=500&h=500"
    assert image_from_html("<meta name='twitter:image' content='https://cdn.shop.com/a.png'>", "https://shop.com/") \
        == "https://cdn.shop.com/a.png"
    assert image_from_html('<meta property="og:title" content="Fridge">', "https://shop.com/") is None


def test_prefers_og_image_over_twitter_image():
    page = '<meta name="twitter:image" content="https://shop.com/t.jpg"><meta property="og:image" content="https://shop.com/og.jpg">'
    assert image_from_html(page, "https://shop.com/") == "https://shop.com/og.jpg"


def test_finds_the_image_after_a_redirect():
    http, fetched = client({
        "https://shop.com/p/1": httpx.Response(301, headers={"location": "/p/1-fridge"}),
        "https://shop.com/p/1-fridge": html_page('<meta property="og:image" content="https://cdn.shop.com/f.jpg">'),
    })
    assert find_image("https://shop.com/p/1", http, resolve) == "https://cdn.shop.com/f.jpg"
    assert fetched == ["https://shop.com/p/1", "https://shop.com/p/1-fridge"]


def test_never_fetches_private_addresses_even_via_redirect():
    http, fetched = client({"https://shop.com/p": httpx.Response(302, headers={"location": "http://intranet/admin"})})
    assert find_image("http://localhost:5000/health", http, resolve) is None
    assert find_image("file:///etc/passwd", http, resolve) is None
    assert find_image("https://shop.com/p", http, resolve) is None
    assert fetched == ["https://shop.com/p"]  # the redirect target was refused, not fetched


def test_pages_that_are_not_html_or_fail_give_no_image():
    http, _ = client({"https://shop.com/a.pdf": httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")})
    assert find_image("https://shop.com/a.pdf", http, resolve) is None
    assert find_image("https://shop.com/missing", http, resolve) is None


def test_store_remembers_the_photo_and_its_absence():
    store = fake_store()
    with_photo, without = store.save_deals("u1", [{"title": "Fridge", "url": "https://shop.com/f"}, {"title": "Freezer"}])
    store.set_deal_image("u1", with_photo["deal_id"], "https://cdn.shop.com/f.jpg")
    store.set_deal_image("u1", without["deal_id"], None)

    assert store.get_deal("u1", with_photo["deal_id"])["image"] == "https://cdn.shop.com/f.jpg"
    assert "image" in store.get_deal("u1", without["deal_id"])  # checked, so not fetched again
