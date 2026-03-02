import csv, datetime, hashlib, html, json, re, requests, time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

code_re = re.compile(r"(\d+)\.(\d+)(?:-([0-9]+))?([A-Za-z])?")
code_pattern = re.compile(r"\d+\.\d+(?:-\d+|[A-Za-z])?")

def build_map_search_values(building_numbers, floor_numbers):
	return [f"{building}.{floor}" for building in building_numbers for floor in floor_numbers]

def parse_rows(page_html):
	soup = BeautifulSoup(page_html, "html.parser")
	tbody = soup.select_one("table.js-map-result-table > tbody")
	if tbody is None:
		return []
	rows = []
	for tr in tbody.select("tr"):
		tds = tr.find_all("td", recursive=False)
		if len(tds) < 3:
			continue
		name = html.unescape(" ".join(tds[0].stripped_strings))
		location = html.unescape(" ".join(tds[2].stripped_strings))
		rows.append((name, location))
	return rows

def fetch_html(session, base_url, map_search, timeout_s, tries):
	for attempt in range(tries):
		try:
			response = session.get(base_url, params={"map-search": map_search}, timeout=timeout_s)
			if response.status_code == 429:
				raise RuntimeError("rate_limited")
			if 500 <= response.status_code <= 599:
				raise RuntimeError(f"server_error_{response.status_code}")
			response.raise_for_status()
			return response.text
		except Exception:
			if attempt + 1 >= tries:
				raise
			time.sleep(0.6 * (2 ** attempt))
	raise RuntimeError("unreachable")

def code_sort_key(code_text):
	match = code_re.search(code_text)
	if match is None:
		return (10**9, 10**9, 1, 10**9, 1, "")
	building_number = int(match.group(1))
	room_number = int(match.group(2))
	section_text = match.group(3)
	letter_text = match.group(4) or ""
	section_number = int(section_text) if section_text is not None else 0
	return (
		building_number,
		room_number,
		0 if section_text is None else 1,
		section_number,
		0 if not letter_text else 1,
		letter_text.upper(),
	)

def location_sort_key(location_text):
	codes = [match.group(0) for match in code_re.finditer(location_text)]
	if not codes:
		return ((10**9, 10**9, 1, 10**9, 1, ""), location_text)
	return (min((code_sort_key(code), code) for code in codes)[0], location_text)

def scrape_all(base_url, map_search_values, max_workers=6, timeout_s=20.0, tries=4):
	session = requests.Session()
	session.headers.update({
		"Accept": "text/html,application/xhtml+xml",
		"User-Agent": "Mozilla/5.0",
	})
	seen = set()
	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		futures = [
			pool.submit(fetch_html, session, base_url, map_search, timeout_s, tries)
			for map_search in map_search_values
		]
		for future in as_completed(futures):
			for row in parse_rows(future.result()):
				seen.add(row)
	return sorted(seen, key=lambda row: location_sort_key(row[1]))

def pad_section(code):
	match = re.fullmatch(r"(\d+)\.(\d+)-(\d+)", code)
	if match is None:
		return code
	building, room, section = match.group(1), match.group(2), match.group(3)
	return f"{building}.{room}-{section.zfill(2)}"

def canonicalise_location(location):
	return [pad_section(code) for code in code_pattern.findall(location)]

def sha256_file(path):
	hasher = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()

def write_jsonl(items, jsonl_path):
	with open(jsonl_path, "w", encoding="utf-8", newline="\n") as out_file:
		for item in items:
			out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

def next_output_name(data_dir):
	index = 0
	while True:
		name = f"{index}.jsonl"
		if not (data_dir / name).exists():
			return name
		index += 1

def append_manifest_row(manifest_path, row_dict):
	is_new = not manifest_path.exists()
	with open(manifest_path, "a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["name", "creation_utc", "sha256"])
		if is_new:
			writer.writeheader()
		writer.writerow(row_dict)

if __name__ == "__main__":
	base_url = "https://www.sutd.edu.sg/contact-us/getting-around-sutd/"
	buildings = range(1, 6)
	floors = range(1, 8)
	map_search_values = build_map_search_values(buildings, floors)
	rows = scrape_all(base_url, map_search_values)
	items = [{"name": name, "address": canonicalise_location(location)} for name, location in rows]
	data_dir = Path("./data")
	data_dir.mkdir(parents=True, exist_ok=True)
	output_name = next_output_name(data_dir)
	output_path = data_dir / output_name
	write_jsonl(items, output_path)
	creation_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
	manifest_path = data_dir / "manifest.csv"
	append_manifest_row(manifest_path, {
		"name": output_name,
		"creation_utc": creation_utc,
		"sha256": sha256_file(output_path),
	})
