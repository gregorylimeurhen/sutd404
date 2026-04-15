import json
import pathlib
import statistics
import time
import torch
import utils


def beam_top(model, tok, dev, text, trie, width):
	text = text.strip().lower()
	prefix = tok.encode_text(text) + [tok.sep_id]
	kw = {"device": dev, "dtype": torch.long}
	model.eval()
	with torch.inference_mode():
		logits, cache = model.forward_cached(torch.tensor([prefix], **kw))
		beams = [(0.0, [], trie, logits, cache)]
		while True:
			next_beams = []
			active = False
			for score, room, node, logits, cache in beams:
				if node is None:
					next_beams.append((score, room, None, None, None))
					continue
				active = True
				scores = logits[0, -1].log_softmax(dim=-1)
				for token_id in node["allowed"]:
					score2 = score + float(scores[token_id].item())
					if token_id == tok.eos_id:
						next_beams.append((score2, room, None, None, None))
						continue
					step = torch.tensor([[token_id]], **kw)
					logits2, cache2 = model.forward_cached(step, cache)
					room2 = room + [token_id]
					node2 = node["children"][token_id]
					next_beams.append((score2, room2, node2, logits2, cache2))
			if not active:
				break
			next_beams.sort(key=lambda beam: beam[0], reverse=True)
			beams = next_beams[:width]
	return [tok.decode_text(room) for score, room, node, logits, cache in beams]


def score_room(model, tok, dev, logits, cache, room_ids):
	kw = {"device": dev, "dtype": torch.long}
	score = 0.0
	for token_id in room_ids + [tok.eos_id]:
		scores = logits[0, -1].log_softmax(dim=-1)
		score += float(scores[token_id].item())
		if token_id == tok.eos_id:
			return score
		step = torch.tensor([[token_id]], **kw)
		logits, cache = model.forward_cached(step, cache)


def exact_top(model, tok, dev, text, room_ids, k):
	text = text.strip().lower()
	prefix = tok.encode_text(text) + [tok.sep_id]
	kw = {"device": dev, "dtype": torch.long}
	model.eval()
	with torch.inference_mode():
		logits, cache = model.forward_cached(torch.tensor([prefix], **kw))
		scored = []
		for room, ids in room_ids:
			score = score_room(model, tok, dev, logits, cache, ids)
			scored.append((score, room))
	scored.sort(key=lambda item: item[0], reverse=True)
	return [room for score, room in scored[:k]]


def beam_addrs(model, tok, dev, text, trie, rm, rset, width):
	text = text.strip().lower()
	if text in rset:
		return [rm[text]]
	rooms = beam_top(model, tok, dev, text, trie, width)
	return [rm[room] for room in rooms]


def exact_addrs(model, tok, dev, text, room_ids, rm, rset, k):
	text = text.strip().lower()
	if text in rset:
		return [rm[text]]
	rooms = exact_top(model, tok, dev, text, room_ids, k)
	return [rm[room] for room in rooms]


def measure(rows, run, label):
	latencies = []
	total = len(rows)
	for index, row in enumerate(rows, start=1):
		start = time.perf_counter()
		run(row["input"])
		latencies.append(time.perf_counter() - start)
		print(f"{label} {index}/{total}")
	return {
		"mean_latency": sum(latencies) / len(latencies),
		"median_latency": statistics.median(latencies),
	}


def beam_stats(rows, model, tok, dev, trie, rm, rset, width, label):
	run = lambda text: beam_addrs(model, tok, dev, text, trie, rm, rset, width)
	return measure(rows, run, label)


def exact_stats(rows, model, tok, dev, room_ids, rm, rset, k, label):
	run = lambda text: exact_addrs(model, tok, dev, text, room_ids, rm, rset, k)
	return measure(rows, run, label)


def main():
	dir_ = pathlib.Path(__file__).resolve().parent
	cfg = utils.load_config(dir_, "test")
	dev = utils.device_for()
	train_dir = dir_ / "runs" / str(cfg["run"]).strip() / "train"
	snap_path = train_dir / "snapshot.zip"
	model_path = train_dir / "model.pt"
	with utils.loaded_snapshot(snap_path) as (root2, ev):
		model, tok, rooms = ev.load_checkpoint(model_path, dev)
		rows = ev.load_rows(root2, "test")[:10]
		rm = ev.load_room_lookup(root2)
		trie = ev.build_room_trie(rooms, tok)
		rset = set(rooms)
		room_ids = [(room, tok.encode_text(room)) for room in rooms]
		results = {}
		specs = [("beam", 3), ("beam", 5), ("beam", 10)]
		specs += [("exact_top", 3), ("exact_top", 5), ("exact_top", 10)]
		total = len(specs)
		for index, (kind, size) in enumerate(specs, start=1):
			name = f"{kind}_{size}"
			label = f"{index}/{total} {name}"
			print(label)
			if kind == "beam":
				stats = beam_stats(rows, model, tok, dev, trie, rm, rset, size, label)
				results[name] = stats
				continue
			stats = exact_stats(rows, model, tok, dev, room_ids, rm, rset, size, label)
			results[name] = stats
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
