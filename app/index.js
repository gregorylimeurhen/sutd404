"use strict"

const ui = {
	input: document.querySelector("#input input"),
	output: document.getElementById("output"),
}

const DELAY = 75

let busy = false
let ready = false
let shownText = ""
let queuedText = null
let timer = 0
let worker = null


function render(text, rows) {
	const frag = document.createDocumentFragment()
	for (const row of rows) {
		const item = document.createElement("div")
		const name = document.createElement("h2")
		const address = document.createElement("p")
		name.textContent = row[0]
		address.textContent = row[1]
		item.append(name, address)
		frag.append(item)
	}
	shownText = text
	ui.output.replaceChildren(frag)
}


function flush() {
	if (!worker || !ready || busy || queuedText == null) {
		return
	}
	const text = queuedText
	queuedText = null
	if (!text) {
		if (shownText) {
			render("", [])
		}
		return
	}
	if (text === shownText) {
		return
	}
	busy = true
	worker.postMessage({text, type: "solve"})
}


function update() {
	if (!ready) {
		return
	}
	queuedText = ui.input.value
	flush()
}


function queueUpdate() {
	if (timer) {
		clearTimeout(timer)
		timer = 0
	}
	if (!ui.input.value) {
		update()
		return
	}
	timer = setTimeout(() => {
		timer = 0
		update()
	}, DELAY)
}


function boot() {
	worker = new Worker("./worker.js")
	worker.addEventListener("message", ({data}) => {
		if (!data || typeof data !== "object") {
			return
		}
		if (data.type === "ready") {
			ready = true
			ui.input.disabled = false
			ui.input.focus()
			update()
			return
		}
		if (data.type === "result") {
			busy = false
			if (data.text === ui.input.value) {
				render(data.text, data.rows || [])
			} else {
				queuedText = ui.input.value
			}
			flush()
			return
		}
		if (data.type === "error") {
			busy = false
			if (data.text === ui.input.value) {
				render("", [])
			} else {
				queuedText = ui.input.value
			}
			flush()
		}
	})
	worker.addEventListener("error", () => {
		busy = false
		render("", [])
	})
}


ui.input.addEventListener("input", queueUpdate)


boot()
