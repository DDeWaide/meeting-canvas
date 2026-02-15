"""
Meeting Canvas v5 â€” Multi-Agent Capacitor Architecture
N agents with dynamic scaling, specialization shifting, parallel extraction.

Run: uvicorn app:app --reload
Open: http://localhost:8000
"""

import os, json, asyncio, time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from anthropic import Anthropic
import websockets

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "claude-sonnet-4-5-20250929")
MIN_AGENTS = int(os.environ.get("MIN_AGENTS", "3"))
MAX_AGENTS = int(os.environ.get("MAX_AGENTS", "6"))
DISPATCH_EVERY_N = int(os.environ.get("DISPATCH_EVERY_N", "2"))   # utterances
MAX_DISPATCH_GAP = int(os.environ.get("MAX_DISPATCH_GAP", "15"))  # seconds

app = FastAPI(title="Meeting Canvas v5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
last_session = {"diagram": None}

@app.get("/")
async def get_frontend():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# â”€â”€ Prompts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ICON_TYPES = """VISUAL TYPES: warehouse, factory, office, store, hospital, school, hotel, house, restaurant, airport, stadium, church, castle, lighthouse, truck, car, plane, train, ship, bicycle, bus, helicopter, forklift, rocket, port, bridge, road, tower, crane, windmill, dam, pipeline, person, team, doctor, chef, student, worker, baby, audience, database, cloud, server, laptop, phone, monitor, printer, wifi, code, robot, chip, satellite, document, book, folder, email, newspaper, certificate, map, blueprint, money, chart, target, trophy, handshake, briefcase, calendar, presentation, stamp, invoice, plate, cup, bottle, pot, ingredient, cake, pizza, apple, tree, mountain, sun, water, leaf, flower, fire, snowflake, globe, animal, key, lock, camera, music, gift, flag, bell, clock, lightbulb, compass, magnet, anchor, umbrella, toolbox, package, barcode, shield, heart, star, warning, beaker, microscope, atom, dna, pill, idea, alert, question, checkmark, loop, connection, growth, decline, generic"""

BASE_PROMPT = f"""You are a real-time meeting whiteboard assistant. You maintain a LIVING whiteboard.

RULES:
1. CREATE SEPARATE ZONES for each distinct topic, option, workstream, or decision area. Meetings typically have 3-8 zones.
2. REUSE EXISTING ZONES when new content belongs to the same topic. Check the CURRENT WHITEBOARD carefully - add items to existing zones and groups before creating new ones. Only create a new zone when the topic is genuinely distinct.
3. Only add NEW items not already on the whiteboard. Check existing labels carefully.
4. Use "modifications" to update existing items when information changes, not duplicates.
5. CONNECTORS only for explicit cross-zone relationships. Max 3-4 total.
6. Items support hierarchy: "group" (container), "step" (ordered/numbered), "item" (standalone).

HIERARCHY & STRUCTURE:
- When someone describes a PROCESS, WORKFLOW, or SEQUENCE, use a "group" container with "step" children. Number steps via the "order" field (1, 2, 3...). Steps should be concise action phrases.
- When someone lists COMPONENTS, OPTIONS, or CATEGORIES, use a "group" container with "item" children.
- Use "parent" to nest items inside groups. This creates visual hierarchy on the whiteboard.
- Standalone items (no parent) should only be used for one-off facts or entities that don't belong in a group.

DETAIL FIELD:
- Every item MUST have a useful "detail" field - a short plain-English explanation of what this item means in context. Examples: "Chicago warehouse has 40% excess stock", "Sarah owns this by Friday", "Estimated 3-week lead time". This shows on hover.
- Details should give someone who just joined the meeting enough context to understand the item.

ZONE GUIDANCE:
- Each major topic, option, lever, or workstream deserves its own zone
- If a zone has more than 10-12 items, it probably needs to be split
- Use descriptive zone names that capture the specific subtopic

{ICON_TYPES}

OUTPUT - ONLY valid JSON, no fencing:
{{
  "zones": [{{"id":"zone_id","name":"Zone Name","color":"amber|blue|green|purple|red|cyan|pink|teal"}}],
  "items_add": [{{"id":"item_id","label":"Label","visual":"type","zone":"zone_id","detail":"Plain English context explanation","parent":"optional_parent_id","item_type":"group|step|item","order":1}}],
  "items_remove": [],
  "modifications": [{{"id":"item_id","field":"label|visual|detail|zone","value":"new"}}],
  "connectors": [{{"from":"item_id","to":"target_id","label":"rel","style":"solid|dashed"}}],
  "topics": [{{"name":"Topic","relevance":0.9}}],
  "notes": [{{"text":"Important thing","type":"decision|fact|action|number|question"}}],
  "ambiguities": [{{"description":"What's unclear","suggestion":"Question"}}]
}}

If nothing new, return empty arrays. Do NOT recreate existing items."""

SPEC_FOCUS = {
    "structural": "\n\nFOCUS: Prioritize building HIERARCHY. Create groups for related items, use steps (with numbered order) for processes/sequences. Add items to EXISTING zones and groups when they fit - only create new zones for genuinely new topics. Every item needs a clear detail field.",
    "notes":      "\n\nFOCUS: Prioritize capturing decisions, action items, facts, specific numbers, and key statements. Place notes in the most specific zone available. Use detail field to add plain-English context.",
    "relationships": "\n\nFOCUS: Prioritize cross-zone connections, how things relate, and explicit links between items in different zones. Look for cause-effect, dependencies, and sequence relationships.",
    "ambiguity":  "\n\nFOCUS: Prioritize contradictions, disagreements, unclear references, unresolved questions.",
    "generalist": "\n\nFOCUS: Balance between reusing existing zones/groups and creating new ones when needed. Use step items with numbered order for any processes described. Ensure every item has a useful detail field. Capture entities, notes, and relationships.",
}

SUMMARY_PROMPT = """You are an executive meeting summarizer. Produce a clean, professional markdown summary with sections: Overview, Key Topics, Decisions, Action Items, Key Facts, Open Questions, Clarifications Needed, Discussion Summary. Be concise and factual."""


# â”€â”€ Agent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Agent:
    def __init__(self, agent_id, specialization="generalist"):
        self.id = agent_id
        self.specialization = specialization
        self.status = "offline"    # offline, initializing, idle, working, cooldown
        self.task = ""
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.total_latency = 0.0
        self.last_latency = 0.0
        self.items_extracted = 0
        self.notes_extracted = 0

    @property
    def avg_latency(self):
        return self.total_latency / max(self.calls, 1)

    @property
    def cost(self):
        return (self.tokens_in * 0.25 + self.tokens_out * 1.25) / 1_000_000

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "specialization": self.specialization,
            "task": self.task,
            "calls": self.calls,
            "last_latency_ms": int(self.last_latency * 1000),
            "items_extracted": self.items_extracted,
            "notes_extracted": self.notes_extracted,
        }


class AgentPool:
    def __init__(self, min_agents=2, max_agents=6):
        specs = ["structural", "generalist", "notes", "relationships", "ambiguity", "generalist"]
        self.agents = [Agent(i, specs[i % len(specs)]) for i in range(max_agents)]
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.content_counts = {"structural": 0, "notes": 0, "relationships": 0, "ambiguity": 0}
        # Boot minimum agents
        for i in range(min_agents):
            self.agents[i].status = "idle"

    @property
    def online(self):
        return [a for a in self.agents if a.status not in ("offline",)]

    @property
    def available(self):
        return [a for a in self.agents if a.status == "idle"]

    @property
    def working(self):
        return [a for a in self.agents if a.status == "working"]

    @property
    def capacity_pct(self):
        on = len(self.online) or 1
        return int(len(self.available) / on * 100)

    def get_agent(self, preferred=None):
        idle = self.available
        if not idle:
            return None
        if preferred:
            match = [a for a in idle if a.specialization == preferred]
            if match:
                return match[0]
        return idle[0]

    def scale_up(self):
        offline = [a for a in self.agents if a.status == "offline"]
        if offline:
            a = offline[0]
            a.status = "initializing"
            return a
        return None

    def scale_down(self):
        idle = self.available
        if len(self.online) > self.min_agents and len(idle) > 1:
            a = idle[-1]
            a.status = "offline"
            a.task = ""
            return a
        return None

    def respecialize(self):
        """Shift idle agents toward highest-demand content types."""
        total = sum(self.content_counts.values())
        if total < 5:
            return
        demand = sorted(self.content_counts.keys(), key=lambda k: self.content_counts[k], reverse=True)
        for agent in self.available:
            if agent.specialization == "generalist":
                top = demand[0]
                spec_count = sum(1 for a in self.online if a.specialization == top)
                if spec_count < 2:
                    agent.specialization = top

    @property
    def total_cost(self):
        return sum(a.cost for a in self.agents)

    @property
    def total_tokens(self):
        return sum(a.tokens_in + a.tokens_out for a in self.agents)

    @property
    def total_extractions(self):
        return sum(a.calls for a in self.agents)

    @property
    def avg_latency(self):
        on = [a for a in self.agents if a.calls > 0]
        if not on:
            return 0
        return sum(a.avg_latency for a in on) / len(on)

    def to_status(self):
        return {
            "type": "agent_status",
            "agents": [a.to_dict() for a in self.agents],
            "metrics": {
                "online": len(self.online),
                "available": len(self.available),
                "working": len(self.working),
                "capacity_pct": self.capacity_pct,
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost, 4),
                "total_extractions": self.total_extractions,
                "avg_latency_ms": int(self.avg_latency * 1000),
            },
        }


# â”€â”€ Content Classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def classify_content(text):
    t = text.lower()
    scores = {"structural": 0, "notes": 0, "relationships": 0, "ambiguity": 0}
    for w in ["called", "named", "is a", "we have", "there's a", "building", "team", "system", "recipe", "ingredient", "include"]:
        if w in t: scores["structural"] += 1
    for w in ["decided", "decision", "action", "we'll do", "let's go", "agreed", "the number", "percent", "dollars", "deadline"]:
        if w in t: scores["notes"] += 1
    for w in ["connects", "related", "for the", "part of", "goes with", "paired", "along with", "feeds", "linked"]:
        if w in t: scores["relationships"] += 1
    for w in ["unclear", "not sure", "disagree", "what about", "?", "allergic", "can't", "rather not", "don't know", "conflict"]:
        if w in t: scores["ambiguity"] += 1
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "generalist"


# â”€â”€ Diagram State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class DiagramState:
    def __init__(self):
        self.zones = {}
        self.items = {}
        self.connectors = []
        self.topics = []
        self.notes = []
        self.ambiguities = []
        self.transcript = []
        self._lock = asyncio.Lock()

    def _find_zone(self, zid, name):
        if zid in self.zones: return zid
        norm = name.lower().replace("&", "and").strip()
        for eid, ei in self.zones.items():
            en = ei["name"].lower().replace("&", "and").strip()
            # Exact or near-exact match only
            if norm == en or norm in en or en in norm:
                return eid
            # Require very high word overlap (80%+) to consider a match
            wa, wb = set(norm.split()), set(en.split())
            if wa and wb and len(wa) > 1 and len(wb) > 1:
                overlap = len(wa & wb) / min(len(wa), len(wb))
                if overlap >= 0.8:
                    return eid
        return None

    async def merge_update(self, update: dict):
        async with self._lock:
            zmap = {}
            for z in update.get("zones", []):
                if "id" not in z or "name" not in z: continue
                ex = self._find_zone(z["id"], z["name"])
                if ex:
                    zmap[z["id"]] = ex
                else:
                    self.zones[z["id"]] = {"name": z["name"], "color": z.get("color", "amber")}
                    zmap[z["id"]] = z["id"]

            added_items = 0
            for item in update.get("items_add", []):
                if "id" not in item or "label" not in item: continue
                if item["id"] in self.items: continue
                # Check label dedup
                lbl = item["label"].lower().strip()
                if any(v["label"].lower().strip() == lbl for v in self.items.values()): continue
                zone = item.get("zone", "")
                self.items[item["id"]] = {
                    "label": item["label"], "visual": item.get("visual", "generic"),
                    "zone": zmap.get(zone, zone), "detail": item.get("detail", ""),
                    "parent": item.get("parent", ""), "item_type": item.get("item_type", "item"),
                    "order": item.get("order", 0),
                }
                added_items += 1

            for iid in update.get("items_remove", []):
                self.items.pop(iid, None)

            for mod in update.get("modifications", []):
                if mod.get("id") in self.items and "value" in mod:
                    f = mod.get("field", "label")
                    if f in ("label", "visual", "detail", "zone"):
                        v = mod["value"]
                        if f == "zone": v = zmap.get(v, v)
                        self.items[mod["id"]][f] = v

            for conn in update.get("connectors", []):
                if "from" not in conn or "to" not in conn: continue
                if conn["from"] not in self.items: continue
                to = conn["to"]
                if to not in self.items and to not in self.zones:
                    to = zmap.get(to, to)
                    if to not in self.zones: continue
                if not any(c["from"] == conn["from"] and c["to"] == to for c in self.connectors):
                    self.connectors.append({"from": conn["from"], "to": to, "label": conn.get("label", ""), "style": conn.get("style", "solid")})
            if len(self.connectors) > 6:
                self.connectors = self.connectors[-6:]

            if update.get("topics"):
                self.topics = sorted(update["topics"], key=lambda t: t.get("relevance", 0), reverse=True)[:6]

            added_notes = 0
            for n in update.get("notes", []):
                if not any(x.get("text") == n.get("text") for x in self.notes):
                    self.notes.append(n); added_notes += 1
            if len(self.notes) > 50: self.notes = self.notes[-50:]

            for a in update.get("ambiguities", []):
                self.ambiguities.append(a)
            if len(self.ambiguities) > 10: self.ambiguities = self.ambiguities[-10:]

            return added_items, added_notes

    def to_state_summary(self):
        if not self.items: return "EMPTY â€” no whiteboard elements yet."
        parts = ["EXISTING WHITEBOARD (reuse zone IDs, do NOT duplicate):"]
        for zid, zi in self.zones.items():
            zitems = [(i, v) for i, v in self.items.items() if v.get("zone") == zid]
            parts.append(f'\n  ZONE id="{zid}" name="{zi["name"]}" ({len(zitems)} items):')
            groups = [(i, v) for i, v in zitems if v.get("item_type") == "group"]
            for gid, gi in groups:
                parts.append(f'    GROUP id="{gid}" label="{gi["label"]}"')
                kids = sorted([(i, v) for i, v in zitems if v.get("parent") == gid], key=lambda x: x[1].get("order", 0))
                for kid, kv in kids:
                    parts.append(f'      {"STEP" if kv.get("item_type") == "step" else "ITEM"} id="{kid}" label="{kv["label"]}"')
            standalone = [(i, v) for i, v in zitems if v.get("item_type") != "group" and not v.get("parent")]
            for sid, sv in standalone:
                parts.append(f'    ITEM id="{sid}" label="{sv["label"]}"')
        if self.connectors:
            parts.append("\n  CONNECTORS:")
            for c in self.connectors: parts.append(f'    {c["from"]} --{c.get("label", "")}--> {c["to"]}')
        if self.topics: parts.append("\nTopics: " + ", ".join(t["name"] for t in self.topics[:3]))
        return "\n".join(parts)

    def to_frontend(self):
        zones = [{"id": z, "name": v["name"], "color": v["color"]} for z, v in self.zones.items()]
        items = [{"id": i, **{k: v[k] for k in ("label", "visual", "zone", "detail", "parent", "item_type", "order")}} for i, v in self.items.items()]
        return {"zones": zones, "items": items, "connectors": self.connectors, "topics": self.topics, "notes": self.notes[-8:], "ambiguities": self.ambiguities[-3:]}

    def to_summary_data(self):
        parts = ["=== ZONES AND ITEMS ==="]
        for zid, zi in self.zones.items():
            parts.append(f"\nZone: {zi['name']}")
            for iid, v in self.items.items():
                if v.get("zone") == zid:
                    d = f" â€” {v['detail']}" if v.get("detail") else ""
                    parts.append(f"  * {v['label']} ({v['visual']}, {v.get('item_type', 'item')}){d}")
        if self.connectors:
            parts.append("\n=== RELATIONSHIPS ===")
            for c in self.connectors: parts.append(f"  {c['from']} --{c.get('label', '')}--> {c['to']}")
        parts.append("\n=== TOPICS ===")
        for t in self.topics: parts.append(f"  * {t['name']} ({int(t.get('relevance', 0) * 100)}%)")
        parts.append("\n=== NOTES ===")
        for n in self.notes: parts.append(f"  [{n.get('type', 'fact').upper()}] {n['text']}")
        if self.ambiguities:
            parts.append("\n=== AMBIGUITIES ===")
            for a in self.ambiguities: parts.append(f"  ! {a['description']}")
        parts.append("\n=== TRANSCRIPT ===")
        for t in self.transcript: parts.append(f"  [Speaker {t.get('speaker', '?')}]: {t['text']}")
        return "\n".join(parts)


# â”€â”€ WebSocket â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Wait for config message with API keys (frontend always sends this first)
    session_dg_key = DEEPGRAM_API_KEY
    session_anthropic_key = ANTHROPIC_API_KEY

    try:
        initial = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
        if initial.get("action") == "configure":
            if initial.get("deepgram_key", "").strip():
                session_dg_key = initial["deepgram_key"].strip()
            if initial.get("anthropic_key", "").strip():
                session_anthropic_key = initial["anthropic_key"].strip()
    except (asyncio.TimeoutError, Exception):
        pass  # Use env vars

    if not session_dg_key or not session_anthropic_key:
        await ws.send_json({"type": "error", "message": "Missing API keys. Please provide your Deepgram and Anthropic API keys."})
        await ws.close(); return

    diagram = DiagramState()
    pool = AgentPool(MIN_AGENTS, MAX_AGENTS)
    transcript_buffer = []
    dispatch_cursor = 0
    last_dispatch_time = time.time()
    client = Anthropic(api_key=session_anthropic_key)
    ws_open = True

    # Make diagram available to export/summary endpoints immediately (not just on disconnect)
    last_session["diagram"] = diagram
    last_session["pool"] = pool
    last_session["anthropic_key"] = session_anthropic_key

    async def safe_send(msg):
        nonlocal ws_open
        if ws_open:
            try: await ws.send_json(msg)
            except Exception: ws_open = False

    async def broadcast_status():
        await safe_send(pool.to_status())

    async def dispatch_agent(agent, chunk, content_type):
        """Run extraction on an agent with a transcript chunk."""
        agent.status = "working"
        agent.task = f"{len(chunk)} utterances ({content_type})"
        await broadcast_status()

        text = "\n".join(f"[Speaker {u.get('speaker', '?')}]: {u['text']}" for u in chunk)
        state = diagram.to_state_summary()
        spec_extra = SPEC_FOCUS.get(agent.specialization, "")

        t0 = time.time()
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=LLM_MODEL, max_tokens=2000,
                system=BASE_PROMPT + spec_extra,
                messages=[{"role": "user", "content": f"CURRENT WHITEBOARD:\n{state}\n\nNEW TRANSCRIPT:\n{text}\n\nExtract updates. ONLY valid JSON."}]
            )
            elapsed = time.time() - t0
            result = response.content[0].text.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            # Token tracking
            agent.tokens_in += response.usage.input_tokens
            agent.tokens_out += response.usage.output_tokens
            agent.last_latency = elapsed
            agent.total_latency += elapsed
            agent.calls += 1

            update = json.loads(result)

            # Merge into shared state
            added_items, added_notes = await diagram.merge_update(update)
            agent.items_extracted += added_items
            agent.notes_extracted += added_notes

            # Track content distribution for respecialization
            pool.content_counts["structural"] += len(update.get("items_add", []))
            pool.content_counts["notes"] += len(update.get("notes", []))
            pool.content_counts["relationships"] += len(update.get("connectors", []))
            pool.content_counts["ambiguity"] += len(update.get("ambiguities", []))

            print(f">>> Agent {agent.id} ({agent.specialization}): +{added_items}i +{added_notes}n {elapsed:.1f}s")

            # Broadcast canvas update
            await safe_send({"type": "canvas_update", **diagram.to_frontend()})

        except json.JSONDecodeError as e:
            print(f"Agent {agent.id} JSON error: {e}")
        except Exception as e:
            import traceback; traceback.print_exc()
        finally:
            agent.status = "cooldown"
            agent.task = "recharging"
            await broadcast_status()
            await asyncio.sleep(0.8)
            agent.status = "idle"
            agent.task = ""
            await broadcast_status()

    async def orchestrator_loop():
        nonlocal dispatch_cursor, last_dispatch_time
        respec_counter = 0
        while ws_open:
            await asyncio.sleep(1)
            new_count = len(transcript_buffer) - dispatch_cursor
            gap = time.time() - last_dispatch_time

            # Should dispatch?
            should = (
                (new_count >= DISPATCH_EVERY_N and pool.available) or
                (new_count > 0 and gap >= MAX_DISPATCH_GAP and pool.available)
            )

            if should:
                chunk = transcript_buffer[dispatch_cursor:]
                dispatch_cursor = len(transcript_buffer)
                last_dispatch_time = time.time()

                content = " ".join(u["text"] for u in chunk)
                ctype = classify_content(content)

                agent = pool.get_agent(ctype)
                if agent:
                    asyncio.create_task(dispatch_agent(agent, chunk, ctype))

            # Scale up if all online agents are busy and we have any buffered utterances
            if not pool.available and new_count >= 1:
                scaled = pool.scale_up()
                if scaled:
                    print(f">>> Scaling up: Agent {scaled.id} coming online")
                    await broadcast_status()
                    await asyncio.sleep(0.6)
                    scaled.status = "idle"
                    scaled.specialization = classify_content(
                        " ".join(u["text"] for u in transcript_buffer[-10:])
                    ) if transcript_buffer else "generalist"
                    await broadcast_status()

            # Scale down if quiet (longer threshold to keep agents available)
            if new_count == 0 and gap > 45 and len(pool.online) > MIN_AGENTS:
                scaled = pool.scale_down()
                if scaled:
                    print(f">>> Scaling down: Agent {scaled.id} offline")
                    await broadcast_status()

            # Respecialize periodically
            respec_counter += 1
            if respec_counter >= 10:
                pool.respecialize()
                respec_counter = 0

            # Periodic status broadcast
            await broadcast_status()

    orch_task = asyncio.create_task(orchestrator_loop())

    # Deepgram via raw websocket
    dg_url = (
        f"wss://api.deepgram.com/v1/listen?"
        f"model=nova-3&language=en-US&smart_format=true"
        f"&diarize=true&interim_results=false"
        f"&utterances=true&punctuate=true"
        f"&encoding=linear16&sample_rate=16000&channels=1"
    )
    dg_headers = {"Authorization": f"Token {session_dg_key}"}

    try:
        dg_ws = await websockets.connect(dg_url, additional_headers=dg_headers)
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Deepgram connection failed: {e}"})
        await ws.close(); return

    async def dg_receiver():
        """Read transcripts from Deepgram websocket."""
        try:
            async for raw_msg in dg_ws:
                try:
                    msg = json.loads(raw_msg)
                    if msg.get("type") != "Results": continue
                    alt = msg["channel"]["alternatives"][0]
                    sentence = alt.get("transcript", "").strip()
                    if not sentence: continue
                    words = alt.get("words", [])
                    speaker = words[0].get("speaker", 0) if words else 0
                    entry = {"speaker": speaker, "text": sentence, "timestamp": time.time()}
                    transcript_buffer.append(entry)
                    diagram.transcript.append(entry)
                    await safe_send({"type": "transcript", "speaker": speaker, "text": sentence})
                except Exception as e:
                    print(f"Transcript parse error: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("Deepgram connection closed")
        except Exception as e:
            print(f"Deepgram receiver error: {e}")

    dg_recv_task = asyncio.create_task(dg_receiver())

    await safe_send({"type": "status", "message": "Connected â€” start speaking!"})
    await broadcast_status()

    try:
        while True:
            data = await ws.receive()
            if data.get("type") == "websocket.receive":
                if "bytes" in data:
                    await dg_ws.send(data["bytes"])
                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("action") == "extract_now":
                        if transcript_buffer[dispatch_cursor:]:
                            chunk = transcript_buffer[dispatch_cursor:]
                            dispatch_cursor = len(transcript_buffer)
                            last_dispatch_time = time.time()
                            ctype = classify_content(" ".join(u["text"] for u in chunk))
                            agent = pool.get_agent(ctype)
                            if agent:
                                asyncio.create_task(dispatch_agent(agent, chunk, ctype))
                            else:
                                await safe_send({"type": "status", "message": "All agents busy â€” queued"})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        ws_open = False
        last_session["diagram"] = diagram
        last_session["pool"] = pool
        orch_task.cancel()
        dg_recv_task.cancel()
        await dg_ws.close()


@app.get("/health")
async def health():
    return {"status": "ok", "server_keys": bool(DEEPGRAM_API_KEY and ANTHROPIC_API_KEY),
            "model": LLM_MODEL, "summary_model": SUMMARY_MODEL,
            "min_agents": MIN_AGENTS, "max_agents": MAX_AGENTS}


@app.post("/summary")
async def gen_summary():
    d = last_session.get("diagram")
    if not d or not d.items: return {"error": "No meeting data yet."}
    try:
        api_key = last_session.get("anthropic_key", ANTHROPIC_API_KEY)
        c = Anthropic(api_key=api_key)
        r = await asyncio.to_thread(c.messages.create, model=SUMMARY_MODEL, max_tokens=4000,
            system=SUMMARY_PROMPT,
            messages=[{"role": "user", "content": f"Generate executive summary:\n\n{d.to_summary_data()}"}])
        return {"markdown": r.content[0].text.strip()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/metrics")
async def metrics():
    p = last_session.get("pool")
    if not p: return {"error": "No active session."}
    return p.to_status()


# â”€â”€ Diagram Export (Draw.io format â€” importable by Lucidchart, Draw.io, etc.) â”€
from fastapi.responses import Response
import html as html_mod

ZONE_COLORS_HEX = {
    "amber": {"fill": "#FFF8E1", "stroke": "#F59E0B"},
    "blue": {"fill": "#E3F2FD", "stroke": "#3B82F6"},
    "green": {"fill": "#E8F5E9", "stroke": "#22C55E"},
    "purple": {"fill": "#F3E8FF", "stroke": "#A855F7"},
    "red": {"fill": "#FFEBEE", "stroke": "#EF4444"},
    "cyan": {"fill": "#E0F7FA", "stroke": "#06B6D4"},
    "pink": {"fill": "#FCE4EC", "stroke": "#EC4899"},
    "teal": {"fill": "#E0F2F1", "stroke": "#14B8A6"},
}

@app.get("/export/drawio")
async def export_drawio():
    """Export diagram as .drawio XML file â€” importable by Lucidchart, Draw.io, etc."""
    d = last_session.get("diagram")
    if not d or not d.items:
        return {"error": "No meeting data to export."}

    cells = []
    cell_id = 1

    def next_id():
        nonlocal cell_id
        cell_id += 1
        return str(cell_id)

    def esc(text):
        if not text:
            return ""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))

    shape_id_map = {}
    zone_ids = list(d.zones.keys())
    cols = 3
    zone_w, zone_h_base = 420, 300
    pad = 50

    for zi, zid in enumerate(zone_ids):
        zinfo = d.zones[zid]
        col = zi % cols
        row = zi // cols
        zx = pad + col * (zone_w + pad)
        zy = pad + row * 600

        zone_items = [(iid, v) for iid, v in d.items.items() if v.get("zone") == zid]
        zone_h = max(zone_h_base, 80 + len(zone_items) * 52)
        colors = ZONE_COLORS_HEX.get(zinfo.get("color", "amber"), ZONE_COLORS_HEX["amber"])
        zname = esc(zinfo["name"])

        z_cell = next_id()
        shape_id_map[zid] = z_cell
        cells.append(
            f'        <mxCell id="{z_cell}" value="{zname}" '
            f'style="swimlane;startSize=30;fillColor={colors["fill"]};strokeColor={colors["stroke"]};'
            f'rounded=1;arcSize=8;fontStyle=1;fontSize=13;collapsible=0;whiteSpace=wrap;html=1;" '
            f'vertex="1" parent="1">\n'
            f'          <mxGeometry x="{zx}" y="{zy}" width="{zone_w}" height="{zone_h}" as="geometry" />\n'
            f'        </mxCell>'
        )

        iy = 40
        groups = [(i, v) for i, v in zone_items if v.get("item_type") == "group"]
        standalone = [(i, v) for i, v in zone_items if v.get("item_type") != "group" and not v.get("parent")]

        for gid, gi in groups:
            g_cell = next_id()
            shape_id_map[gid] = g_cell
            glabel = esc(gi["label"])
            cells.append(
                f'        <mxCell id="{g_cell}" value="{glabel}" '
                f'style="rounded=1;fillColor={colors["stroke"]};fontColor=#FFFFFF;fontStyle=1;'
                f'strokeColor={colors["stroke"]};fontSize=11;whiteSpace=wrap;html=1;" '
                f'vertex="1" parent="{z_cell}">\n'
                f'          <mxGeometry x="10" y="{iy}" width="{zone_w - 20}" height="32" as="geometry" />\n'
                f'        </mxCell>'
            )
            iy += 38

            children = sorted(
                [(i, v) for i, v in zone_items if v.get("parent") == gid],
                key=lambda x: x[1].get("order", 0)
            )
            for cid, cv in children:
                c_cell = next_id()
                shape_id_map[cid] = c_cell
                prefix = f"{cv.get('order', '')}. " if cv.get("item_type") == "step" else ""
                clabel = esc(f"{prefix}{cv['label']}")
                cells.append(
                    f'        <mxCell id="{c_cell}" value="{clabel}" '
                    f'style="rounded=1;fillColor=#FFFFFF;strokeColor={colors["stroke"]};fontSize=10;whiteSpace=wrap;html=1;" '
                    f'vertex="1" parent="{z_cell}">\n'
                    f'          <mxGeometry x="25" y="{iy}" width="{zone_w - 50}" height="28" as="geometry" />\n'
                    f'        </mxCell>'
                )
                iy += 34

        for sid, sv in standalone:
            s_cell = next_id()
            shape_id_map[sid] = s_cell
            slabel = esc(sv["label"])
            cells.append(
                f'        <mxCell id="{s_cell}" value="{slabel}" '
                f'style="rounded=1;fillColor=#FFFFFF;strokeColor={colors["stroke"]};fontSize=10;whiteSpace=wrap;html=1;" '
                f'vertex="1" parent="{z_cell}">\n'
                f'          <mxGeometry x="10" y="{iy}" width="{zone_w - 20}" height="32" as="geometry" />\n'
                f'        </mxCell>'
            )
            iy += 38

    for ci, conn in enumerate(d.connectors):
        from_id = conn["from"]
        to_id = conn["to"]
        src = shape_id_map.get(from_id)
        tgt = shape_id_map.get(to_id)
        if not src or not tgt:
            continue
        e_cell = next_id()
        elabel = esc(conn.get("label", ""))
        dash = "1" if conn.get("style") == "dashed" else "0"
        cells.append(
            f'        <mxCell id="{e_cell}" value="{elabel}" '
            f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#475569;'
            f'fontSize=9;dashed={dash};html=1;" '
            f'edge="1" source="{src}" target="{tgt}" parent="1">\n'
            f'          <mxGeometry relative="1" as="geometry" />\n'
            f'        </mxCell>'
        )

    if d.notes:
        ny = pad + ((len(zone_ids) - 1) // cols + 1) * 600 + pad
        nh_cell = next_id()
        cells.append(
            f'        <mxCell id="{nh_cell}" value="Key Notes" '
            f'style="text;fontSize=16;fontStyle=1;fillColor=none;strokeColor=none;html=1;" '
            f'vertex="1" parent="1">\n'
            f'          <mxGeometry x="{pad}" y="{ny}" width="200" height="30" as="geometry" />\n'
            f'        </mxCell>'
        )
        ny += 36
        for ni, note in enumerate(d.notes[:15]):
            n_cell = next_id()
            tag = note.get("type", "fact").upper()
            ntext = esc(f"[{tag}] {note['text']}")
            cells.append(
                f'        <mxCell id="{n_cell}" value="{ntext}" '
                f'style="rounded=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontSize=9;align=left;spacingLeft=8;whiteSpace=wrap;html=1;" '
                f'vertex="1" parent="1">\n'
                f'          <mxGeometry x="{pad}" y="{ny}" width="700" height="26" as="geometry" />\n'
                f'        </mxCell>'
            )
            ny += 30

    cells_xml = "\n".join(cells)

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2025-01-01T00:00:00.000Z" agent="5.0" version="24.0.0" type="device">
  <diagram id="mc_export" name="Meeting Canvas Export">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{cells_xml}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=meeting-canvas-export.drawio"}
    )
