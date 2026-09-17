# Project status — SchoolAI (samenvatting)

Dit document beschrijft hoe het project op dit moment werkt en welke wijzigingen recent zijn doorgevoerd.

## Overzicht

SchoolAI is een lokale desktop-app die de NebulaONE chat endpoint gebruikt als agentische hulp bij coderen met lees-/schrijfrechten naar een gekozen map (workspace). De app gebruikt SSE (Server-Sent Events) van het endpoint en een kleine tag-taal waarmee het model bestandsbewerkingen kan aanvragen:

- `<READ_FILE>path/to/file</READ_FILE>` — vraag om een bestand te lezen
- `<WRITE_FILE name="path/to/file">...contents...</WRITE_FILE>` — vraag om een bestand te overschrijven

De belangrijkste modules:

- `run.py` — entry point (GUI / selftest / debug-sse)
- `schoolai/config.py` — API constants, token opslag/normalisatie, system prompt
- `schoolai/ui.py` — tkinter GUI, workspace selector, toolbar, tokenveld
- `schoolai/nebula.py` — SSE client + base64 tolerant decoder
- `schoolai/agent.py` — TagStreamParser, AgentHarness, tool executie
- `schoolai/security.py` — workspace confinement + veilige read/write

## Wat is er recent toegevoegd / gewijzigd

Kort samengevat zijn de volgende verbeteringen toegevoegd:

1. Automatische "Login to AI" flow (optioneel)
   - In `schoolai/ui.py` is een nieuwe knop **Login to AI** toegevoegd.
   - Deze knop kan (optioneel) Playwright gebruiken om een zichtbare Chromium-browser te openen
     naar `https://ai-chat.hhs.nl`, de netwerkrequests en `localStorage` te monitoren en
     automatisch een Bearer token (JWT) te detecteren en veilig op te slaan.
   - Als Playwright niet aanwezig is, krijgt de gebruiker instructies om de token handmatig
     te plakken in het tokenveld en op **Save token** te klikken.

2. UI verbeteringen
   - Token-entry heeft nu een show/hide toggle en Enter-to-save binding.
   - Toolbar bevat `Select Workspace`, `Login to AI`, `Save token`, `New chat` en `Stop`.
   - Statusbar toont run-time meldingen (Ready, Thinking…, Logged in, etc.).

3. Token verwerking
   - `schoolai/config.py` bevat een `normalize_bearer_token()`-functie die:
     - automatisch `Bearer ` toevoegt als nodig,
     - een JWT uit geplakte ruis (bv. "<jwt> gebruik de verbeterde code...") kan extraheren.
   - Settings worden veilig opgeslagen in `~/.schoolai/config.json` met permissies `0600`.

4. Achtergrond streaming en threading
   - SSE streaming draait in een worker thread; updates worden naar de GUI gepusht via
     een queue en verwerkt in de Tk hoofdthread met `root.after()`.
   - `nebula.NebulaClient.stream()` decodeert base64 `response-updated` payloads
     en yieldt tekst-chunks die het agent-layer verwerkt.

5. Agentische tools en veiligheid
   - `TagStreamParser` (in `schoolai/agent.py`) scant incrementeel op `<READ_FILE>` en
     `<WRITE_FILE>` tags en zorgt dat gesplitste tags niet vroegtijdig uitlekken.
   - Uitgevoerde reads/writes gaan via `schoolai/security.py`, die padtraversal en symlink-escapes
     blokkeert en harde caps oplegt op lees- en schrijfgroottes.

## Hoe het geheel werkt (flow)

1. GUI: gebruiker selecteert een workspace-map en logt in (handmatig of via automatische Login).
2. Gebruiker typt een verzoek en drukt Send.
3. `AgentHarness` bouwt een verborgen prompt met `SYSTEM_INSTRUCTIONS` (bevat de tag-taal)
   en verstuurt dit als `question` naar de NebulaONE endpoint via `NebulaClient.stream(...)`.
4. `NebulaClient` leest SSE-frames, decodeert base64 payloads en yieldt tekst-chunks.
5. `TagStreamParser` ontvangt chunks, emiteert zichtbare prose en bewaart voltooide tool-calls.
6. Wanneer een `<READ_FILE>` wordt gevraagd: `security.read_text_file()` wordt aangeroepen,
   het resultaat wordt in een systeemmessage (verborgen voor de modelprompt) teruggestuurd
   en de loop gaat door.
7. Wanneer een `<WRITE_FILE>` wordt gevraagd: `security.write_text_file()` schrijft het bestand
   met veilige flags (O_NOFOLLOW, O_NONBLOCK), en de UI toont een korte bevestiging
   (in Dutch: `✅ Bestand <pad> succesvol bewerkt`).
8. Loop stopt wanneer het model geen tool-tag meer uitstoot of na veiligheidslimieten
   (`MAX_TOOL_ROUNDS`, `MAX_TOOL_CALLS_PER_ROUND`).

## Hoe start je de app (kort)

1. Open een terminal en ga naar het project:

```bash
cd /Users/hamza/IdeaProjects/SchoolAIGUI
```

2. Maak en activeer de virtuele omgeving (of gebruik de `.venv/python` direct):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Installeer requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. (Optioneel) Installatie Playwright (alleen nodig voor automatische Login):

```bash
python -m pip install playwright
python -m playwright install chromium
```

5. Start de GUI:

```bash
export TK_SILENCE_DEPRECATION=1   # optioneel op macOS
python run.py
```

6. Login en token:
- Gebruik **Login to AI** (als Playwright is geïnstalleerd) of
- Plak de Authorization header (of JWT) uit je browser DevTools in het tokenveld
  en klik **Save token**.

## Troubleshooting (veelvoorkomend)

- ``ModuleNotFoundError: No module named 'requests'``:
  - Installeer dependencies (`pip install -r requirements.txt`) in dezelfde venv.

- Playwright issues:
  - Vergeet niet `python -m playwright install chromium` uit te voeren.
  - Playwright moet in dezelfde Python/venv geïnstalleerd zijn als de app.

- `tkinter` ontbreekt op macOS:
  - Gebruik een python.org Python build of installeer `python-tk` via Homebrew (soms problematisch).

- 401/403 van de API:
  - Token is ongeldig of verlopen. Verkrijg een verse token via DevTools en plak of herlogin.

- Automatische Login vangt geen token:
  - Site kan token in cookies/response bodies verbergen. In dat geval:
    - Open DevTools en zoek waar de token verschijnt (Network/Cookies/LocalStorage).
    - Geef die aanwijzing door zodat de capture-heuristiek aangepast kan worden.

## Veiligheidsopmerkingen

- De app **past bestandswijzigingen automatisch toe** wanneer de model een `<WRITE_FILE>` tag output.
  Houd de workspace onder versiebeheer zodat je wijzigingen kunt terugdraaien.
- Tokens zijn gevoelig. `~/.schoolai/config.json` wordt met permissie `0600` opgeslagen.
  Deel nooit je token in openbare chat.
- De security-module voorkomt padtraversal, escape via symlinks en beperkt lees-/schrijfgroottes,
  maar houdt geen garanties tegen lokale TOCTOU-aanvallen of kwaadaardige bestanden in de workspace.

## Wat kan er nog verbeterd worden (opties)

- Token-validatie endpoint-check: korte API-call na opslaan om direct "Token OK" te tonen.
- Uitbreiding van Playwright-capture om ook cookies en response bodies te inspecteren.
- Verbeterde UI status / token expiry countdown.
- Optionele bevestiging voor writes (veiligheidskeuze).

---

Als je wilt dat ik dit bestand onder een andere naam opsla (bijv. `README_UPDATED.md`) of het in de bestaande `README.md` opneem, zeg het dan. Wil je dat ik nog aanvullende details (bijv. voorbeeld-screenshots of exacte file-locaties met regels) toevoeg, dan werk ik het uit.
