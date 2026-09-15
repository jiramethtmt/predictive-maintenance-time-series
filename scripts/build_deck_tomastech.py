import os
import sys

SK = os.environ.get(
    "TOMASTECH_DECK_SKILL",
    os.path.expanduser("~/.claude/plugins/cache/tomastech/tomastc-plugin/0.19.0/skills/create-powerpoint"),
)
sys.path.insert(0, os.path.join(SK, "scripts"))
sys.path.insert(0, os.path.join(SK, "helpers"))

from deckkit import Deck, clear_text, para, font, PP_ALIGN, MSO_ANCHOR, RGBColor
import tomastech_deck as HT

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/predictive-maintenance-tomastech.pptx"

REPO = "https://github.com/jiramethtmt/predictive-maintenance-time-series"
NOTEBOOK = f"https://nbviewer.org/github/jiramethtmt/predictive-maintenance-time-series/blob/main/notebooks/01_eda_scania_component_x.ipynb"
DOI = "https://doi.org/10.5878/jvb5-d390"

D = Deck(os.path.join(SK, "assets", "tomastech-template.pptx"), script_font="Segoe UI")
HT.init(D)
BLUE, CYAN, INK, MUT, NAVY, SURFACE, WHITE = HT.BLUE, HT.CYAN, HT.INK, HT.MUT, HT.NAVY, HT.SURFACE, HT.WHITE
S = D.slides


def drop_empty_placeholders(s):
    # A repurposed template slide keeps its placeholders; an empty one renders the
    # layout's sample prompt text on top of the composed content.
    for shape in list(s.shapes):
        if shape.is_placeholder and shape.has_text_frame and not shape.text_frame.text.strip():
            shape._element.getparent().remove(shape._element)


def link_block(s, y, label, caption, url):
    HT.blk(s, 0.7, y, 11.9, 1.15, SURFACE)
    HT.blk(s, 0.7, y, 0.16, 1.15, CYAN)
    frame = HT.txt(s, 1.15, y + 0.2, 11.0, 0.42, label, 17, BLUE, bold=True)
    frame.paragraphs[0].runs[0].hyperlink.address = url
    HT.txt(s, 1.15, y + 0.68, 11.0, 0.3, caption, 11, MUT)


def cost_bar(s, y, label, cost, fill, fg):
    scale = 6.6 / 56100.0
    HT.txt(s, 0.7, y + 0.12, 3.4, 0.4, label, 14, INK, bold=(fill is BLUE))
    HT.blk(s, 4.3, y, cost * scale, 0.62, fill)
    HT.txt(s, 4.3 + cost * scale + 0.15, y + 0.13, 1.6, 0.4, f"{cost:,}", 15, fg, bold=True)


s = S[0]
D.title(s, "Predicting component failure")
detail = [sh for sh in s.shapes if sh.name == "TextBox 2"]
if detail:
    tf = detail[0].text_frame
    tf.word_wrap = True
    p = clear_text(tf)
    r = p.add_run()
    r.text = "Calling trucks into the workshop before the part fails"
    font(r, 16, True, BLUE)
    para(tf, "SCANIA Component X, 33,000 trucks, IDA 2024 challenge cost matrix. Three-seed result.", 12, False, MUT, 0)

s = HT.content("What the model is asked", 2, slide=S[1])
drop_empty_placeholders(s)
HT.headline(s, "One truck, one decision.", y=1.15, size=34, color=INK)
HT.numbered_rows(s, [
    ("What we predict", "How close the monitored component is to failure, using only the readouts that truck has already sent home."),
    ("What comes out", "An action, not a score: call this truck in, or leave it running. Five classes, class 4 is 0 to 6 steps from failure."),
    ("What makes it hard", "Only 2.7% of the fleet is at risk. The rest are healthy and must not be disturbed."),
], y0=2.25)

s = HT.content("Why accuracy is the wrong target", 3, slide=S[2])
drop_empty_placeholders(s)
HT.headline(s, "The economics pick the model.", y=1.15, size=34, color=INK)
HT.block_grid(s, [
    ("Missing a failure costs 500", "The truck fails in service. This is the number the whole design is built around."),
    ("A wasted check costs 10", "A workshop inspection that finds nothing is cheap by comparison."),
    ("One catch pays for 50 alarms", "Low precision is correct here. Optimising precision on this loss destroys value."),
    ("Doing nothing scores 97%", "A model graded on accuracy predicts healthy for everyone and saves nothing."),
], y=2.3, h=1.7)

s = HT.content("Result", 4)
HT.lead(s, "Total challenge cost on the 5,045-truck test set. Lower is better.", y=1.15)
HT.txt(s, 0.7, 1.6, 7.0, 1.0, "35,469 \u00b1 849", 52, BLUE, bold=True)
HT.txt(s, 0.7, 2.62, 11.9, 0.4, "Mean of three seeds. 29% cheaper than checking every truck, catching 77% of the at-risk fleet.", 13, MUT)
cost_bar(s, 3.45, "Our model", 35469, BLUE, BLUE)
cost_bar(s, 4.35, "Check every truck", 49671, NAVY, INK)
cost_bar(s, 5.25, "Act on the most likely class", 56100, SURFACE, MUT)
HT.txt(s, 0.7, 6.3, 11.9, 0.4, "Most-likely-class collapses onto doing nothing: at a 2.7% positive rate the likeliest class is always healthy.", 12, MUT)

s = HT.content("What that buys the workshop", 5)
HT.lead(s, "At the tuned operating point.", y=1.15)
HT.split(s, [
    "The catch rate",
    "109 of 142 at-risk trucks",
    "about 2,100 of 5,045 called in",
    "33 trucks still missed",
], y=2.4)
for index, (head, body) in enumerate([
    ("Where the saving comes from", "Not inspecting the other 2,900 trucks, while still reaching most of the ones that were going to fail."),
    ("The price of certainty", "Checking all 5,045 catches every failure and costs 40% more than this model."),
    ("The residual", "Those 33 missed trucks are the honest cost of the model today, and where further work pays."),
]):
    yy = 2.4 + index * 1.42
    HT.blk(s, 5.65, yy, 6.95, 1.22, SURFACE)
    HT.blk(s, 5.65, yy, 0.14, 1.22, CYAN)
    HT.txt(s, 6.05, yy + 0.18, 6.2, 0.36, head, 15, INK, bold=True)
    HT.txt(s, 6.05, yy + 0.58, 6.2, 0.6, body, 11.5, MUT, line=1.2)

s = HT.content("Against the published results", 6)
HT.lead(s, "Every result published on this benchmark, by total test cost.", y=1.1)
HT.layered(s, [
    ("Our work, 35,469 \u00b1 849", "AutoGluon LightGBM, cost-optimal decision rule, mean of three seeds"),
    ("CatBoost, 36,724", "Same study's best test row, but not the model they selected"),
    ("XGBoost, 37,733", "The model the leading study selected. They used AutoGluon too, so the gap is protocol, not toolkit."),
    ("Check every truck, 49,671", "The trivial baseline, level with Carpentier et al."),
], y0=1.75)

s = HT.content("Why the number is reportable", 7)
HT.headline(s, "Protocol before result.", y=1.1, size=32, color=INK)
HT.timeline(s, [
    ("Three seeds, not one", "A single run on this data moves by more than most of the improvements teams chase on it."),
    ("The noise floor was measured", "Single-split tuning spread 1,668 across seeds. Pooling validation into training and tuning out-of-fold halved it to 849."),
    ("A leakage audit found something", "A backward fill let a scoring point read its own future. Removing it moved a flattering 35,925 to an honest 39,379."),
    ("Test was scored once", "At the end, after every decision had been fixed on training data."),
], y0=2.3)

s = HT.content("The binding constraint", 8)
HT.lead(s, "15 of the 142 at-risk test trucks sit in the top 100 by risk score.", y=1.1)
HT.headline(s, "Ranking, not thresholds.", y=1.55, size=32, color=INK)
HT.numbered_rows(s, [
    ("The cost win comes from alarming broadly", "The model raises about 2,100 alarms. It is not finding the failures sharply."),
    ("Under a daily cap of 100 trucks", "This model would find roughly one in ten failures, not eight in ten."),
    ("Better ordering is the next work", "That is a modelling problem, not more threshold tuning."),
], y0=2.75)

s = HT.content("Running it for real", 9)
HT.lead(s, "Batch scoring on each upload cycle, not a real-time alarm.", y=1.15)
for index, stage in enumerate(["New readouts arrive", "Features at the cut point", "Risk and cost rule", "Ranked work list"]):
    x = 0.7 + index * 3.15
    HT.blk(s, x, 2.3, 2.6, 1.45, BLUE if index % 2 == 0 else NAVY)
    HT.blk(s, x, 2.3, 0.12, 1.45, CYAN)
    HT.txt(s, x + 0.35, 2.62, 1.95, 0.85, stage, 14, HT.LIGHTTX, bold=True, line=1.2)
    if index < 3:
        HT.arrow(s, x + 2.72, 2.88, 0.32, 0.3, CYAN)
HT.blk(s, 0.7, 4.35, 11.9, 1.05, SURFACE)
HT.blk(s, 0.7, 4.35, 0.14, 1.05, CYAN)
HT.txt(s, 1.15, 4.52, 11.0, 0.35, "Every dispatched truck returns a label", 15, INK, bold=True)
HT.txt(s, 1.15, 4.92, 11.0, 0.3, "Found or not found feeds the next training round. Without that loop the model stops improving on day one.", 11.5, MUT)
HT.txt(s, 0.7, 5.75, 11.9, 1.0, "Two more things the rollout needs: the work list trimmed to what the workshop can absorb that day, and a reason a mechanic will accept, such as the counter that accelerated threefold over three cycles. Input drift is watched and retraining is scheduled, not incidental.", 12, MUT, line=1.35)

s = HT.content("What kind of model this is", 10)
HT.lead(s, "Discrete-time survival classification on panel data, not sequence forecasting.", y=1.15)
HT.block_grid(s, [
    ("History becomes features", "The readouts up to a scoring point are compressed into one row, and a boosted tree predicts the degradation class."),
    ("Rates, not levels", "The counters are cumulative and never reset, so wear speed over 3, 10 and 20 readout windows carries the signal."),
    ("89% of outcomes are censored", "Most trucks never failed inside the window, which is why a survival framing beats a plain binary classifier."),
    ("Folds grouped by vehicle", "Near-duplicate scoring points from one truck can never straddle a split."),
], y=2.3, h=1.7)

s = HT.content("Go deeper", 11)
HT.lead(s, "The analysis behind these numbers.", y=1.15)
link_block(s, 2.1, "Open the full EDA notebook", NOTEBOOK, NOTEBOOK)
link_block(s, 3.5, "Repository, code and reproduction steps", REPO, REPO)
link_block(s, 4.9, "SCANIA Component X dataset, CC BY 4.0", DOI, DOI)

D.order([S[0], S[1], S[2]] + [D.slides[i] for i in range(4, len(D.slides))] + [S[3]])
D.save(OUT)
print("saved", OUT, "slides", len(D.slides))