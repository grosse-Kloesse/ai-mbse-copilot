from pathlib import Path

OUT = Path("data/raw/sample_mbse_v2.xml")

# keep it small/fast but more realistic
N = 60  # number of REQ/FUNC/BLK (each)

THEMES = [
    ("Over-voltage protection", "over-voltage", "prevent over-voltage conditions"),
    ("Under-voltage monitoring", "under-voltage", "detect undervoltage and trigger safe mode"),
    ("Thermal protection", "overheating", "limit temperature to avoid thermal runaway"),
    ("Vibration anomaly detection", "vibration", "detect abnormal vibration patterns"),
    ("Cooling control", "cooling", "control coolant flow and fan speed"),
    ("Insulation monitoring", "insulation", "detect insulation degradation and leakage"),
]

REQ_TPL = [
    "System shall {req_action}.",
    "The system shall ensure it can {req_action}.",
    "Requirement: the system must {req_action}.",
]
FUNC_TPL = [
    "Monitor {signal} and {func_action}.",
    "Detect {signal} and {func_action}.",
    "Measure {signal} then {func_action}.",
]
BLK_TPL = [
    "Implements {feature} and control logic.",
    "Provides {feature} with safety-oriented control.",
    "Handles {feature} and related diagnostics.",
]

# small noise to avoid pure-number matching
SIGNALS = {
    "over-voltage": ["bus voltage", "DC-link voltage", "supply voltage"],
    "under-voltage": ["battery voltage", "input voltage", "rail voltage"],
    "overheating": ["temperature", "module temperature", "coolant temperature"],
    "vibration": ["vibration RMS", "accelerometer signal", "bearing vibration"],
    "cooling": ["coolant flow", "fan speed", "coolant temperature"],
    "insulation": ["insulation resistance", "leakage current", "ground fault signal"],
}
ACTIONS = {
    "over-voltage": ["trigger protection", "activate clamp", "limit output"],
    "under-voltage": ["enter safe mode", "reduce load", "notify controller"],
    "overheating": ["reduce power", "enable cooling", "shut down safely"],
    "vibration": ["raise alarm", "schedule maintenance", "log anomaly"],
    "cooling": ["adjust fan", "control pump", "optimize cooling"],
    "insulation": ["raise warning", "isolate circuit", "log fault"],
}


def pick(lst, i):
    return lst[i % len(lst)]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<MBSEModel name="ToyMBSE-V2">')

    # Elements + Relations: REQ-i -> FUNC-i -> BLK-i
    for i in range(1, N + 1):
        theme_title, key, req_action = pick(THEMES, i - 1)

        rid = f"REQ-{i:03d}"
        fid = f"FUNC-{i:03d}"
        bid = f"BLK-{i:03d}"

        req_text = pick(REQ_TPL, i).format(req_action=req_action)
        signal = pick(SIGNALS[key], i)
        func_action = pick(ACTIONS[key], i)
        func_text = pick(FUNC_TPL, i + 1).format(signal=signal, func_action=func_action)
        blk_text = pick(BLK_TPL, i + 2).format(feature=theme_title.lower())

        # Slightly more realistic names
        lines.append(
            f'  <Element id="{rid}" type="Requirement" name="{theme_title} requirement" path="Model::Safety">{req_text}</Element>'
        )
        lines.append(
            f'  <Element id="{fid}" type="Function" name="{theme_title} function" path="Model::Functions">{func_text}</Element>'
        )
        lines.append(
            f'  <Element id="{bid}" type="Block" name="{theme_title} module" path="Model::Architecture">{blk_text}</Element>'
        )

        lines.append(f'  <Relation src="{rid}" dst="{fid}" type="refine" />')
        lines.append(f'  <Relation src="{fid}" dst="{bid}" type="satisfy" />')

    lines.append("</MBSEModel>")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", OUT, "lines:", len(lines))


if __name__ == "__main__":
    main()
