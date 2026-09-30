1. Install the current multimodal toolchains
Cell 2 of 15
1. Install the current multimodal toolchains
Code cell 3
Cell 3 of 15
%pip install -q -U           "pyrit @ git+https://github.com/microsoft/PyRIT.git@main"           "garak @ git+https://github.com/NVIDIA/garak.git@main"           ipywidgets datasets soundfile librosa
Code cell 4
Cell 4 of 15
import garak
import pyrit

print("PyRIT:", pyrit.__version__)
print("garak:", garak.__version__)
2. Configure authorized targets
Cell 5 of 15
2. Configure authorized targets
The image test uses an OpenAI-compatible Chat Completions endpoint that accepts image input.
The audio test uses a Realtime WebSocket endpoint that accepts audio_path input. Examples:

OpenAI chat: https://api.openai.com/v1
Azure/OpenAI-compatible chat: your resource's /openai/v1 endpoint
OpenAI Realtime: wss://api.openai.com/v1
Azure Realtime: your resource's wss://.../openai/v1 endpoint
Keys are requested with getpass and are not written to the notebook or a config file.

Code cell 6
Cell 6 of 15
import getpass
import os

CHAT_ENDPOINT = input("Vision chat endpoint [https://api.openai.com/v1]: ").strip() or "https://api.openai.com/v1"
CHAT_MODEL = input("Vision model/deployment: ").strip()
CHAT_KEY = getpass.getpass("Vision chat API key: ")

REALTIME_ENDPOINT = input("Realtime endpoint [wss://api.openai.com/v1]: ").strip() or "wss://api.openai.com/v1"
REALTIME_MODEL = input("Realtime model/deployment: ").strip()
REALTIME_KEY = getpass.getpass("Realtime API key (Enter = reuse chat key): ") or CHAT_KEY

if not CHAT_MODEL or not REALTIME_MODEL:
    raise ValueError("Both model/deployment names are required.")

# PyRIT chat target.
os.environ["OPENAI_CHAT_ENDPOINT"] = CHAT_ENDPOINT
os.environ["OPENAI_CHAT_MODEL"] = CHAT_MODEL
os.environ["OPENAI_CHAT_KEY"] = CHAT_KEY

# PyRIT currently registers the Realtime target used by the Garak audio scenario
# under the name azure_openai_realtime. The endpoint determines the actual provider.
for prefix in ("OPENAI_REALTIME", "AZURE_OPENAI_REALTIME"):
    os.environ[f"{prefix}_ENDPOINT"] = REALTIME_ENDPOINT
    os.environ[f"{prefix}_API_KEY"] = REALTIME_KEY
    os.environ[f"{prefix}_MODEL"] = REALTIME_MODEL
os.environ["AZURE_OPENAI_REALTIME_UNDERLYING_MODEL"] = REALTIME_MODEL

# Native garak's OpenAI generator reads this variable.
os.environ["OPENAI_API_KEY"] = CHAT_KEY

print("Credentials loaded into this runtime only.")
3. Initialize PyRIT and verify multimodal targets
Cell 7 of 15
3. Initialize PyRIT and verify multimodal targets
Code cell 8
Cell 8 of 15
from pyrit.registry import TargetRegistry
from pyrit.setup import IN_MEMORY, initialize_pyrit_async
from pyrit.setup.initializers import ScorerInitializer, TargetInitializer, TechniqueInitializer

await initialize_pyrit_async(
    memory_db_type=IN_MEMORY,
    initializers=[TargetInitializer(), ScorerInitializer(), TechniqueInitializer()],
)

target_registry = TargetRegistry.get_registry_singleton()
registered_targets = target_registry.instances.get_names()
print("Registered targets:", registered_targets)

if "openai_chat" not in registered_targets:
    raise RuntimeError("openai_chat was not registered. Check CHAT_ENDPOINT, CHAT_MODEL, and CHAT_KEY.")
if "azure_openai_realtime" not in registered_targets:
    raise RuntimeError(
        "azure_openai_realtime was not registered. Check the Realtime WebSocket endpoint, model, and key."
    )
4. PyRIT runners for the Garak multimodal scenarios
Cell 9 of 15
4. PyRIT runners for the Garak multimodal scenarios
Code cell 10
Cell 10 of 15
from pyrit.output import output_scenario_async
from pyrit.scenario import DatasetAttackConfiguration
from pyrit.scenario.garak import FigStep
from pyrit.scenario.garak.audio_achilles_heel import (
    AudioAchillesHeel,
    AudioAchillesHeelDatasetConfiguration,
)


def require_target(name: str):
    target = TargetRegistry.get_registry_singleton().instances.get(name)
    if target is None:
        raise RuntimeError(f"Required target {name!r} is not registered.")
    return target


async def run_pyrit_garak_image(max_tests: int = 1, figstep_pro: bool = False):
    # Run a bounded FigStep or FigStep-Pro image scan.
    dataset_name = "figstep_pro" if figstep_pro else "figstep"
    scenario = FigStep()
    scenario.set_params_from_args(
        args={
            "objective_target": require_target("openai_chat"),
            "dataset_config": DatasetAttackConfiguration(
                dataset_names=[dataset_name],
                max_dataset_size=int(max_tests),
            ),
            "include_baseline": False,
        }
    )
    await scenario.initialize_async()
    print(f"Running {scenario.name}: {scenario.atomic_attack_count} atomic attack(s)")
    result = await scenario.run_async()
    await output_scenario_async(result)
    return result


async def run_pyrit_garak_audio(max_tests: int = 1):
    # Run a bounded Audio Achilles Heel scan against a Realtime target.
    scenario = AudioAchillesHeel()
    scenario.set_params_from_args(
        args={
            "objective_target": require_target("azure_openai_realtime"),
            "dataset_config": AudioAchillesHeelDatasetConfiguration(
                dataset_names=["garak_audio_achilles_heel"],
                max_dataset_size=int(max_tests),
            ),
        }
    )
    await scenario.initialize_async()
    print(f"Running {scenario.name}: {scenario.atomic_attack_count} atomic attack(s)")
    result = await scenario.run_async()
    await output_scenario_async(result)
    return result
5. Audio/image testing form
Cell 11 of 15
5. Audio/image testing form
Code cell 12
Cell 12 of 15
import asyncio
import ipywidgets as widgets
from IPython.display import display

modality = widgets.Dropdown(
    options=[
        ("Image — FigStep", "image"),
        ("Image — FigStep-Pro", "image_pro"),
        ("Audio — Achilles Heel", "audio"),
    ],
    description="Test:",
    style={"description_width": "initial"},
)
max_tests = widgets.BoundedIntText(value=1, min=1, max=20, description="Number of samples:")
authorization = widgets.Checkbox(
    value=False,
    description="I am authorized to test this endpoint",
    indent=False,
)
run_button = widgets.Button(description="Run security test", button_style="danger", icon="shield")
output = widgets.Output(layout={"border": "1px solid #bbb", "padding": "8px"})


async def run_selected_test():
    run_button.disabled = True
    with output:
        output.clear_output(wait=True)
        if not authorization.value:
            print("Confirm authorization before running a test.")
            run_button.disabled = False
            return
        try:
            if modality.value == "audio":
                await run_pyrit_garak_audio(max_tests.value)
            else:
                await run_pyrit_garak_image(
                    max_tests.value,
                    figstep_pro=(modality.value == "image_pro"),
                )
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")
            print("Check that the selected endpoint supports the requested modality and review the runtime log.")
        finally:
            run_button.disabled = False


def on_run_clicked(_):
    asyncio.create_task(run_selected_test())


run_button.on_click(on_run_clicked)
display(widgets.VBox([modality, max_tests, authorization, run_button, output]))
6. Optional native Garak comparison
Cell 13 of 15
6. Optional native Garak comparison
These commands run Garak directly against OpenAI's Chat Completions API. The selected model
must accept the requested modality. The audio probe downloads its full adversarial corpus and
may make many requests; keep RUN_NATIVE_AUDIO = False until you have reviewed cost, scope,
rate limits, and retention requirements.

Native Garak writes detailed JSONL/HTML reports under garak_runs/.

Code cell 14
Cell 14 of 15
import json
import subprocess
import sys
from pathlib import Path


def run_native_garak(probe: str, model: str, prompt_cap: int = 1):
    config_path = Path("/content/garak_multimodal_config.json")
    config_path.write_text(
        json.dumps(
            {
                "run": {"generations": 1, "soft_probe_prompt_cap": int(prompt_cap)},
                "system": {"lite": True, "parallel_attempts": False},
                "reporting": {"report_dir": "/content/garak_runs"},
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "garak",
        "--config",
        str(config_path),
        "--target_type",
        "openai",
        "--target_name",
        model,
        "--spec",
        probe,
        "--generations",
        "1",
    ]
    print("Running:", " ".join(command[:-1] + [command[-1]]))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    if process.wait() != 0:
        raise RuntimeError(f"garak exited with status {process.returncode}")


# Bounded visual comparison. FigStep uses Garak's SafeBench-Tiny catalog.
run_native_garak("probes.visual_jailbreak.FigStep", CHAT_MODEL, prompt_cap=1)

# Full-corpus audio comparison; explicitly opt in after reviewing the warning above.
RUN_NATIVE_AUDIO = False
GARAK_AUDIO_MODEL = ""  # e.g. an audio-capable Chat Completions model
if RUN_NATIVE_AUDIO:
    if not GARAK_AUDIO_MODEL:
        raise ValueError("Set GARAK_AUDIO_MODEL before enabling the native audio scan.")
    run_native_garak("probes.audio.AudioAchillesHeel", GARAK_AUDIO_MODEL, prompt_cap=1)
