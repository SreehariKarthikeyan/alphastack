# Streamlit Community Cloud entry point.
# Delegates to nylaris/ui.py so the app can be deployed from the repo root.
import sys
import runpy
from pathlib import Path

# Ensure the repo root is on sys.path so `nylaris.*` imports resolve correctly.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Execute nylaris/ui.py as a fresh script on every Streamlit rerun.
# Using runpy.run_path avoids the module-cache problem that would cause
# st.set_page_config() to be called twice if we used import + reload.
runpy.run_path(str(_ROOT / "nylaris" / "ui.py"), run_name="__main__")
