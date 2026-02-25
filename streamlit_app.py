# Streamlit Community Cloud entry point.
# Delegates to nylaris/ui.py so the app can be deployed from the repo root.
import importlib
import nylaris.ui as _ui
importlib.reload(_ui)
