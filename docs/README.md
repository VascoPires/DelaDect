# Documentation

The user guide and API reference are in `source/`. Published documentation is
available at https://deladect.readthedocs.io.

From the repository root, in the Python 3.10 environment:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs/source docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open http://localhost:8000 in your browser.

Edit the `.rst` sources and rebuild to update the local pages. Sphinx generates
the API pages from the package source. Build output and generated API files
are excluded from version control.

Keep images and animations referenced by the guide in `source/_static/`.
Duplicate media, rendering intermediates and local figure exports are excluded
from version control.

For shift-correction usage, see the [user guide](source/shift_correction.rst).
