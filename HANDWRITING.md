# Personal handwriting in Text2Note

The sample upload now extracts handwritten word images and uses the pretrained
[Handwriting Transformers](https://github.com/ankanbhunia/Handwriting-Transformers)
model to synthesize the requested English text. The model code and checkpoint
come from the [authors' HWT Space](https://huggingface.co/spaces/ankankbhunia/HWT).
The model is installed locally and runs on CPU. No generic handwriting font is
used for this path.
The personal handwriting path does not require an OpenAI API key.

## One-time setup on Windows

Run from the repository root with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-handwriting.txt
.\.venv\Scripts\python.exe scripts\setup_handwriting_model.py
```

Then start the existing Django backend with that environment:

```powershell
cd backend
..\.venv\Scripts\python.exe manage.py runserver
```

Upload a clear photo with several separated handwritten words in the existing
"Match my handwriting" control. The app stores the sample under
`Text2Note/handwriting_profiles/`, extracts style references, and uses the
returned style ID when generating the notebook image. The uploaded notebook
photo is analyzed separately to locate its ruling, so the personal renderer
uses the current page's lines. The editor shows line placement; the personal
handwriting itself appears after pressing Generate. Profiles and the downloaded
checkpoint are ignored by Git.

The current model covers its published Latin alphabet and punctuation set. The
API reports unsupported characters instead of silently changing the user's
text. The result imitates the supplied writing style, but a single sample
cannot guarantee an exact copy of every unseen letter or connection.
