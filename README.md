# Tiny Engine

This project uses a generative model to predict the next frame of a game based on the current frame and a player's action. It's served via a FastAPI backend and includes an interactive Next.js frontend.

## Setup and Installation

This project uses `uv` for Python package management.

1.  **Install `uv`**:
    If you don't have `uv` installed, follow the official installation instructions:
    ```bash
    # For macOS and Linux:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Virtual Environment**:
    ```bash
    uv venv
    ```
    This will create a `.venv` directory in your project folder.

3.  **Activate the Virtual Environment**:
    ```bash
    # For macOS and Linux:
    source .venv/bin/activate
    ```

4.  **Install Python Dependencies**:
    Install the required packages, including PyTorch from its specific download source.
    ```bash
    uv pip install --find-links https://download.pytorch.org/whl/cpu -e .
    ```

## Running the Application

You need to run the backend and frontend servers in two separate terminals.

**1. Start the Backend Server**:

Make sure your virtual environment is activated.

```bash
uv run python app.py
```
The backend will be available at `http://localhost:8000`.

**2. Start the Frontend Server**:

In a new terminal, navigate to the `frontend` directory.

```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at `http://localhost:3000`. You can now open this URL in your browser to play the game.
