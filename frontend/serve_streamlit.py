from pathlib import Path

import modal

image = modal.Image.debian_slim().pip_install(
    "streamlit~=1.32.2", "pandas~=2.2.1", "plotly~=5.20.0", "PyPDF2"
)

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    remote_path=streamlit_script_remote_path,
)

streamlit_config_mount = modal.Mount.from_local_file(
    Path(__file__).parent / ".streamlit/config.toml",
    remote_path=Path("/root/.streamlit/config.toml"),
)

backend_mount = modal.Mount.from_local_python_packages("backend_hf")

stub = modal.Stub(name="streamlit_app",
                  image=image,
                  mounts=[streamlit_script_mount, streamlit_config_mount, backend_mount])


@stub.function(
    allow_concurrent_inputs=100,
    concurrency_limit=1,
    timeout=60 * 60,
)
@modal.web_server(8000)
def run():
    import shlex
    import subprocess

    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)