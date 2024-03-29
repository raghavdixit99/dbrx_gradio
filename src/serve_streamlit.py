import shlex
import subprocess
from pathlib import Path
from modal import  Mount, web_server
from src.common import stub

streamlit_script_local_path = Path(__file__).parent.resolve()
# streamlit_script_remote_dir = Path("/root/")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = Mount.from_local_dir(
    streamlit_script_local_path,
    remote_path = '/root/src/'
)

@stub.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
)
@web_server(8000)
def run():
    target = shlex.quote('/root/src/app.py')
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)




